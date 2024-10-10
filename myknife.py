import torch.nn as nn
import torch
import numpy as np
# from .utils import FF
from model.my_model import ConvNet
"""
H(Y|Xe,Xh) = H(Y|phi,psi) + H(phi|Xe) - H(phi|Xe,Y) + H(psi|Xh) - H(psi|Xh,Y)
           = H(Y|phi,psi) + I(phi;Y|Xe) + I(psi;Y|Xh)
"""
class MyKNIFE(nn.Module):
    def __init__(self, phi_dim, y_cat):
        super(MyKNIFE, self).__init__()
        self.kernel_phi_kn_x = CondKernel(4, phi_dim, phi_dim)
        self.kernel_phi_kn_x_y = CondCondKernel(4,
                                                phi_dim,
                                                phi_dim,
                                                y_cat)

    def compute_entropies(self, x, phi, y):  # samples have shape [sample_size, dim]
        marg_ent = self.kernel_phi_kn_x(x,phi)
        cond_ent = self.kernel_phi_kn_x_y(x,phi,y)
        return marg_ent, cond_ent
    
    def forward(self, x, phi, y):  # samples have shape [sample_size, dim]
        marg_ent, cond_ent = self.compute_entropies(x,phi,y)               #kernel_cond(z_c, z_d)
        return marg_ent - cond_ent, marg_ent, cond_ent

    def learning_loss(self, x, phi, y):
        marg_ent, cond_ent = self.compute_entropies(x,phi,y)
        return marg_ent + cond_ent

    def I(self, *args, **kwargs):
        return self.forward(*args[:2], **kwargs)[0]


class MargKernel(nn.Module):
    def __init__(self, args, zc_dim, zd_dim, init_samples=None):
        self.conv = ConvNet()
        # self.optimize_mu = args.optimize_mu
        # self.K = args.marg_modes if self.optimize_mu else args.batch_size
        self.d = zc_dim
        # self.use_tanh = args.use_tanh
        # self.init_std = args.init_std
        super(MargKernel, self).__init__()

        self.logC = torch.tensor([-self.d / 2 * np.log(2 * np.pi)])

        if init_samples is None:
            init_samples = self.init_std * torch.randn(self.K, self.d)
        # self.means = nn.Parameter(torch.rand(self.K, self.d), requires_grad=True)
        if self.optimize_mu:
            self.means = nn.Parameter(init_samples, requires_grad=True)  # [K, db]
        else:
            self.means = nn.Parameter(init_samples, requires_grad=False)

        if args.cov_diagonal == 'var':
            diag = self.init_std * torch.randn((1, self.K, self.d))
        else:
            diag = self.init_std * torch.randn((1, 1, self.d))
        self.logvar = nn.Parameter(diag, requires_grad=True)

        if args.cov_off_diagonal == 'var':
            tri = self.init_std * torch.randn((1, self.K, self.d, self.d))
            tri = tri.to(init_samples.dtype)
            self.tri = nn.Parameter(tri, requires_grad=True)
        else:
            self.tri = None

        weigh = torch.ones((1, self.K))
        if args.average == 'var':
            self.weigh = nn.Parameter(weigh, requires_grad=True)
        else:
            self.weigh = nn.Parameter(weigh, requires_grad=False)

    def logpdf(self, x):
        assert len(x.shape) == 2 and x.shape[1] == self.d, 'x has to have shape [N, d]'
        x = x[:, None, :]
        w = torch.log_softmax(self.weigh, dim=1)
        y = x - self.means
        logvar = self.logvar
        if self.use_tanh:
            logvar = logvar.tanh()
        var = logvar.exp()
        y = y * var
        # print(f"Marg : {var.min()} | {var.max()} | {var.mean()}")
        if self.tri is not None:
            y = y + torch.squeeze(torch.matmul(torch.tril(self.tri, diagonal=-1), y[:, :, :, None]), 3)
        y = torch.sum(y ** 2, dim=2)

        y = -y / 2 + torch.sum(torch.log(torch.abs(var) + 1e-8), dim=-1) + w
        y = torch.logsumexp(y, dim=-1)
        return self.logC.to(y.device) + y

    def update_parameters(self, z):
        self.means = z

    def forward(self, x):
        y = -self.logpdf(x)
        return torch.mean(y)


class CondKernel(nn.Module):

    def __init__(self, cond_modes, zc_dim, phi_dim): #, heads=1):
        super(CondKernel, self).__init__()
        self.K, self.d = cond_modes, phi_dim
        # self.use_tanh = args.use_tanh
        self.logC = torch.tensor([-self.d / 2 * np.log(2 * np.pi)])

        self.convnet = ConvNet()
        self.mu = nn.Sequential(nn.ReLU(),nn.Linear(32,32),nn.ReLU(),nn.Linear(32,phi_dim),nn.ReLU(),nn.Linear(phi_dim,phi_dim*self.K))
        self.logvar = nn.Sequential(nn.ReLU(),nn.Linear(32,32),nn.ReLU(),nn.Linear(32,phi_dim),nn.ReLU(),nn.Linear(phi_dim,phi_dim*self.K),nn.Tanh())
        

        self.weight = nn.Sequential(nn.ReLU(),nn.Linear(32,32),nn.ReLU(),nn.Linear(32,self.K)) #FF(args, self.d, self.d, self.K)
        self.tri = None
        # if args.cov_off_diagonal == 'var':
        #     self.tri = FF(args, self.d, self.d, self.K * self.d * self.d)
        self.zc_dim = zc_dim

    def logpdf(self, x, z_d):  # H(X|Y)

        z_d = z_d[:, None, :]  # [N, 1, d]
        
        z_c = self.convnet(x)
        w = torch.log_softmax(self.weight(z_c), dim=-1)  # [N, K]
        mu = self.mu(z_c)
        logvar = self.logvar(z_c)

        var = logvar.exp().reshape(-1, self.K, self.d)
        mu = mu.reshape(-1, self.K, self.d)
        # print(f"Cond : {var.min()} | {var.max()} | {var.mean()}")
        # print(z_d.shape)
        # print(mu.shape)
        z = z_d - mu  # [N, K, d]
        z = var * z
        if self.tri is not None:
            tri = self.tri(z_c).reshape(-1, self.K, self.d, self.d)
            z = z + torch.squeeze(torch.matmul(torch.tril(tri, diagonal=-1), z[:, :, :, None]), 3)
        z = torch.sum(z ** 2, dim=-1)  # [N, K]

        z = -z / 2 + torch.log(torch.abs(var) + 1e-8).sum(-1) + w
        z = torch.logsumexp(z, dim=-1)
        return self.logC.to(z.device) + z

    def forward(self, z_c, z_d):
        z = -self.logpdf(z_c, z_d)
        return torch.mean(z)

class Heads(nn.Module):
    def __init__(self, heads=4, dim=32,out_dim = 32*4, use_tanh=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = 4
        self.use_tanh = use_tanh
        self.heads = nn.ModuleList([nn.Sequential(nn.ReLU(),nn.Linear(dim,dim),nn.ReLU(),nn.Linear(dim,dim),nn.ReLU(),nn.Linear(dim,out_dim)) for _ in range(heads)])  # 4 separate heads for each category

    def forward(self, x, y):
        y_output = torch.stack([self.heads[i](x) for i in range(4)], dim=1)  # Stack heads' outputs
        output = y_output[torch.arange(y.size(0)), y]  # Select output based on y
        if self.use_tanh:
            output = output.tanh()
        return output
    
class CondCondKernel(nn.Module):

    def __init__(self, cond_modes, zc_dim, phi_dim, heads=4):
        super(CondCondKernel, self).__init__()
        self.K, self.d = cond_modes, phi_dim
        # self.use_tanh = args.use_tanh
        self.logC = torch.tensor([-self.d / 2 * np.log(2 * np.pi)])

        self.convnet = ConvNet()
        self.mu = Heads(heads=heads,dim=phi_dim,out_dim=self.K*self.d,use_tanh=False)
        self.logvar = Heads(heads=heads,dim=phi_dim,out_dim=self.K*self.d,use_tanh=True)

        self.weight = Heads(heads=heads,dim=phi_dim,out_dim=self.K,use_tanh=False)
        self.tri = None
        # if args.cov_off_diagonal == 'var':
        #     self.tri = FF(args, self.d, self.d, self.K * self.d * self.d)
        self.zc_dim = zc_dim

    def logpdf(self, x, z_d, y):  # H(X|Y)

        z_d = z_d[:, None, :]  # [N, 1, d]
        
        z_c = self.convnet(x)
        w = torch.log_softmax(self.weight(z_c,y), dim=-1)  # [N, K]
        mu = self.mu(z_c,y)
        logvar = self.logvar(z_c,y)

        var = logvar.exp().reshape(-1, self.K, self.d)
        mu = mu.reshape(-1, self.K, self.d)
        # print(f"Cond : {var.min()} | {var.max()} | {var.mean()}")

        # print(z_d.shape)
        # print(mu.shape)
        z = z_d - mu  # [N, K, d]
        z = var * z
        if self.tri is not None:
            tri = self.tri(z_c).reshape(-1, self.K, self.d, self.d)
            z = z + torch.squeeze(torch.matmul(torch.tril(tri, diagonal=-1), z[:, :, :, None]), 3)
        z = torch.sum(z ** 2, dim=-1)  # [N, K]

        z = -z / 2 + torch.log(torch.abs(var) + 1e-8).sum(-1) + w
        z = torch.logsumexp(z, dim=-1)
        return self.logC.to(z.device) + z

    def forward(self, x, z_d,y):
        z = -self.logpdf(x, z_d,y)
        return torch.mean(z)