import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from base import BaseModel



class SignalCNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(SignalCNN, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.adaptive_avg_pool1d(x, 1)  # Aggregate the features (global pooling)
        x = x.view(x.size(0), -1)  # Flatten
        return x

class AggregateModel(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(AggregateModel, self).__init__()
        self.cnn = SignalCNN(input_channels, num_classes)

    def forward(self, signals):
        # signals is expected to be of shape (batch_size, num_signals, input_channels, signal_length)
        batch_size, num_signals, input_channels, signal_length = signals.size()
        
        # Flatten the signals to pass each one through the CNN
        signals = signals.view(-1, input_channels, signal_length)
        
        # Pass each signal through the CNN
        cnn_outs = self.cnn(signals)
        
        # Reshape to (batch_size, num_signals, -1)
        cnn_outs = cnn_outs.view(batch_size, num_signals, -1)
        
        # Aggregate using mean across the signal dimension
        aggregated_features = torch.mean(cnn_outs, dim=1)
        
        return aggregated_features


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=5)
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3)
        self.conv4 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3)
        self.conv5 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3)
        self.conv6 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3)
        self.conv7 = nn.Conv1d(in_channels=32, out_channels=256, kernel_size=3)
        self.conv8 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3)

        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.global_maxpool = nn.AdaptiveMaxPool1d(1)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)

        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.dropout1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.maxpool(x)
        x = self.dropout1(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.maxpool(x)
        x = self.dropout1(x)
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.global_maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout2(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class MLPClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLPClassifier, self).__init__()

        self.mlp = nn.Sequential(nn.Linear(input_size,64),
                                  nn.ReLU(),
                                  nn.Linear(64,32),
                                  nn.ReLU(),
                                  nn.Linear(32,output_size)
                                  )
    def forward(self,x):
        out  = self.mlp(x)
        return out


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, embed_size, bias=False)
        self.keys = nn.Linear(self.head_dim, embed_size, bias=False)
        self.queries = nn.Linear(self.head_dim, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.embed_size
        )

        out = self.fc_out(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, embed_size, forward_expansion):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, forward_expansion * embed_size)
        self.fc2 = nn.Linear(forward_expansion * embed_size, embed_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = FeedForward(embed_size, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_length):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_length, embed_size)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        N, T, _ = x.size()
        return x + self.encoding[:, :T, :].to(x.device)

class Transformer(nn.Module):
    def __init__(
        self,
        d,
        T,
        embed_size=64,
        output_size=None,
        num_layers=3,
        forward_expansion=4,
        heads=8,
        dropout=0.1,
        max_length=500,
    ):
        super(Transformer, self).__init__()
        self.positional_encoding = PositionalEncoding(embed_size, max_length)
        self.input_projection = nn.Linear(d, embed_size)

        self.encoder_layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    forward_expansion,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)
        if output_size is None:
            output_size = d
        self.fc_out = nn.Linear(embed_size, output_size)

    def forward(self, x):
        N, T, _ = x.shape
        x = self.input_projection(x)
        x = self.positional_encoding(x)

        for layer in self.encoder_layers:
            x = layer(x, x, x, mask=None)

        out = self.fc_out(x)
        return out

class BigModel(nn.Module):
    def __init__(self, feature_size):
        super().__init__()
        # self.flatten = nn.Flatten()

        self.hrv_conv = ConvNet()
        self.ecg_conv = ConvNet()

        self.mlp = MLPClassifier(feature_size + 2 * 32)

    def forward(self, x):
        out1 = self.ecg_conv(x["ecg_time_series"])
        out2 = self.hrv_conv(x["hrv_time_series"])

        features = x["features"]
        
        mlp_input = torch.cat([out1, out2, features],dim=-1)

        return self.mlp(mlp_input)


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SqueezeExcitation, self).__init__()
        self.fc1 = nn.Conv1d(in_channels, in_channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv1d(in_channels // reduction, in_channels, kernel_size=1)

    def forward(self, x):
        batch, channels, _ = x.size()
        out = F.adaptive_avg_pool1d(x, 1).view(batch, channels, 1)
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        return x * out

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, reduction=16, use_se=True):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SqueezeExcitation(out_channels, reduction) if use_se else nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet1D(nn.Module):
    def __init__(self, output_size=32):
        super(ResNet1D, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=11, stride=1, padding=5)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=3, padding=1)

        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1, use_se=True)
        self.layer2 = self._make_layer(64, 64, num_blocks=3, stride=3, use_se=True)
        self.layer3 = self._make_layer(64, 128, num_blocks=2, stride=1, use_se=True)
        self.layer4 = self._make_layer(128, 128, num_blocks=3, stride=3, use_se=True)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        # print("A")
        self.fc = nn.Linear(128, output_size) #output_size
        # print("B")

    def _make_layer(self, in_channels, out_channels, num_blocks, stride, use_se):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride=stride, use_se=use_se))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1, use_se=use_se))
        return nn.Sequential(*layers)

    def forward(self, x):
        # print(1)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.pool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.global_avg_pool(out)
        out = out.view(out.size(0), -1)
        # print(2)
        out = self.fc(out)
        # print(3)
        return out



class Empty(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return torch.empty(x.shape[:-1] + (0,)).to(x.device)


class SelfAttentionLayer(nn.Module):
    def __init__(self, feature_size):
        super(SelfAttentionLayer, self).__init__()
        self.feature_size = feature_size

        # Linear transformations for Q, K, V from the same source
        self.key = nn.Linear(feature_size, feature_size)
        self.query = nn.Linear(feature_size, feature_size)
        self.value = nn.Linear(feature_size, feature_size)

    def forward(self, x, mask=None):
        # Apply linear transformations
        keys = self.key(x)
        queries = self.query(x)
        values = self.value(x)

        # Scaled dot-product attention
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.feature_size, dtype=torch.float32))

        # Apply mask (if provided)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)

        # Multiply weights with values
        output = torch.matmul(attention_weights, values)

        return output, attention_weights

class AttentionClassifier(nn.Module):
  def __init__(self,input_size,num_classes, heads=1):
    super(AttentionClassifier, self).__init__()
    self.attention = SelfAttentionLayer(input_size)
    self.mlp_classifier = nn.Sequential(nn.Linear(input_size,64),
                                        nn.ReLU(),
                                        nn.Linear(64,32),
                                        nn.ReLU(),
                                        nn.Linear(32,num_classes))

  def forward(self,x):
    x, _ = self.attention(x)
    x = F.relu(x)
    return self.mlp_classifier(x)

class MyModel(BaseModel):
    def __init__(self, num_classes = None,
                 feature_size = None,
                 use_ecg_time_series=False,
                 use_hrv_time_series=False,
                 use_features=False,
                 use_transformer=False,
                 transformer_feature_size=None,
                 transformer_T=None):
        super().__init__()
        self.flatten = nn.Flatten()

        mlp_input_size = 0

        if use_hrv_time_series:
            self.hrv_conv = ConvNet() # ResNet1D()# ConvNet()
            mlp_input_size += 32
        else:
            self.hrv_conv = Empty()

        if use_ecg_time_series:
            self.ecg_conv = ConvNet() # ResNet1D() #ConvNet()
            mlp_input_size += 32
        else:
            self.ecg_conv = Empty()

        if use_features:
            mlp_input_size += feature_size
            self.features = nn.Identity()
        else:
            self.features = Empty()

        if use_transformer:
            Transformer(d=transformer_feature_size, T=transformer_T, embed_size=64, output_size= 32, num_layers=3, heads=8, forward_expansion=4, dropout=0.1, max_length=transformer_T)
            print("Transformer not implemented")
        
        #self.mlp = AttentionClassifier(mlp_input_size, num_classes)
        self.mlp = MLPClassifier(input_size=mlp_input_size, output_size=num_classes)#AttentionClassifier(mlp_input_size, num_classes)

    def forward(self, x):
        out1 = self.ecg_conv(x["ecg_time_series"])
        out2 = self.hrv_conv(x["hrv_time_series"])


        features = self.features(x["features"])
        
        mlp_input = torch.cat([self.flatten(out1), self.flatten(out2), features],dim=-1)

        return self.mlp(mlp_input)
