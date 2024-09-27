#!/bin/bash
# Create a virtual environment
python3 -m venv venv
# Activate the virtual environment
source venv/bin/activate
# Upgrade pip (optional)
pip install --upgrade pip
# Install dependencies from requirements.txt (if exists)
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
fi
echo "Virtual environment setup complete!"
