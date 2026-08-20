import torch
import torch.nn as nn

class CustomNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        # 1. Initialize the parent nn.Module class
        super().__init__() 
        
        # 2. Define layers (sub-modules)
        self.hidden = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.output = nn.Linear(64, output_dim)
        
    def forward(self, x):
        # 3. Define the forward pass logic
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x
# Instantiate the model
model = CustomNetwork(input_dim=10, output_dim=2)

# Pass data through the model (never call model.forward(x) directly)
dummy_input = torch.randn(1, 10)
output = model(dummy_input)
print(output)