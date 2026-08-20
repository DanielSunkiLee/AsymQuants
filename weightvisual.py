import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

# Create a sample Linear layer (e.g., 20 inputs, 10 outputs)
layer = nn.Linear(20, 10)
weights = layer.weight.detach().numpy() # Extract weight matrix

# Plot weight intensity matrix as a heatmap
plt.figure(figsize=(10, 4))
sns.heatmap(weights, cmap="coolwarm", center=0, annot=False)
plt.title("Neural Network Layer Weight Matrix")
plt.xlabel("Input Features")
plt.ylabel("Output Neurons")
plt.show()

