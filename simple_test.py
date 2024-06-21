#%%
# packages
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset

from mfa import MFA

n_samples = 1000
n_latent = 2
n_features = 6
noise_variance = 0.1
n_components = 2

def generate_toy_data(n_samples=1000, n_features=6, n_latent=2, noise_variance=0.1, seed=0):
    torch.manual_seed(seed)  # for reproducibility
    # generate latent variables
    Z = torch.randn((n_samples, n_latent))
    # create a random linear transformation matrix
    W = torch.randn((n_latent, n_features))
    # create mean vector
    mu = torch.randn((n_features))
    # map latent variables to observed space
    X_clean = Z @ W + mu
    # add Gaussian noise
    noise = noise_variance * torch.randn((n_samples, n_features))
    X = X_clean + noise

    return X

#%%
# Generate the data
X = generate_toy_data(n_samples, n_features, n_latent, noise_variance)


# Create a figure and a set of subplots
fig, axes = plt.subplots(n_features, n_features, figsize=(15, 15))

# Loop through each subplot and plot the data
for i in range(n_features):
    for j in range(n_features):
        if i == j:
            axes[i, j].text(0.5, 0.5, f'Dim {i+1}', horizontalalignment='center', verticalalignment='center', fontsize=12)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
        else:
            axes[i, j].scatter(X[:, j], X[:, i], alpha=0.5)
            if j > 0:
                axes[i, j].set_yticks([])
            if i < n_features - 1:
                axes[i, j].set_xticks([])

# Adjust the layout
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.show()

# wrap the dataset in a PyTorch DataLoader
class ToyDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# Create a dataset and dataloader
dataset = ToyDataset(X)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

#%%
model = MFA(n_components=n_components, n_features=n_features, n_factors=n_latent,
            init_method=init_method).to(device=device)

#%%
# Example of iterating through the dataloader
for batch in dataloader:
    print(batch)
    break
