#%%
# packages
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset

from mfa import MFA

n_samples = 1000
n_features = 4
n_latent = 2
n_components = 2
noise_variance = 0.1

pastelBlue = "#0072B2"
pastelRed = "#F5615C"

def generate_mppca_data(n_samples=1000, n_features=6, n_latent=2, n_components=2, noise_variance=0.1, seed=0):
    torch.manual_seed(seed)  # for reproducibility
    # generate latent variables
    Z = torch.randn((n_samples, n_latent))
    # create a random linear transformation matrix for each component
    W = torch.randn((n_components, n_latent, n_features))
    # create a mean vector for each component
    mu = torch.randn((n_components, n_features))
    # define the mixture component weights (assumed uniform for example)
    mixture_weights = torch.ones((n_components,)) / n_components
    # sample random component assignments
    component_assignments = torch.multinomial(mixture_weights, n_samples, replacement=True)

    # generate data for each component
    X = torch.zeros(n_samples, n_features)
    for k in range(n_components):
        mask = (component_assignments == k)
        Z_k = Z[mask]
        if Z_k.shape[0] > 0:
            X_clean = torch.matmul(Z_k, W[k, ...]) + mu[k, :]
            noise = torch.randn(Z_k.size(0), n_features) * noise_variance
            X[mask] = X_clean + noise

    return X


def plot_data(n_features, X, color=pastelBlue):
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
                axes[i, j].scatter(X[:, j], X[:, i], alpha=0.5, color=color)
                if j > 0:
                    axes[i, j].set_yticks([])
                if i < n_features - 1:
                    axes[i, j].set_xticks([])

    # Adjust the layout
    plt.subplots_adjust(wspace=0.1, hspace=0.1)


X = generate_mppca_data(n_samples, n_features, n_latent, n_components, noise_variance, seed = 0)
plot_data(n_features, X[:500], color=pastelBlue)


#%%


# wrap the dataset in a PyTorch DataLoader
class ToyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Create a dataset and dataloader
y = torch.randint(10, (n_samples,))
dataset = ToyDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)




#%%
n_samples = 10000
X = generate_mppca_data(n_samples, n_features, n_latent, n_components, noise_variance, seed = 2)

plot_data(n_features, X[:500], color=pastelBlue)
plt.savefig("train.png")

y = torch.randint(4, (n_samples,))
train_dataset = ToyDataset(X, y)

X = generate_mppca_data(n_samples, n_features, n_latent, n_components, noise_variance, seed = 1)
y = torch.randint(4, (n_samples,))
test_dataset = ToyDataset(X, y)


model = MFA(n_components=n_components, n_features=n_features, n_factors=n_latent,
            init_method='kmeans')

#%%
ll_log = model.batch_fit(train_dataset, test_dataset, batch_size=100, test_size = 100, max_iterations=20,
                             feature_sampling=False)

plt.plot(ll_log)
# %%



Z = torch.randn((n_samples, n_latent))
# create a random linear transformation matrix for each component
W = model.A.transpose(1,2)
# create a mean vector for each component
mu = model.MU
# define the mixture component weights (assumed uniform for example)
mixture_weights = torch.softmax(model.PI_logits.data, dim=0)
# sample random component assignments
component_assignments = torch.multinomial(mixture_weights, n_samples, replacement=True)

# generate data for each component
X = torch.zeros(n_samples, n_features)
for k in range(n_components):
    mask = (component_assignments == k)
    Z_k = Z[mask]
    if Z_k.shape[0] > 0:
        X_clean = torch.matmul(Z_k, W[k, ...]) + mu[k, :]
        noise = torch.randn(Z_k.size(0), n_features) * noise_variance
        X[mask] = X_clean + noise

plot_data(n_features, X[:500], color=pastelRed)
plt.savefig("gen.png")
# %%
