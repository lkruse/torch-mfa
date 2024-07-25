#%%
# packages
import matplotlib.pyplot as plt
import torch
from mppca import MPPCA
from mfa import MFA

from torch.utils.data import DataLoader, Dataset

n_samples = 100000
n_features = 4
n_latent = 2
n_components = 2
noise_variance = 0.01

pastelBlue = "#0072B2"
pastelRed = "#F5615C"

def generate_test_data(n_samples=1000, n_features=6, n_latent=2, n_components=2, noise_variance=0.01, seed=0):
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
            noise = torch.randn(Z_k.size(0), n_features) * noise_variance**0.5
            X[mask] = X_clean + noise
    '''
    pi = torch.ones((n_components,)) / n_components
    sampled_components = torch.multinomial(pi, n_samples, replacement=True)
    z_l = torch.randn(n, l, device=self.W.device)

    if with_noise:
        z_d = torch.randn(n, d, device=self.W.device)  
    else:
        z_d = torch.zeros(n, d, device=self.W.device)

    Wz = self.W[sampled_components] @ z_l[..., None]
    mu = self.mu[sampled_components][..., None]
    epsilon = (z_d * self.sigma2[sampled_components][..., None]**0.5)[..., None]

    samples = Wz + mu + epsilon
    '''

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


#X = generate_mppca_data(n_samples, n_features, n_latent, n_components, noise_variance, seed = 10)
#plot_data(n_features, X[:500], color=pastelBlue)
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

#%%

X_train = generate_test_data(n_samples, n_features, n_latent, n_components, noise_variance, seed = 2)
y_train = torch.randint(4, (n_samples,))

X_test = generate_test_data(n_samples, n_features, n_latent, n_components, noise_variance, seed = 1)
y_test = torch.randint(4, (n_samples,))

plot_data(n_features, X_train[:500], color=pastelBlue)
plt.savefig("train.png")
plt.show()

n_features = 500
n_latent = 10
n_components = 42

X_train = torch.randn((n_samples, n_features))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#model = MPPCA(n_components=n_components, n_features=n_features, n_factors=n_latent).to(device)
model = MFA(n_components=n_components, n_features=n_features, n_factors=n_latent).to("cpu")

X_train.to(device)



# Create a dataset and dataloader
dataset = ToyDataset(X_train, y)
dataloader = DataLoader(dataset, batch_size=500, shuffle=True, drop_last=True)



#ll_log = model.batch_fit(X_train, max_iterations=100)
ll_log = model.batch_fit(dataset, max_iterations=100, batch_size=1000)

plt.plot(ll_log)
plt.show()

# %%
'''
Z = torch.randn((n_samples, n_latent))
# create a random linear transformation matrix for each component
W = model.A.transpose(1,2)
# create a mean vector for each component
mu = model.MU
# define the mixture component weights (assumed uniform for example)
mixture_weights = model.pi
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

#plot_data(n_features, X[:500], color=pastelRed)
'''

sampled_X, _ = model.sample(500, with_noise=True)

plot_data(n_features, sampled_X, color=pastelRed)

plt.savefig("gen.png")
plt.show()


#%%
################################################################################
import json
import matplotlib.pyplot as plt
import torch

from mppca import MPPCA

pastelBlue = "#0072B2"
pastelRed = "#F5615C"


#%%
with open('data/train_data.json', 'r') as f:
    train_data = torch.tensor(json.load(f))
with open('data/train_labels.json', 'r') as f:
    train_labels = json.load(f)

#with open('data/test_data.json', 'r') as f:
#    test_data = torch.tensor(json.load(f))
#with open('data/test_labels.json', 'r') as f:
#    test_labels = json.load(f)
#%%
N, L, D = train_data.shape
flat_train = train_data.view(N, L*D)

# define constants
n_features = L*D
n_factors = 6
n_components = 42

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
flat_train.to(device)

model = MPPCA(n_components=n_components, n_features=n_features, n_factors=n_factors).to(device)

ll_log = model.fit(flat_train, max_iterations=100)

#X_samples, _ = model.sample(n_plot, with_noise=True)

# plot log-likelihood
plt.plot(ll_log, c = pastelBlue)
plt.xlabel("EM iteration")
plt.ylabel("Log-Likelihood")
plt.show()
# %%