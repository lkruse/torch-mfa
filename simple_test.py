#%%
import matplotlib.pyplot as plt
import torch

from torch.utils.data import DataLoader, Dataset

from mppca import MPPCA
from mfa import MFA
from my_mfa import LowRankMixtureModel

pastelBlue = "#0072B2"
pastelRed = "#F5615C"


def generate_test_data(n_samples, n_components, n_features, n_factors, noise_variance, seed=0):
    """
    Generate samples from a ground-truth latent variable model.

    Parameters:
    n_samples (int):    total number of samples to draw (alias: n)
    n_components (int): number of mixture components (alias: K)
    n_features (int):   number of input dimensions (alias: d)
    n_factors (int):    number of underlying factors (alias: l)
    sigma2: (float):    noise variance

    Returns:
    samples (torch.Tensor): [n x d] tensor of generated samples
    """
    torch.manual_seed(seed)  # for reproducibility
    # generate random mixture parameters
    W = torch.randn((n_components, n_features, n_factors))
    mu = torch.randn((n_components, n_features))
    pi = torch.ones((n_components,)) / n_components
    sigma2 = torch.full((n_components,), noise_variance)
    # sample random mixture components to generate each data point
    sampled_components = torch.multinomial(pi, n_samples, replacement=True)
    # sample latent variables and noise
    z_l = torch.randn(n_samples, n_factors)
    z_d = torch.randn(n_samples, n_features)  
    # draw from generative model
    Wz = W[sampled_components] @ z_l[..., None]
    mu = mu[sampled_components][..., None]
    epsilon = (z_d * sigma2[sampled_components][..., None]**0.5)[..., None]
    samples = Wz + mu + epsilon

    return samples.squeeze()


def plot_data(n_features, X, axes, color=pastelBlue):
    """
    Plot samples from an MPPCA model.

    Parameters:
    n_features (int): number of input dimensions (alias: d)
    X (torch.Tensor): [n x d] tensor of data samples
    axes (np.array): array of matplotlib Axes objects
    color (str): hex color code
    """
    for i in range(n_features):
        for j in range(n_features):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            if i == j:
                axes[i, j].text(0.5, 0.5, f'Dim {i+1}', ha='center', va='center', fontsize=12)
            else:
                axes[i, j].scatter(X[:, j], X[:, i], alpha=0.5, color=color)

    plt.subplots_adjust(wspace=0.1, hspace=0.1)

class ToyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

#if __name__ == "__main__":

#%%
# define constants
seed = 0
n_samples = 1000
n_features = 5
n_factors = 2
n_components = 2
noise_variance = 0.01
n_plot = 500

mfa_sgd_epochs = 200
max_iterations = 1
feature_sampling = False
init_method = 'rnd_samples'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = 'cpu'

X_train = generate_test_data(n_samples, n_components, n_features, n_factors, noise_variance, seed)
X_train.to(device)
y = torch.randint(10, (n_samples,))

#model = MFA(n_components=n_components, n_features=n_features, n_factors=n_factors, init_method=init_method).to(device)
model = LowRankMixtureModel(n_components=n_components, n_features=n_features, n_factors=n_factors, init_method=init_method).to(device)

dataset = ToyDataset(X_train, y)

#ll_log = model.fit(X_train, max_iterations=250, feature_sampling=feature_sampling)
ll_log = model.batch_fit(dataset, max_iterations=max_iterations, batch_size=256, feature_sampling=feature_sampling)

if mfa_sgd_epochs > 0:
    print('Continuing training using SGD with diagonal (instead of isotropic) noise covariance...')
    ll_log_sgd = model.sgd_mfa_train(dataset, test_size=256, max_epochs=mfa_sgd_epochs, learning_rate=0.01,
                                        feature_sampling=feature_sampling)
    ll_log += ll_log_sgd
    
# plot log-likelihood
plt.plot(ll_log, c = pastelBlue)
plt.xlabel("EM iteration")
plt.ylabel("Log-Likelihood")
plt.show()

# plot sample comparison
X_samples, _ = model.sample(n_plot, with_noise=True)
plot_features = 5
fig, axes = plt.subplots(plot_features, plot_features, figsize=(30, 15))
plot_data(plot_features, X_train[:n_plot], axes[:, :], color=pastelBlue)
plot_data(plot_features, X_samples[:n_plot], axes[:, :],  color=pastelRed)
fig.text(0.3, 0.9, 'Truth Data', ha='left', va='center', fontsize=20)
fig.text(0.7, 0.9, 'Learned Model', ha='center', va='center', fontsize=20)
plt.savefig("model_eval.png")
plt.show()

# %%
