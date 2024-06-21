#%%
# packages
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import torch

# update plotting parameters
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = ['serif']
plt.rcParams['font.serif'] = ['Computer Modern']

# colorblind-friendly palette
pastelBlue = "#0072B2"
pastelRed = "#F5615C"
pastelGreen = "#009E73"
pastelPurple = "#8770FE"

def plot_ellipsoid(sigma, mu, ax, color='b', alpha = 0.3, n_points=100):
    # calculate eigenvalues and eigenvectors of the covariance matrix
    eigvals, eigvecs = torch.linalg.eigh(sigma)
    # sort eigenvalues and eigenvectors in descending order
    idx = eigvals.argsort(descending=True)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    # compute radii of the ellipsoid
    radii = torch.sqrt(eigvals)
    # generate points on a unit sphere
    u = torch.linspace(0, 2*torch.pi, n_points)
    v = torch.linspace(0, torch.pi, n_points)
    x = torch.outer(torch.cos(u), torch.sin(v))
    y = torch.outer(torch.sin(u), torch.sin(v))
    z = torch.outer(torch.ones(u.shape), torch.cos(v))
    # transform points to the shape of the ellipsoid
    points = torch.vstack([x.flatten(), y.flatten(), z.flatten()]).T @ \
        torch.diag(radii)
    points = points @ eigvecs.T + mu
    # reshape points to grid
    points = points.reshape((n_points, n_points, 3))
    # plot the ellipsoid
    ax.plot_surface(points[:,:,0], points[:,:,1], points[:,:,2], 
                    color=color, alpha = alpha)
    return ax


# %%

x1 = torch.tensor([1.0, 2.0, 5.0])
x2 = torch.tensor([3.0, 4.5, 7.0])

kde1 = gaussian_kde(x1, bw_method=0.25)
kde2 = gaussian_kde(x2, bw_method=0.25)

# Generate x values for plotting the KDEs
x_grid = np.linspace(0, 8, 1000)
kde1_values = kde1(x_grid)
kde2_values = kde2(x_grid)

plt.figure(figsize=(10, 6))

# Plot KDEs
plt.plot(x_grid, kde1_values, color=pastelBlue)
plt.plot(x_grid, kde2_values, color=pastelRed)

plt.fill_between(x_grid, kde1_values, alpha=0.3, color=pastelBlue)
plt.fill_between(x_grid, kde2_values, alpha=0.3, color=pastelRed)

# Mark data points
plt.scatter(x1, np.zeros_like(x1), marker='o', color=pastelBlue, s=100, label='Motion 1')
plt.scatter(x2, np.zeros_like(x2), marker='o', color=pastelRed, s=100, label='Motion 2')

# Add titles and labels
plt.title('KDE Plots with Data Points')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()

# Show plot
plt.savefig("kde.png", dpi = 600)

# %%

sigma1 = torch.eye(3)
sigma2 = torch.eye(3)

mu1 = x1
mu2 = x2

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


r1sig = torch.sqrt(-2*torch.log(torch.tensor(1-0.6827)))
r2sig = torch.sqrt(-2*torch.log(torch.tensor(1-0.9545)))
r3sig = torch.sqrt(-2*torch.log(torch.tensor(1-0.9973)))

ax = plot_ellipsoid(r1sig*sigma1, mu1, ax, alpha = 0.2, color=pastelBlue)
ax = plot_ellipsoid(r2sig*sigma1, mu1, ax, alpha = 0.2, color=pastelBlue)
ax = plot_ellipsoid(r3sig*sigma1, mu1, ax, alpha = 0.2, color=pastelBlue)


ax = plot_ellipsoid(r1sig*sigma2, mu2, ax, alpha = 0.2, color=pastelRed)
ax = plot_ellipsoid(r2sig*sigma2, mu2, ax, alpha = 0.2, color=pastelRed)
ax = plot_ellipsoid(r3sig*sigma2, mu2, ax, alpha = 0.2, color=pastelRed)

ax.scatter(mu1[0], mu1[1], mu1[2], color=pastelBlue, s=50, label = "Motion 1")
ax.scatter(mu2[0], mu2[1], mu2[2], color=pastelRed, s=50, label = "Motion 2")

ax.set_xlabel('$\mathbf{x}_1$'); ax.set_ylabel('$\mathbf{x}_2$')
ax.set_zlabel('$\mathbf{x}_3$')


#.set_aspect('equal')
ax.set_box_aspect(None, zoom=0.85)
plt.legend()
plt.savefig("gmm.png", bbox_inches='tight', dpi = 600)