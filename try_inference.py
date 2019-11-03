import os
import torch
import numpy as np
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, BatchSampler, SequentialSampler, RandomSampler
from mfa import MFA
from utils import *
from matplotlib import pyplot as plt
from imageio import imwrite
from tqdm import tqdm

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

w = 64
model_dir = './models/celeba'
n_components = 100
n_factors = 10
batch_size = 128

# Load a trained model
model = MFA(n_components=n_components, n_features=w*w*3, n_factors=n_factors).to(device=device)
model.load_state_dict(torch.load(os.path.join(model_dir, 'model_mfa_sgd_c_100_l_10.pth')))

image = visualize_model(model)
plt.imshow(image)
plt.axis('off')
plt.show()

# Sampling from the model - generating new images
# rnd_samples, _ = model.sample(100, with_noise=False)
# rnd_samples = rnd_samples.cpu().numpy().reshape([-1, 3, w, w]).transpose([0, 2, 3, 1]).reshape([-1, w*w*3])
# mosaic = samples_to_mosaic(rnd_samples, image_shape=[w, w, 3])
# plt.imshow(mosaic)
# plt.show()

trans = transforms.Compose([transforms.Resize(w), transforms.ToTensor(), ReshapeTransform([-1])])
dataset = ImageFolder(root='/mnt/local/eitanrich/PhD/Datasets/CelebA/cropped', transform=trans)

# Likelihood computation
# loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)
# all_ll = []
# for batch_x, _ in tqdm(loader):
#     all_ll.append(model.log_prob(batch_x.to(device)))
# all_ll = torch.cat(all_ll, dim=0)
# ll_sorted = torch.argsort(all_ll).cpu().numpy()
#
# all_keys = [key for key in SequentialSampler(dataset)]
# outlier_samples, _ = zip(*[dataset[all_keys[ll_sorted[i]]] for i in range(100)])
# outlier_samples = torch.stack(outlier_samples).cpu().numpy().reshape([-1, 3, w, w]).transpose([0, 2, 3, 1]).reshape([-1, w*w*3])
# mosaic = samples_to_mosaic(outlier_samples, image_shape=[w, w, 3])
# plt.imshow(mosaic)
# plt.show()

# Reconstruction from the model
random_samples, _ = zip(*[dataset[k] for k in RandomSampler(dataset, replacement=True, num_samples=100)])
random_samples = torch.stack(random_samples)

inpainting = True
if inpainting:
    # Hide part of each image
    mask = np.ones([3, w, w], dtype=np.float32)
    mask[:, :, w//2:w*3//4] = 0
    mask = torch.from_numpy(mask.flatten()).reshape([1, -1])
    original_full_samples = random_samples.clone()
    random_samples *= mask
    used_features = torch.nonzero(mask.flatten()).flatten()
else:
    used_features = None

# reconstructed_samples = model.reconstruct(random_samples.to(device), sampled_features=used_features).cpu()
reconstructed_samples = model.conditional_reconstruct(random_samples.to(device), observed_features=used_features).cpu()
if inpainting:
    reconstructed_samples = random_samples * mask + reconstructed_samples * (1 - mask)
reconstructed_samples = reconstructed_samples.numpy().reshape([-1, 3, w, w]).transpose([0, 2, 3, 1]).reshape([-1, w*w*3])

original_samples = random_samples.numpy().reshape([-1, 3, w, w]).transpose([0, 2, 3, 1]).reshape([-1, w*w*3])
mosaic_original = samples_to_mosaic(original_samples, image_shape=[w, w, 3])
mosaic_recontructed = samples_to_mosaic(reconstructed_samples, image_shape=[w, w, 3])
imwrite(os.path.join('models/celeba/random_samples_original.jpg'), mosaic_original)
imwrite(os.path.join('models/celeba/random_samples_reconstructed.jpg'), mosaic_recontructed)
plt.imshow(mosaic_original)
plt.figure()
plt.imshow(mosaic_recontructed)
plt.show()


