import os
import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from mfa import MFA
from utils import ReshapeTransform, samples_to_mosaic
import matplotlib
if os.name == 'posix' and "DISPLAY" not in os.environ:
    matplotlib.use('Agg')
from matplotlib import pyplot as plt
from imageio import imwrite

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

w = 64
model_dir = './models/celeba'
trans = transforms.Compose([transforms.Resize(w), transforms.ToTensor(), ReshapeTransform([-1])])
train_set = ImageFolder(root='/mnt/local/eitanrich/PhD/Datasets/CelebA/cropped', transform=trans)

for n_components in [200]:
    for n_factors in [10]:
        for batch_size in [1000]:   #[1000, 10000]:
            for try_num in range(1):
                model = MFA(n_components=n_components, n_features=w*w*3, n_factors=n_factors).to(device=device)

                print('EM fitting: {} components / {} factors / batch size {} / try {}...'.format(
                    n_components, n_factors, batch_size, try_num))

                ll_log = model.batch_fit(train_set, batch_size=batch_size, max_iterations=10, responsibility_sampling=0.2)

                plt.plot(ll_log, label='c{}_l{}_b{}'.format(n_components, n_factors, batch_size))
                plt.grid(True)
                plt.pause(0.1)
                rnd_samples, _ = model.sample(100, with_noise=False)
                rnd_samples = rnd_samples.cpu().numpy().reshape([-1, 3, w, w]).transpose([0, 2, 3, 1]).reshape([-1, w*w*3])
                mosaic = samples_to_mosaic(rnd_samples, image_shape=[w, w, 3])
                imwrite(os.path.join(model_dir, 'samples_c_{}_l_{}_b_{}_try_{}.jpg'.format(
                    n_components, n_factors, batch_size, try_num)), mosaic)
print('Done')
plt.savefig('celeba_batches.pdf')
plt.show()
