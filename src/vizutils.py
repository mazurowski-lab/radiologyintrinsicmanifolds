import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch

def plot_imgbatch(imgs):
    imgs = imgs.cpu()
    imgs = imgs.type(torch.IntTensor) 
    plt.figure(figsize=(15, 3*(imgs.shape[0])))
    grid_img = make_grid(imgs, nrow=5)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.show()