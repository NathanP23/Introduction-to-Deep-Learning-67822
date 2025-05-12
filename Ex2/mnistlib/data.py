# mnistlib/data.py
import torch, torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

TF_MNIST = transforms.Compose([transforms.ToTensor()])     # (1,28,28) in [0,1]

def get_mnist_loaders(bs: int = 256, root: str = "."):
    train_ds = torchvision.datasets.MNIST(root, True , download=True,  transform=TF_MNIST)
    test_ds  = torchvision.datasets.MNIST(root, False, download=True,  transform=TF_MNIST)
    kw = dict(num_workers=2, pin_memory=True)
    return (DataLoader(train_ds, bs, shuffle=True , **kw),
            DataLoader(test_ds , bs, shuffle=False, **kw))
