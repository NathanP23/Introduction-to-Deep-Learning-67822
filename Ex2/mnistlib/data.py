# mnistlib/data.py
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

TF_MNIST = transforms.Compose([transforms.ToTensor()])     # (1,28,28) in [0,1]

def get_mnist_loaders(batch_size: int = 256, root: str = ".", num_workers: int = 0):
    train_ds = torchvision.datasets.MNIST(root, True , download=True,  transform=TF_MNIST)
    test_ds  = torchvision.datasets.MNIST(root, False, download=True,  transform=TF_MNIST)
    kw = dict(num_workers=num_workers, pin_memory=True)
    return (DataLoader(train_ds, batch_size, shuffle=True , **kw),
            DataLoader(test_ds, batch_size, shuffle=False, **kw))
