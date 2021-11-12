import torch
import torchvision
from torch.utils.data import ConcatDataset, TensorDataset, DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.transforms.functional import rotate

class MultipleDomainDataset:
    N_STEPS = 5001           # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    N_WORKERS = 8            # Default, subclasses may override
    ENVIRONMENTS = None      # Subclasses should override
    INPUT_SHAPE = None       # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)

class RMNIST(MultipleDomainDataset):
    def __init__(self, root, environments, num_picture):
        super(RMNIST, self).__init__()
        if root is None:
            raise ValueError('Data directory not specified!')

        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])

        original_dataset_tr = MNIST(root, train=True, download=True, transform=transform)
        original_dataset_te = MNIST(root, train=False, download=True, transform=transform)

        data = ConcatDataset([original_dataset_tr, original_dataset_te])
        original_images = torch.cat([img for img, _ in data])
        original_labels = torch.cat([torch.tensor(label).unsqueeze(0) for _, label in data])

        shuffle = torch.randperm(len(original_images))

        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []

        for i in range(len(environments)):
            images = original_images[i::len(environments)]
            labels = original_labels[i::len(environments)]
            self.datasets.append(self.rotate_dataset(images, labels, environments[i]))

        #self.input_shape = input_shape
        #self.num_classes = num_classes

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)

    def rotate_dataset(self, images, labels, angle):
        rotation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda x: rotate(x, angle, fill=(0,),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR)),
            transforms.ToTensor()])

        x = torch.zeros(len(images), 1, 28, 28)
        for i in range(len(images)):
            x[i] = rotation(images[i])
        y = labels.view(-1)
        return TensorDataset(x, y)

def get_mnist_loaders():
    environment = ['0','10','20','30','40','50','60','70']
    domain_list = []
    for i in environment:
        picture = RMNIST('../data/MNIST', int(i), 1)
        domain_list.append(picture)
    datasets = ConcatDataset(domain_list)
    # each time pickup one picture
    loader = DataLoader(datasets, batch_size=1, num_workers=4, pin_memory=True)
    return loader, loader, loader, loader