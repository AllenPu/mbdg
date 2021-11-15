# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import torch
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.datasets.folder
from torch.utils.data import TensorDataset, Subset, ConcatDataset
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms.functional import rotate

#from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
#from wilds.datasets.fmow_dataset import FMoWDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    # Debug
    "Debug28",
    "Debug224",
    # Small images
    "ColoredMNIST",
    "RotatedMNIST",
    # Big images
    "VLCS",
    "PACS",
    "OfficeHome",
    "TerraIncognita",
    "DomainNet",
    "SVIRO",
    # WILDS datasets
    "WILDSCamelyon",
    "WILDSFMoW",
    "EDGEvolCircle",
    "EDGRPlate",
    "EDGPortrait",
    "EDGForestCover"
]

def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


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


class Debug(MultipleDomainDataset):
    def __init__(self, root, test_envs, hparams):
        super().__init__()
        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2
        self.datasets = []
        for _ in [0, 1, 2]:
            self.datasets.append(
                TensorDataset(
                    torch.randn(16, *self.INPUT_SHAPE),
                    torch.randint(0, self.num_classes, (16,))
                )
            )

class Debug28(Debug):
    INPUT_SHAPE = (3, 28, 28)
    ENVIRONMENTS = ['0', '1', '2']

class Debug224(Debug):
    INPUT_SHAPE = (3, 224, 224)
    ENVIRONMENTS = ['0', '1', '2']


class MultipleEnvironmentMNIST(MultipleDomainDataset):
    def __init__(self, root, environments, dataset_transform, input_shape,
                 num_classes):
        super().__init__()
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
            self.datasets.append(dataset_transform(images, labels, environments[i]))

        self.input_shape = input_shape
        self.num_classes = num_classes


class ColoredMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['+90%', '+80%', '-90%']

    def __init__(self, root, test_envs, hparams):
        super(ColoredMNIST, self).__init__(root, [0.1, 0.2, 0.9],
                                         self.color_dataset, (2, 32, 32,), 2)

        self.input_shape = (2, 32, 32,)
        self.num_classes = 2

    def color_dataset(self, images, labels, environment):

        # Assign a binary label based on the digit
        labels = (labels < 5).float()

        # Flip label with probability 0.25
        labels = self.torch_xor_(labels, self.torch_bernoulli_(0.25, len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels, self.torch_bernoulli_(environment, len(labels)))
        images = torch.stack([images, images], dim=1)

        # Apply the color to the image by zeroing out the other color channel
        images[torch.tensor(range(len(images))), (1 - colors).long(), :, :] *= 0

        x = images.float() #.div_(255.0)
        y = labels.view(-1).long()

        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()


class RotatedMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['0', '15', '30', '45', '60', '75']

    def __init__(self, root, test_envs, hparams):
        super(RotatedMNIST, self).__init__(root, [0, 15, 30, 45, 60, 75],
                                           self.rotate_dataset, (1, 28, 28,), 10)

    def rotate_dataset(self, images, labels, angle):
        rotation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda x: rotate(x, angle, fill=(0,),
                resample=Image.BICUBIC)),
            transforms.ToTensor()])

        x = torch.zeros(len(images), 1, 32, 32)
        for i in range(len(images)):
            x[i] = rotation(images[i])

        y = labels.view(-1)

        return TensorDataset(x, y)


class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root, test_envs, augment, hparams):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)

        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            # transforms.Resize((224,224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []
        for i, environment in enumerate(environments):

            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            path = os.path.join(root, environment)
            env_dataset = ImageFolder(path,
                transform=env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = len(self.datasets[-1].classes)

class VLCS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["C", "L", "S", "V"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "VLCS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class PACS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "S"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "PACS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class DomainNet(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 1000
    ENVIRONMENTS = ["clip", "info", "paint", "quick", "real", "sketch"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "domain_net/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class OfficeHome(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "R"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "office_home/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class TerraIncognita(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["L100", "L38", "L43", "L46"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "terra_incognita/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class SVIRO(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["aclass", "escape", "hilux", "i3", "lexus", "tesla", "tiguan", "tucson", "x5", "zoe"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "sviro/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class WILDSEnvironment:
    def __init__(
            self,
            wilds_dataset,
            metadata_name,
            metadata_value,
            transform=None):
        self.name = metadata_name + "_" + str(metadata_value)

        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_array = wilds_dataset.metadata_array
        subset_indices = torch.where(
            metadata_array[:, metadata_index] == metadata_value)[0]

        self.dataset = wilds_dataset
        self.indices = subset_indices
        self.transform = transform

    def __getitem__(self, i):
        x = self.dataset.get_input(self.indices[i])
        if type(x).__name__ != "Image":
            x = Image.fromarray(x)

        y = self.dataset.y_array[self.indices[i]]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.indices)


class WILDSDataset(MultipleDomainDataset):
    INPUT_SHAPE = (3, 96, 96)
    def __init__(self, dataset, metadata_name, test_envs, augment, hparams):
        super().__init__()

        transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.RandomResizedCrop(96, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []

        for i, metadata_value in enumerate(
                self.metadata_values(dataset, metadata_name)):
            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            env_dataset = WILDSEnvironment(
                dataset, metadata_name, metadata_value, env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = dataset.n_classes

    def metadata_values(self, wilds_dataset, metadata_name):
        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_vals = wilds_dataset.metadata_array[:, metadata_index]
        return sorted(list(set(metadata_vals.view(-1).tolist())))


class WILDSCamelyon(WILDSDataset):
    ENVIRONMENTS = [ "hospital_0", "hospital_1", "hospital_2", "hospital_3",
            "hospital_4"]
    def __init__(self, root, test_envs, hparams):
        dataset = Camelyon17Dataset(root_dir=root)
        super().__init__(
            dataset, "hospital", test_envs, hparams['data_augmentation'], hparams)


class WILDSFMoW(WILDSDataset):
    ENVIRONMENTS = [ "region_0", "region_1", "region_2", "region_3",
            "region_4", "region_5"]
    def __init__(self, root, test_envs, hparams):
        dataset = FMoWDataset(root_dir=root)
        super().__init__(
            dataset, "region", test_envs, hparams['data_augmentation'], hparams)



#
# ADD NEW DATASET
#
class SimpleSyntheticDataset(MultipleDomainDataset):
    def __init__(self, data_dir, test_envs, hparams):
        self.data_dir = data_dir
        # load data
        data_pkl = self.load_data(data_dir)
        # config
        domain_num = len(list(set(data_pkl['domain'])))
        self.input_shape = data_pkl['data'][0].shape
        self.num_classes = len(list(set(data_pkl['label'])))
        self.ENVIRONMENTS = ['Domain '+ str(i) for i in range(domain_num)]
        # convert to torch Dataset
        self.datasets = []
        for d in range(domain_num):
            # get x, y from data_pkl
            idx = data_pkl['domain'] == d
            x = data_pkl['data'][idx].astype(np.float32)
            y = data_pkl['label'][idx].astype(np.int64)
            y = torch.tensor(y).view(-1).long()  # turn  1, 2, 3 to 0, 1, 2
            self.datasets.append(TensorDataset(torch.tensor(x).float(), y))

    def load_data(self, path=None):
        if not path: raise NotImplementedError
        return self.read_pickle(path)

    def read_pickle(self, name):
        with open(name, 'rb') as f:
            data = pickle.load(f)
        return data

#########
class EDGRPlate(SimpleSyntheticDataset):
    def __init__(self, data_dir, test_envs, hparams):
        super(EDGRPlate, self).__init__(
            '../datasets_for_domainbed/RPlate/data/RPlate.pkl',
            test_envs, hparams)


#########
class EDGEvolCircle(SimpleSyntheticDataset):
    def __init__(self, data_dir, test_envs, hparams):
        super(EDGEvolCircle, self).__init__(
            '../datasets_for_domainbed/toy-circle/data/half-circle.pkl',
            test_envs, hparams)

##########
class EDGForestCover(MultipleDomainDataset):
    def __init__(self, data_dir, test_envs, hparams):
        self.data_dir = data_dir

        COL = 'Elevation'
        MAX = 3451  # df[COL].max()
        MIN = 2061  # df[COL].min()
        COUNT = hparams['env_number']

        # pre
        self.datasets = []
        # df = self.load_forestcover_data().drop('Id', axis = 1)
        df = self.load_forestcover_data()
        # MAX = df[COL].max() # 3451 # df[col].max()
        # MIN = df[COL].min() # 2061 # df[col].min()
        bins = np.arange(MIN, MAX, (MAX - MIN)/COUNT)
        se1 = pd.cut(df[COL], bins)
        df = df.drop(COL, axis=1)
        gb = df.groupby(se1)
        gbs = [gb.get_group(x) for x in gb.groups]
        # groupby('Cover_Type').size()
        for each in gbs:
            print(each.groupby('label').size())
        gbs = [self.get_xy_from_df(each) for each in gbs]
        for x, y in gbs:
            y = torch.tensor(y).view(-1).long()  # turn  1, 2, 3 to 0, 1, 2
            # print(y)
            self.datasets.append(TensorDataset(torch.tensor(x).float(), y))
        self.input_shape = (54, )
        self.num_classes = 2
        self.ENVIRONMENTS = [str(i)
                             for i in range(COUNT-1)]
        return

    def load_forestcover_data(self, path='ForestCover/train.csv'):
        df = pd.read_csv(os.path.join(self.data_dir, path))
        df = df.rename(columns={"Cover_Type": "label"})
        df = self.group_labels(df)
        df = df.drop('Id', axis=1)
        df = df.sample(frac=1).reset_index(drop=True)
        df = df.sample(frac=1).reset_index(drop=False)  # [index, label]
        return df

    def group_labels(self, df):
        groups = [
            [0, 1, 6, 3],
            [4, 5, 2, 7],
        ]
        # print(df)

        def new_label(row):
            for new_l in range(len(groups)):
                if row['label'] in groups[new_l]:
                    return new_l
        df['label'] = df.apply(new_label, axis=1)
        # print(df)
        return df

    def get_xy_from_df(self, df):
        Y = df['label'].to_numpy()
        X = df.drop('label', axis='columns').to_numpy()
        return (X, Y)
