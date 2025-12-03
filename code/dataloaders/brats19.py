# import os
# import torch
# import numpy as np
# from glob import glob
# from torch.utils.data import Dataset
# import h5py
# import itertools
# from torch.utils.data.sampler import Sampler
# from skimage import transform as sk_trans


# class BraTS2019(Dataset):
#     """ BraTS2019 Dataset """

#     def __init__(self, base_dir=None, split='train', num=None, transform=None):
#         self._base_dir = base_dir
#         self.transform = transform
#         self.sample_list = []

#         train_path = self._base_dir+'/train.txt'
#         test_path = self._base_dir+'/test.txt'

#         if split == 'train':
#             with open(train_path, 'r') as f:
#                 self.image_list = f.readlines()
#         elif split == 'test' or split == 'val':
#             with open(test_path, 'r') as f:
#                 self.image_list = f.readlines()

#         self.image_list = [item.replace('\n', '').split(",")[0] for item in self.image_list]
#         if num is not None:
#             self.image_list = self.image_list[:num]
#         print("total {} samples".format(len(self.image_list)))

#     def __len__(self):
#         return len(self.image_list)

#     def __getitem__(self, idx):
#         image_name = self.image_list[idx]
#         h5f = h5py.File(self._base_dir + "/data/{}.h5".format(image_name), 'r')
#         image = h5f['image'][:]
#         label = h5f['label'][:]
#         sample = {'image': image, 'label': label.astype(np.uint8)}
#         if self.transform:
#             sample = self.transform(sample)
#         return sample
    
    
# class BraTS20192(Dataset):
#     """ BraTS2019 Dataset """

#     def __init__(self, base_dir=None, split='train', num=None, transform=None):
#         self._base_dir = base_dir
#         self.transform = transform
#         self.sample_list = []

#         train_path = self._base_dir+'/train.txt'
#         test_path = self._base_dir+'/val2.txt'
#         # test_path = self._base_dir+'/test.txt'

#         if split == 'train':
#             with open(train_path, 'r') as f:
#                 self.image_list = f.readlines()
#         elif split == 'test' or split == 'val':
#             with open(test_path, 'r') as f:
#                 self.image_list = f.readlines()

#         self.image_list = [item.replace('\n', '').split(",")[0] for item in self.image_list]
#         if num is not None:
#             self.image_list = self.image_list[:num]
#         print("total {} samples".format(len(self.image_list)))

#     def __len__(self):
#         return len(self.image_list)

#     def __getitem__(self, idx):
#         image_name = self.image_list[idx]
#         h5f = h5py.File(self._base_dir + "/data/{}.h5".format(image_name), 'r')
#         image = h5f['image'][:]
#         label = h5f['label'][:]
#         sample = {'image': image, 'label': label.astype(np.uint8)}
#         if self.transform:
#             sample = self.transform(sample)
#         return sample, image_name

# class SagittalToAxial(object):
#     """
#     Convert the input 3D MRI images and masks from sagittal view to axial view.

#     Parameters:
#     -----------
#     sample : dict
#         A dictionary with 'image' and 'label' where:
#         - image: (H, W, D) [Sagittal view]
#         - label: (H, W, D) [Sagittal view]
    
#     Returns:
#     --------
#     sample : dict
#         A dictionary with 'image' and 'label' converted to axial view:
#         - image: (W, H, D) [Axial view]
#         - label: (W, H, D) [Axial view]
#     """
#     def __call__(self, sample):
#         image, label = sample['image'], sample['label']

#         # Verify the shapes are compatible
#         if image.shape != label.shape:
#             raise ValueError("Shape mismatch between image and label")

#         """To Coronal view"""
#         # # Convert image and label from sagittal to axial view
#         # # Sagittal view: (H, W, D)
#         # # Axial view: (W, H, D)
#         # image_axial = np.transpose(image, (1, 2, 0))  # Transpose (H, W, D) to (W, D, H)
#         # label_axial = np.transpose(label, (1, 2, 0))  # Transpose (H, W, D) to (W, D, H)
        
#         """To Axial view"""
#         # Convert image and label from coronal to axial view
#         # Coronal view: (H, W, D)
#         # Axial view: (D, W, H)
#         image_axial = np.transpose(image, (2, 1, 0))  # Transpose (H, W, D) to (D, W, H)
#         label_axial = np.transpose(label, (2, 1, 0))  # Transpose (H, W, D) to (D, W, H)


#         return {'image': image_axial, 'label': label_axial}



# class Resize(object):

#     def __init__(self, output_size):
#         self.output_size = output_size

#     def __call__(self, sample):
#         image, label = sample['image'], sample['label']
#         (w, h, d) = image.shape
#         label = label.astype(np.bool_)
#         image = sk_trans.resize(image, self.output_size, order = 1, mode = 'constant', cval = 0)
#         label = sk_trans.resize(label, self.output_size, order = 0)
#         assert(np.max(label) == 1 and np.min(label) == 0)
#         assert(np.unique(label).shape[0] == 2)
        
#         return {'image': image, 'label': label}

# class CenterCrop(object):
#     def __init__(self, output_size):
#         self.output_size = output_size

#     def __call__(self, sample):
#         image, label = sample['image'], sample['label']

#         # pad the sample if necessary
#         if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
#                 self.output_size[2]:
#             pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
#             ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
#             pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
#             image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)],
#                            mode='constant', constant_values=0)
#             label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)],
#                            mode='constant', constant_values=0)

#         (w, h, d) = image.shape

#         w1 = int(round((w - self.output_size[0]) / 2.))
#         h1 = int(round((h - self.output_size[1]) / 2.))
#         d1 = int(round((d - self.output_size[2]) / 2.))

#         label = label[w1:w1 + self.output_size[0], h1:h1 +
#                       self.output_size[1], d1:d1 + self.output_size[2]]
#         image = image[w1:w1 + self.output_size[0], h1:h1 +
#                       self.output_size[1], d1:d1 + self.output_size[2]]

#         return {'image': image, 'label': label}


# class RandomCrop(object):
#     """
#     Crop randomly the image in a sample
#     Args:
#     output_size (int): Desired output size
#     """

#     def __init__(self, output_size, with_sdf=False):
#         self.output_size = output_size
#         self.with_sdf = with_sdf

#     def __call__(self, sample):
#         image, label = sample['image'], sample['label']
#         if self.with_sdf:
#             sdf = sample['sdf']

#         # pad the sample if necessary
#         if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
#                 self.output_size[2]:
#             pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
#             ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
#             pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
#             image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)],
#                            mode='constant', constant_values=0)
#             label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)],
#                            mode='constant', constant_values=0)
#             if self.with_sdf:
#                 sdf = np.pad(sdf, [(pw, pw), (ph, ph), (pd, pd)],
#                              mode='constant', constant_values=0)

#         (w, h, d) = image.shape
#         # if np.random.uniform() > 0.33:
#         #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
#         #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
#         # else:
#         w1 = np.random.randint(0, w - self.output_size[0])
#         h1 = np.random.randint(0, h - self.output_size[1])
#         d1 = np.random.randint(0, d - self.output_size[2])

#         label = label[w1:w1 + self.output_size[0], h1:h1 +
#                       self.output_size[1], d1:d1 + self.output_size[2]]
#         image = image[w1:w1 + self.output_size[0], h1:h1 +
#                       self.output_size[1], d1:d1 + self.output_size[2]]
#         if self.with_sdf:
#             sdf = sdf[w1:w1 + self.output_size[0], h1:h1 +
#                       self.output_size[1], d1:d1 + self.output_size[2]]
#             return {'image': image, 'label': label, 'sdf': sdf}
#         else:
#             return {'image': image, 'label': label}


# class RandomRotFlip(object):
#     """
#     Crop randomly flip the dataset in a sample
#     Args:
#     output_size (int): Desired output size
#     """

#     def __call__(self, sample):
#         image, label = sample['image'], sample['label']
#         k = np.random.randint(0, 4)
#         image = np.rot90(image, k)
#         label = np.rot90(label, k)
#         axis = np.random.randint(0, 2)
#         image = np.flip(image, axis=axis).copy()
#         label = np.flip(label, axis=axis).copy()

#         return {'image': image, 'label': label}


# class RandomNoise(object):
#     def __init__(self, mu=0, sigma=0.1):
#         self.mu = mu
#         self.sigma = sigma

#     def __call__(self, sample):
#         image, label = sample['image'], sample['label']
#         noise = np.clip(self.sigma * np.random.randn(
#             image.shape[0], image.shape[1], image.shape[2]), -2*self.sigma, 2*self.sigma)
#         noise = noise + self.mu
#         image = image + noise
#         return {'image': image, 'label': label}


# class CreateOnehotLabel(object):
#     def __init__(self, num_classes):
#         self.num_classes = num_classes

#     def __call__(self, sample):
#         image, label = sample['image'], sample['label']
#         onehot_label = np.zeros(
#             (self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
#         for i in range(self.num_classes):
#             onehot_label[i, :, :, :] = (label == i).astype(np.float32)
#         return {'image': image, 'label': label, 'onehot_label': onehot_label}


# class ToTensor(object):
#     """Convert ndarrays in sample to Tensors."""

#     def __call__(self, sample):
#         image = sample['image']
#         image = image.reshape(
#             1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
#         if 'onehot_label' in sample:
#             return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
#                     'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
#         else:
#             return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long()}


# class TwoStreamBatchSampler(Sampler):
#     """Iterate two sets of indices

#     An 'epoch' is one iteration through the primary indices.
#     During the epoch, the secondary indices are iterated through
#     as many times as needed.
#     """

#     def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
#         self.primary_indices = primary_indices
#         self.secondary_indices = secondary_indices
#         self.secondary_batch_size = secondary_batch_size
#         self.primary_batch_size = batch_size - secondary_batch_size

#         assert len(self.primary_indices) >= self.primary_batch_size > 0
#         assert len(self.secondary_indices) >= self.secondary_batch_size > 0

#     def __iter__(self):
#         primary_iter = iterate_once(self.primary_indices)
#         secondary_iter = iterate_eternally(self.secondary_indices)
#         return (
#             primary_batch + secondary_batch
#             for (primary_batch, secondary_batch)
#             in zip(grouper(primary_iter, self.primary_batch_size),
#                    grouper(secondary_iter, self.secondary_batch_size))
#         )

#     def __len__(self):
#         return len(self.primary_indices) // self.primary_batch_size


# def iterate_once(iterable):
#     return np.random.permutation(iterable)


# def iterate_eternally(indices):
#     def infinite_shuffles():
#         while True:
#             yield np.random.permutation(indices)
#     return itertools.chain.from_iterable(infinite_shuffles())


# def grouper(iterable, n):
#     "Collect data into fixed-length chunks or blocks"
#     # grouper('ABCDEFG', 3) --> ABC DEF"
#     args = [iter(iterable)] * n
#     return zip(*args)

# if __name__ == "__main__":
#     from torchvision import transforms as T
#     import matplotlib.pyplot as plt

#     def plot_sample(sample):
#         image = sample['image'][0]  # Remove the channel dimension
#         label = sample['label']

#         fig, axes = plt.subplots(1, 2, figsize=(10, 5))
#         axes[0].imshow(image[:, :, image.shape[2] // 2], cmap='gray')
#         axes[0].set_title('MRI Image')
#         axes[1].imshow(label[:, :, label.shape[2] // 2], cmap='gray')
#         axes[1].set_title('Mask')
#         plt.savefig("./sample_images/brats19_sample.png")
#         plt.close()
#         # plt.show()

#     data_path = "/content/drive/MyDrive/Research/Dataset/data/BraTS19"
#     patch_size = (96, 96, 96)
#     dataset = BraTS2019(
#         base_dir=data_path, split='train', transform=T.Compose([
#             SagittalToAxial(),
#             RandomCrop(patch_size),
#             RandomRotFlip(),
#             ToTensor()
#         ]))
    
#     sample = dataset[0]
#     print(f"image shape: {sample['image'].shape}")
#     print(f"label shape: {sample['label'].shape}")

#     plot_sample(sample)




import os
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
import itertools
from torch.utils.data.sampler import Sampler
from skimage import transform as sk_trans


class BraTS2019(Dataset):
    """ BraTS2019 Dataset """

    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []

        train_path = self._base_dir+'/train.txt'
        test_path = self._base_dir+'/test.txt'

        if split == 'train':
            with open(train_path, 'r') as f:
                self.image_list = f.readlines()
        elif split == 'test' or split == 'val':
            with open(test_path, 'r') as f:
                self.image_list = f.readlines()

        self.image_list = [item.replace('\n', '').split(",")[0] for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    # def __getitem__(self, idx):
    #     image_name = self.image_list[idx]
    #     h5f = h5py.File(self._base_dir + "/data/{}.h5".format(image_name), 'r')
    #     image = h5f['image'][:]
    #     label = h5f['label'][:]
    #     sample = {'image': image, 'label': label.astype(np.uint8)}
    #     if self.transform:
    #         sample = self.transform(sample)
    #     return sample
    
    def __getitem__(self, idx):
        image_name = self.image_list[idx]

        # now: <base_dir>/<case>/<case>.h5
        h5_path = os.path.join(self._base_dir, image_name, f"{image_name}.h5")

        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"h5 file not found: {h5_path}")

        h5f = h5py.File(h5_path, 'r')
        image = h5f['image'][:]   # or 'flair'
        label = h5f['label'][:]   # or 'seg'
        sample = {'image': image, 'label': label.astype(np.uint8)}
        if self.transform:
            sample = self.transform(sample)
        return sample


    
class BraTS20192(Dataset):
    """ BraTS2019 Dataset """

    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []

        train_path = self._base_dir+'/train.txt'
        test_path = self._base_dir+'/val2.txt'
        # test_path = self._base_dir+'/test.txt'

        if split == 'train':
            with open(train_path, 'r') as f:
                self.image_list = f.readlines()
        elif split == 'test' or split == 'val':
            with open(test_path, 'r') as f:
                self.image_list = f.readlines()

        self.image_list = [item.replace('\n', '').split(",")[0] for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    # def __getitem__(self, idx):
    #     image_name = self.image_list[idx]
    #     h5f = h5py.File(self._base_dir + "/data/{}.h5".format(image_name), 'r')
    #     image = h5f['image'][:]
    #     label = h5f['label'][:]
    #     sample = {'image': image, 'label': label.astype(np.uint8)}
    #     if self.transform:
    #         sample = self.transform(sample)
    #     return sample, image_name
    
    def __getitem__(self, idx):
        image_name = self.image_list[idx]

        # now: <base_dir>/<case>/<case>.h5
        h5_path = os.path.join(self._base_dir, image_name, f"{image_name}.h5")

        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"h5 file not found: {h5_path}")

        h5f = h5py.File(h5_path, 'r')
        image = h5f['image'][:]   # or 'flair'
        label = h5f['label'][:]   # or 'seg'
        sample = {'image': image, 'label': label.astype(np.uint8)}
        if self.transform:
            sample = self.transform(sample)
        return sample



class SagittalToAxial(object):
    """
    Convert the input 3D MRI images and masks from sagittal view to axial view.

    Parameters:
    -----------
    sample : dict
        A dictionary with 'image' and 'label' where:
        - image: (H, W, D) [Sagittal view]
        - label: (H, W, D) [Sagittal view]
    
    Returns:
    --------
    sample : dict
        A dictionary with 'image' and 'label' converted to axial view:
        - image: (W, H, D) [Axial view]
        - label: (W, H, D) [Axial view]
    """
    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # Verify the shapes are compatible
        if image.shape != label.shape:
            raise ValueError("Shape mismatch between image and label")

        """To Coronal view"""
        # # Convert image and label from sagittal to axial view
        # # Sagittal view: (H, W, D)
        # # Axial view: (W, H, D)
        # image_axial = np.transpose(image, (1, 2, 0))  # Transpose (H, W, D) to (W, D, H)
        # label_axial = np.transpose(label, (1, 2, 0))  # Transpose (H, W, D) to (W, D, H)
        
        """To Axial view"""
        # Convert image and label from coronal to axial view
        # Coronal view: (H, W, D)
        # Axial view: (D, W, H)
        image_axial = np.transpose(image, (2, 1, 0))  # Transpose (H, W, D) to (D, W, H)
        label_axial = np.transpose(label, (2, 1, 0))  # Transpose (H, W, D) to (D, W, H)


        return {'image': image_axial, 'label': label_axial}



class Resize(object):

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        (w, h, d) = image.shape
        label = label.astype(np.bool_)
        image = sk_trans.resize(image, self.output_size, order = 1, mode = 'constant', cval = 0)
        label = sk_trans.resize(label, self.output_size, order = 0)
        assert(np.max(label) == 1 and np.min(label) == 0)
        assert(np.unique(label).shape[0] == 2)
        
        return {'image': image, 'label': label}

class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]

        return {'image': image, 'label': label}


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if self.with_sdf:
            sdf = sample['sdf']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)
            if self.with_sdf:
                sdf = np.pad(sdf, [(pw, pw), (ph, ph), (pd, pd)],
                             mode='constant', constant_values=0)

        (w, h, d) = image.shape
        # if np.random.uniform() > 0.33:
        #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
        #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
        # else:
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
        if self.with_sdf:
            sdf = sdf[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
            return {'image': image, 'label': label, 'sdf': sdf}
        else:
            return {'image': image, 'label': label}


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()

        return {'image': image, 'label': label}


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        noise = np.clip(self.sigma * np.random.randn(
            image.shape[0], image.shape[1], image.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        image = image + noise
        return {'image': image, 'label': label}


class CreateOnehotLabel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        onehot_label = np.zeros(
            (self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(np.float32)
        return {'image': image, 'label': label, 'onehot_label': onehot_label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        image = image.reshape(
            1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        if 'onehot_label' in sample:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        else:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long()}


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

class BalancedTwoStreamBatchSampler(Sampler):
    """
    Batch sampler that:

    - Uses *all* unlabeled samples once per epoch (up to integer division).
    - Cycles through labeled samples as needed (oversampling them).
    - Each batch has:
        labeled_bs labeled samples
        + (batch_size - labeled_bs) unlabeled samples.

    This is unlabeled-driven: epoch length is determined by the unlabeled pool.
    """

    def __init__(self, labeled_indices, unlabeled_indices, batch_size, labeled_bs):
        assert len(labeled_indices) > 0
        assert len(unlabeled_indices) > 0
        assert batch_size > labeled_bs > 0

        self.labeled_indices = np.array(labeled_indices)
        self.unlabeled_indices = np.array(unlabeled_indices)
        self.batch_size = batch_size
        self.labeled_bs = labeled_bs
        self.unlabeled_bs = batch_size - labeled_bs

        # Number of batches per epoch driven by unlabeled pool
        # (each unlabeled sample used at most once per epoch)
        self.num_batches = len(self.unlabeled_indices) // self.unlabeled_bs

    def __iter__(self):
        # Shuffle unlabeled indices once per epoch
        unlabeled_perm = np.random.permutation(self.unlabeled_indices)

        # Cycle through labeled indices indefinitely (oversampling allowed)
        labeled_cycle = itertools.cycle(np.random.permutation(self.labeled_indices))

        u_ptr = 0
        for _ in range(self.num_batches):
            batch = []

            # Labeled part (oversampled / cycled)
            for _ in range(self.labeled_bs):
                batch.append(int(next(labeled_cycle)))

            # Unlabeled part (each index used at most once per epoch)
            for _ in range(self.unlabeled_bs):
                batch.append(int(unlabeled_perm[u_ptr]))
                u_ptr += 1

            yield batch

    def __len__(self):
        # Number of batches per epoch is determined by unlabeled pool
        return self.num_batches



def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)

if __name__ == "__main__":
    from torchvision import transforms as T
    import matplotlib.pyplot as plt

    def plot_sample(sample):
        image = sample['image'][0]  # Remove the channel dimension
        label = sample['label']

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(image[:, :, image.shape[2] // 2], cmap='gray')
        axes[0].set_title('MRI Image')
        axes[1].imshow(label[:, :, label.shape[2] // 2], cmap='gray')
        axes[1].set_title('Mask')
        plt.savefig("/content/drive/MyDrive/Research/fig/brats19_sample.png")
        plt.close()
        # plt.show()

    data_path = "/content/drive/MyDrive/Research/Dataset/data/BraTS19"
    patch_size = (96, 96, 96)
    dataset = BraTS2019(
        base_dir=data_path, split='train', transform=T.Compose([
            SagittalToAxial(),
            RandomCrop(patch_size),
            RandomRotFlip(),
            ToTensor()
        ]))
    
    sample = dataset[0]
    print(f"image shape: {sample['image'].shape}")
    print(f"label shape: {sample['label'].shape}")

    plot_sample(sample)





