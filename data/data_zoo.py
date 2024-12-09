from torch.utils.data import Dataset
from torchvision.datasets.folder import ImageFolder, default_loader
import os
import numpy as np

RSNA_MEAN = (0.24190991913821105, 0.24190991913821105, 0.24190991913821105)
RSNA_STD = (0.06858030350893875, 0.06858030350893875, 0.06858030350893875)

class RSNAData(ImageFolder):
    def __init__(self, root, transform=None, key='train', seperator='|', gender_filter=None, **kwargs):
        self.dataset_root = root
        self.loader = default_loader
        self.target_transform = None
        self.transform = transform
        train_list_path = os.path.join(self.dataset_root, 'rsna-train.csv')
        val_list_path = os.path.join(self.dataset_root, 'rsna-validation.csv')
        test_list_path = os.path.join(self.dataset_root, 'rsna-test.csv')

        self.label_mean = 118.9484375
        self.label_std = 50.01946962242396

        self.samples = []
        count = -1
        if key == 'train':
            with open(train_list_path, 'r') as f:
                for line in f:
                    count += 1
                    if count < 1:
                        continue
                    img_name, score, gender = line.strip('\n').split(seperator)
                    score = (float(score) - self.label_mean) / self.label_std
                    gender = self.convert_gender(gender)
                    if gender_filter is not None and gender[1] == gender_filter:
                        continue
                    self.samples.append((os.path.join(root, 'rsna-train', "{}.png".format(img_name)), np.float32(gender + [score])))
        elif key == 'val':
            with open(val_list_path, 'r') as f:
                for line in f:
                    count += 1
                    if count < 1:
                        continue
                    img_name, score, gender = line.strip('\n').split(seperator)
                    score = (float(score) - self.label_mean) / self.label_std
                    gender = self.convert_gender(gender)
                    if gender_filter is not None and gender[1] == gender_filter:
                        continue
                    self.samples.append((os.path.join(root, 'rsna-validation', "{}.png".format(img_name)), np.float32(gender + [score])))
        elif key == 'test':
            with open(test_list_path, 'r') as f:
                for line in f:
                    count += 1
                    if count < 1:
                        continue
                    img_name, score, gender = line.strip('\n').split(seperator)
                    score = (float(score) - self.label_mean) / self.label_std
                    gender = self.convert_gender(gender)
                    if gender_filter is not None and gender[1] == gender_filter:
                        continue
                    self.samples.append((os.path.join(root, 'rsna-test', "{}.png".format(img_name)), np.float32(gender + [score])))
        else:
            raise ValueError(f"Unsupported key: {key}")

    def convert_gender(self, gender):
        if gender.lower() == 'm' or gender.lower() == 'true':
            return [1, 0]
        else:
            return [0, 1]