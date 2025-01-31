import os
from torch.utils.data import Dataset
from PIL import Image

class TotalText(Dataset):

    # Base URL for the Total Text dataset
    __URL = os.path.join("datasets", "Total Text")
    # Image URLs from Base URL
    __IMAGE_URL = os.path.join(__URL, "totaltext", "Images")
    # Ground Truth URLs from Base URL
    __GT_URL = os.path.join(
        __URL, "groundtruth_textregion", "Text_Region_Mask")

    def __init__(self, train=False, transform=None):

        self.images = os.listdir(os.path.join(self.__IMAGE_URL, "Test"))
        self.img_url = os.path.join(self.__IMAGE_URL, "Test")
        self.gt_url = os.path.join(self.__GT_URL, "Test")
        self.length = len(self.images)

        if train:
            self.images = os.listdir(
                os.path.join(self.__IMAGE_URL, "Train"))
            self.img_url = os.path.join(self.__IMAGE_URL, "Train")
            self.gt_url = os.path.join(self.__GT_URL, "Train")
            self.length = len(self.images)

        self.transform = transform
        self.train = train

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        img_name: str = self.images[idx]
        gt_name: str = img_name.split(".")[0] + ".png"

        img = Image.open(os.path.join(self.img_url, img_name))
        gt = Image.open(os.path.join(self.gt_url, gt_name))

        if self.transform:
            img = self.transform(img)
            gt = self.transform(gt)

        return img, gt
