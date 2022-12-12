import random
import math
from torchvision import datasets, transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import re


class blockPermutationTransform(object):
    """8x8 blocks permutation"""
    def __init__(self, probability):
        self.probability = probability

    def __call__(self, x, blocksize=32):
        p = random.uniform(0, 1)
        if p >= self.probability:
            return x

        _, row, col = x.shape[0], x.shape[1], x.shape[2]
        blocks = x.reshape([3, -1, blocksize, blocksize])
        nums = [i for i in range(int(row *col /(blocksize *blocksize)))]
        # random.shuffle(nums)
        nums = self.shuffleNums(nums)
        blocksPermutation = blocks[:, nums, :, :]
        imgPermutation = blocksPermutation.reshape([3, row, col])
        return imgPermutation

    def shuffleNums(self, nums, ratio = 0.1):
        changesNum = int(ratio * len(nums) // 1)
        maxNum = len(nums)
        for i in range(0, changesNum):
            num1 = random.randint(0, maxNum -1)
            num2 = random.randint(0, maxNum -1)
            while num1 == num2:
                num2 = random.randint(0, maxNum -1)
            temp = nums[num1]
            nums[num1] = nums[num2]
            nums[num2] = temp
        return nums


class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    Origin: https://github.com/zhunzhong07/Random-Erasing
    '''

    def __init__(self, probability=0.3, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(20):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img


def build_transforms(is_train=True):
    if is_train:
        transform = transforms.Compose(
                        [
                            transforms.Resize((192,192)),
                            transforms.RandomHorizontalFlip(p=0.3),
                            transforms.RandomVerticalFlip(p=0.3),
#                             transforms.RandomRotation(5),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                            blockPermutationTransform(0.5),
                            RandomErasing(),
                        ])

    else:
        transform = transforms.Compose(
                        [
                            transforms.Resize((192,192)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                        ])

    return transform


class E2E_Dataset(Dataset):

    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path).convert('RGB')
#         p = Image.new('RGB', (192,192), (255,255,255))
#         p.paste(img)
#         if img.size[0] > img.size[1]:
#             img = img.transpose(Image.ROTATE_90)
        if self.transform is not None:
            im_1 = self.transform(img)

        # label = int(re.split(r'(\d+)', img_path.split('/')[-1])[1]) // 4
        label = int(img_path.split("/")[-1].split('.')[0].split('_')[0])

        return im_1, label

    def __len__(self):
        return len(self.file_list)
