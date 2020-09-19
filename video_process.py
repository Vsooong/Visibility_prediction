import torchvision.transforms as T
import torch
import random
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from util_project import args
from torchvision.io import read_video_timestamps, read_video


class VideoDataset(Dataset):
    def __init__(self, transforms=None):
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(args.img_root_dir)))

    def __getitem__(self, idx):
        # print("image _idx", idx)
        img_path = os.path.join(args.img_root_dir, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        label = self.imgs[idx].split('_')[-1].split('.')[0]
        label = float(label) / 50
        if self.transforms is not None:
            img = self.transforms(img)
        label = torch.as_tensor(label, dtype=torch.float)
        return img, label

    def __len__(self):
        return len(self.imgs)


class VideoSequenceDataset(Dataset):
    def __init__(self, transforms=None, interval=60, wsize=10, stride=60):
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(args.img_root_dir)))
        self.interval, self.wsize, self.stride = interval, wsize, stride
        self.img_seq= self._build_video_seq()

    def __getitem__(self, idx):
        imgs=self.img_seq[idx]
        images=[]
        labels=[]
        for i in imgs:
            img_path = os.path.join(args.img_root_dir, i)
            img = Image.open(img_path).convert("RGB")
            label = self.imgs[idx].split('_')[-1].split('.')[0]
            label = float(label) / 50
            if self.transforms is not None:
                img = self.transforms(img)
            images.append(img)
            labels.append(torch.as_tensor(label))
        images=torch.stack(images,dim=0)
        labels=torch.stack(labels,dim=0)
        return images, labels

    def __len__(self):
        return len(self.img_seq)

    # x: (b, c, t, h, w)
    def _build_video_seq(self):
        img_data = []
        start_idx = 0
        length = len(self.imgs)
        while start_idx < length:
            end_idx = min(length, start_idx +self.interval* self.wsize,)
            excerpt = self.imgs[start_idx:end_idx]
            start=0
            record=[]
            while start + self.interval<=len(excerpt):
                record.append(excerpt[start])
                start += self.interval
            if len(record)==self.wsize:
                img_data.append(record)
            start_idx += self.stride
        return img_data
        # for idx in range(len(self.imgs)):
        #     time_stamp = int(self.imgs[idx].split('_')[1])
        #     img_data.append(time_stamp)
        #     label = self.imgs[idx].split('_')[-1].split('.')[0]
        #     label = float(label) / 50
        #     label = torch.as_tensor(label, dtype=torch.float)
        #     label_data.append(label)
        # return torch.as_tensor(img_data), torch.as_tensor(label_data)


# t,b,c,h,w

def get_transform(train=False):
    transforms = []
    transforms.append(T.Resize((args.img_height, args.img_width)))
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


if __name__ == '__main__':
    import torch

    device = args.device
    dataset = VideoSequenceDataset(get_transform(train=False))
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=args.workers,
    )
    for i,j in data_loader:
        print(i.size())
        print(j.size())

    # file='F:/data/visibility/data/Fog20200313000026.mp4'
    # pts,video_fps=read_video_timestamps(file)
    # print(pts)
    # print(video_fps)
    # for images, targets in data_loader:
    #     print(targets)
