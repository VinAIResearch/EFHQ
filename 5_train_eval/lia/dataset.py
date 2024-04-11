import glob
import os
import random

import numpy as np
import pandas as pd
import torch
from augmentations import AugmentationTransform
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from tqdm import tqdm


ImageFile.LOAD_TRUNCATED_IMAGES = True

placeholder = torch.zeros((3, 512, 512), dtype=torch.int32)


class Vox256(Dataset):
    def __init__(self, split, transform=None, augmentation=False):

        if split == "train":
            self.ds_path = "./datasets/vox-png/train"
        elif split == "test":
            self.ds_path = "./datasets/vox-png/test"
        else:
            raise NotImplementedError

        self.videos = [i for i in os.listdir(self.ds_path)]
        self.augmentation = augmentation

        if self.augmentation:
            self.aug = AugmentationTransform(False, True, True)
        else:
            self.aug = None

        self.transform = transform

    def __getitem__(self, idx):
        video_path = os.path.join(self.ds_path, self.videos[idx])
        frames_paths = sorted(glob.glob(video_path + "/*.png"))
        nframes = len(frames_paths)

        if nframes < 2:
            return None

        n1, n2 = random.sample(list(range(nframes)), 2)
        init_idx = [n1, n2]
        while True:
            try:
                img_source = Image.open(frames_paths[n1]).convert("RGB")
                break
            except:
                n1 = random.sample([i for i in range(nframes) if i not in init_idx], 1)[0]
                init_idx.append(n1)

        while True:
            try:
                img_target = Image.open(frames_paths[n2]).convert("RGB")
                break
            except:
                n2 = random.sample([i for i in range(nframes) if i not in init_idx], 1)[0]
                init_idx.append(n2)

        if self.augmentation:
            img_source, img_target = self.aug(img_source, img_target)

        if self.transform is not None:
            img_source = self.transform(img_source)
            img_target = self.transform(img_target)

            return img_source, img_target

    def __len__(self):
        return len(self.videos)


class Vox256_vox2german(Dataset):
    def __init__(self, transform=None):
        self.source_root = "./datasets/german/"
        self.driving_root = "./datasets/vox/test/"

        self.anno = pd.read_csv("pairs_annotations/german_vox.csv")

        self.source_imgs = os.listdir(self.source_root)
        self.transform = transform

    def __getitem__(self, idx):
        source_name = str("%03d" % self.anno["source"][idx])
        driving_name = self.anno["driving"][idx]

        source_vid_path = self.source_root + source_name
        driving_vid_path = self.driving_root + driving_name

        source_frame_path = sorted(glob.glob(source_vid_path + "/*.png"))[0]
        driving_frames_path = sorted(glob.glob(driving_vid_path + "/*.png"))[:100]

        source_img = self.transform(Image.open(source_frame_path).convert("RGB"))
        driving_vid = [self.transform(Image.open(p).convert("RGB")) for p in driving_frames_path]

        return source_img, driving_vid, source_name, driving_name

    def __len__(self):
        return len(self.source_imgs)


class Vox256_eval(Dataset):
    def __init__(self, transform=None):
        self.ds_path = "./datasets/vox-png/test/"
        self.videos = [i for i in os.listdir(self.ds_path)]
        self.transform = transform

    def __getitem__(self, idx):
        vid_name = self.videos[idx]
        video_path = os.path.join(self.ds_path, vid_name)

        frames_paths = sorted(glob.glob(video_path + "/*.png"))
        vid_target = [self.transform(Image.open(p).convert("RGB")) for p in frames_paths]

        return vid_name, vid_target

    def __len__(self):
        return len(self.videos)


class Vox256_cross(Dataset):
    def __init__(self, transform=None):
        self.ds_path = "./datasets/vox-png/test/"
        self.videos = os.listdir(self.ds_path)
        self.anno = pd.read_csv("pairs_annotations/vox256.csv")
        self.transform = transform

    def __getitem__(self, idx):
        source_name = self.anno["source"][idx]
        driving_name = self.anno["driving"][idx]

        source_vid_path = os.path.join(self.ds_path, source_name)
        driving_vid_path = os.path.join(self.ds_path, driving_name)

        source_frame_path = sorted(glob.glob(source_vid_path + "/*.png"))[0]
        driving_frames_path = sorted(glob.glob(driving_vid_path + "/*.png"))[:100]

        source_img = self.transform(Image.open(source_frame_path).convert("RGB"))
        driving_vid = [self.transform(Image.open(p).convert("RGB")) for p in driving_frames_path]

        return source_img, driving_vid, source_name, driving_name

    def __len__(self):
        return len(self.videos)


class Taichi(Dataset):
    def __init__(self, split, transform=None, augmentation=False):

        if split == "train":
            self.ds_path = "./datasets/taichi/train/"
        else:
            self.ds_path = "./datasets/taichi/test/"

        self.videos = os.listdir(self.ds_path)
        self.augmentation = augmentation

        if self.augmentation:
            self.aug = AugmentationTransform(True, True, True)
        else:
            self.aug = None

        self.transform = transform

    def __getitem__(self, idx):

        video_path = self.ds_path + self.videos[idx]
        frames_paths = sorted(glob.glob(video_path + "/*.png"))
        nframes = len(frames_paths)

        try:
            items = random.sample(list(range(nframes)), 2)
        except:
            print(video_path, nframes)
        img_source = Image.open(frames_paths[items[0]]).convert("RGB")
        img_target = Image.open(frames_paths[items[1]]).convert("RGB")

        if self.augmentation:
            img_source, img_target = self.aug(img_source, img_target)

        if self.transform is not None:
            img_source = self.transform(img_source)
            img_target = self.transform(img_target)

        return img_source, img_target

    def __len__(self):
        return len(self.videos)


class Taichi_eval(Dataset):
    def __init__(self, transform=None):
        self.ds_path = "./datasets/taichi/test/"
        self.videos = os.listdir(self.ds_path)
        self.transform = transform

    def __getitem__(self, idx):
        vid_name = self.videos[idx]
        video_path = os.path.join(self.ds_path, vid_name)

        frames_paths = sorted(glob.glob(video_path + "/*.png"))
        vid_target = [self.transform(Image.open(p).convert("RGB")) for p in frames_paths]

        return vid_name, vid_target

    def __len__(self):
        return len(self.videos)


class TED(Dataset):
    def __init__(self, split, transform=None, augmentation=False):

        if split == "train":
            self.ds_path = "./datasets/ted/train/"
        else:
            self.ds_path = "./datasets/ted/test/"

        self.videos = os.listdir(self.ds_path)
        self.augmentation = augmentation

        if self.augmentation:
            self.aug = AugmentationTransform(False, True, True)
        else:
            self.aug = None

        self.transform = transform

    def __getitem__(self, idx):
        video_path = os.path.join(self.ds_path, self.videos[idx])
        frames_paths = sorted(glob.glob(video_path + "/*.png"))
        nframes = len(frames_paths)

        items = random.sample(list(range(nframes)), 2)

        img_source = Image.open(frames_paths[items[0]]).convert("RGB")
        img_target = Image.open(frames_paths[items[1]]).convert("RGB")

        if self.augmentation:
            img_source, img_target = self.aug(img_source, img_target)

        if self.transform is not None:
            img_source = self.transform(img_source)
            img_target = self.transform(img_target)

        return img_source, img_target

    def __len__(self):
        return len(self.videos)


class TED_eval(Dataset):
    def __init__(self, transform=None):
        self.ds_path = "./datasets/ted/test/"
        self.videos = os.listdir(self.ds_path)
        self.transform = transform

    def __getitem__(self, idx):
        vid_name = self.videos[idx]
        video_path = os.path.join(self.ds_path, vid_name)

        frames_paths = sorted(glob.glob(video_path + "/*.png"))
        vid_target = [self.transform(Image.open(p).convert("RGB")) for p in frames_paths]

        return vid_name, vid_target

    def __len__(self):
        return len(self.videos)


class Ext256(Dataset):
    """Dataset of videos, each video can be represented as:

    - an image of concatenated frames
    - '.mp4' or '.gif'
    - folder with all frames
    """

    def __init__(self, data_list, split, skip, transform=None, augmentation=False):
        self.split = split
        self.data_list = data_list
        self.augmentation = augmentation
        if self.augmentation:
            self.aug = AugmentationTransform(False, True, True)
        else:
            self.aug = None
        self.transform = transform

        video_ids = []
        videos = {}
        self.root_dir = {}
        self.skip = skip

        for key, val in data_list.items():
            data_name = val["name"]
            root_dir = val["root_dir"]
            data_list = val["data_list_test"] if self.split == "test" else val["data_list"]

            video_ids_sub = []
            videos_sub = {}

            flag = False
            if os.path.exists(f"{data_name}_{split}_{skip}.pt"):
                checkpoint = torch.load(f"{data_name}_{split}_{skip}.pt")
                print(f"Load: {data_name}_{split}_{skip}.pt {len(checkpoint['video_ids'])}")
                video_ids += checkpoint["video_ids"]
                videos.update(checkpoint["videos"])
                self.root_dir[data_name] = root_dir
                flag = True

            if not flag:
                if data_name not in self.root_dir.keys():
                    self.root_dir[data_name] = root_dir
                f = open(data_list)
                file_list = f.readlines()

                for file_name in tqdm(file_list):
                    if "voxceleb1" in data_name:
                        img_name = file_name.strip().split("/")[1]
                        instance_id = file_name.strip().split("/")[0]
                    elif data_name == "vfhq":
                        (
                            img_path,
                            category,
                        ) = file_name.strip().split()  # B6vmGabgzH4_0_2/00000001.png frontal
                        img_name = img_path.split("/")[-1]  # 00000001.png
                        instance_id = img_path.split("/")[0]  # Clip+dA1MOwymy4o+P0+C0+F31904-32010
                    else:  # celeb
                        (
                            img_path,
                            category,
                        ) = file_name.strip().split()  # 58DPO_8Bd88/id0/58DPO_8Bd88_4/0000001.png frontal
                        img_name = img_path.split("/")[-1]  # 0000001.png
                        instance_id = "/".join(img_path.split("/")[:-1])  # 58DPO_8Bd88/id0/58DPO_8Bd88_4

                    if [data_name, instance_id] not in video_ids_sub:
                        video_ids_sub.append([data_name, instance_id])
                    if instance_id not in list(videos_sub.keys()):
                        videos_sub[instance_id] = {}
                        if data_name == "voxceleb1" or data_name == "voxceleb1_org":
                            videos_sub[instance_id]["all"] = []
                        else:
                            videos_sub[instance_id]["frontal"] = []
                            videos_sub[instance_id]["extreme"] = []

                    if data_name == "voxceleb1" or data_name == "voxceleb1_org":
                        videos_sub[instance_id]["all"].append(img_name)
                    else:
                        videos_sub[instance_id][category].append(img_name)
                f.close()
                video_ids += video_ids_sub
                videos.update(videos_sub)

                if not os.path.exists(f"{data_name}_{split}_{skip}.pt"):
                    print(f"Save: {data_name}_{split}_{skip}.pt")
                    ckpt = {
                        "video_ids": video_ids_sub,
                        "videos": videos_sub,
                        "root_dir": self.root_dir,
                    }
                    torch.save(ckpt, f"{data_name}_{split}_{skip}.pt")

        self.videos = video_ids
        self.video_dicts = videos
        print("Done init dataset")

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        data_name, instance_name = self.videos[idx]
        if "voxceleb1" in data_name:
            video_path = os.path.join(self.root_dir[data_name], self.split, instance_name)
        else:
            video_path = os.path.join(self.root_dir[data_name], instance_name)

        if "voxceleb1" in data_name:
            frames_paths = sorted(glob.glob(video_path + "/*.png"))
            if len(frames_paths) == 0:
                frames_paths = sorted(glob.glob(video_path + "/*.jpg"))
            nframes = len(frames_paths)

            if nframes < 2:
                return None, None

            n1, n2 = random.sample(list(range(nframes)), 2)
            init_idx = [n1, n2]
            while True:
                try:
                    img_source = Image.open(frames_paths[n1]).convert("RGB")
                    break
                except:
                    n1 = random.sample([i for i in range(nframes) if i not in init_idx], 1)[0]
                    init_idx.append(n1)

            while True:
                try:
                    img_target = Image.open(frames_paths[n2]).convert("RGB")
                    break
                except:
                    n2 = random.sample([i for i in range(nframes) if i not in init_idx], 1)[0]
                    init_idx.append(n2)
        else:
            frames = self.video_dicts[instance_name]
            num_frames_frontal = len(frames["frontal"])
            num_frames_extreme = len(frames["extreme"])

            flag = False
            if num_frames_extreme == 0 or (num_frames_frontal == 0 and self.skip):
                print([data_name, instance_name])
                return self.__getitem__(idx + 1) if self.split == "train" else None
            elif num_frames_frontal == 0 and not self.skip:
                flag = True
                frame_frontal_idx, frame_extreme_idx = np.sort(
                    np.random.choice(num_frames_extreme, replace=True, size=2)
                )
            else:
                frame_frontal_idx = np.sort(np.random.choice(num_frames_frontal, replace=True, size=1))
                frame_extreme_idx = np.sort(np.random.choice(num_frames_extreme, replace=True, size=1))

            frontal_info = (frame_frontal_idx.item(), "extreme") if flag else (frame_frontal_idx.item(), "frontal")
            extreme_info = (frame_extreme_idx.item(), "extreme")

            img_source = Image.open(os.path.join(video_path, frames[frontal_info[1]][frontal_info[0]])).convert("RGB")
            img_target = Image.open(os.path.join(video_path, frames[extreme_info[1]][extreme_info[0]])).convert("RGB")

        if self.augmentation:
            img_source, img_target = self.aug(img_source, img_target)

        if self.transform is not None:
            img_source = self.transform(img_source)
            img_target = self.transform(img_target)

        return img_source, img_target


class Ext256_eval(Dataset):
    """Dataset of videos, each video can be represented as:

    - an image of concatenated frames
    - '.mp4' or '.gif'
    - folder with all frames
    """

    def __init__(self, data_list, skip=False, transform=None):
        self.transform = transform
        self.split = split = "test"

        video_ids = []
        videos = {}
        self.root_dir = {}

        for key, val in data_list.items():
            data_name = val["name"]
            root_dir = val["root_dir"]
            data_list = val["data_list_test"]

            video_ids_sub = []
            videos_sub = {}

            flag = False
            if os.path.exists(f"{data_name}_{split}_{skip}.pt"):
                checkpoint = torch.load(f"{data_name}_{split}_{skip}.pt")
                print(f"Load: {data_name}_{split}_{skip}.pt {len(checkpoint['video_ids'])}")
                video_ids += checkpoint["video_ids"]
                videos.update(checkpoint["videos"])
                self.root_dir[data_name] = root_dir
                # self.root_dir.update(checkpoint['root_dir'])
                flag = True

            if not flag:
                if data_name not in self.root_dir.keys():
                    self.root_dir[data_name] = root_dir
                f = open(data_list)
                file_list = f.readlines()

                for file_name in file_list:
                    if "voxceleb1" in data_name:
                        img_name = file_name.strip().split("/")[1]
                        instance_id = file_name.strip().split("/")[0]
                    elif data_name == "vfhq":
                        (
                            img_path,
                            category,
                        ) = file_name.strip().split()  # B6vmGabgzH4_0_2/00000001.png frontal
                        img_name = img_path.split("/")[-1]  # 00000001.png
                        instance_id = img_path.split("/")[0]  # Clip+dA1MOwymy4o+P0+C0+F31904-32010
                    else:  # celeb
                        (
                            img_path,
                            category,
                        ) = file_name.strip().split()  # 58DPO_8Bd88/id0/58DPO_8Bd88_4/0000001.png frontal
                        img_name = img_path.split("/")[-1]  # 0000001.png
                        instance_id = "/".join(img_path.split("/")[:-1])  # 58DPO_8Bd88/id0/58DPO_8Bd88_4

                    if [data_name, instance_id] not in video_ids_sub:
                        video_ids_sub.append([data_name, instance_id])
                    if instance_id not in list(videos_sub.keys()):
                        videos_sub[instance_id] = {}
                        if data_name == "voxceleb1" or data_name == "voxceleb1_org":
                            videos_sub[instance_id]["all"] = []
                        else:
                            videos_sub[instance_id]["frontal"] = []
                            videos_sub[instance_id]["extreme"] = []

                    if data_name == "voxceleb1" or data_name == "voxceleb1_org":
                        videos_sub[instance_id]["all"].append(img_name)
                    else:
                        videos_sub[instance_id][category].append(img_name)
                f.close()
                video_ids += video_ids_sub
                videos.update(videos_sub)

                if not os.path.exists(f"{data_name}_{split}_{skip}.pt"):
                    print(f"Save: {data_name}_{split}_{skip}.pt")
                    ckpt = {
                        "video_ids": video_ids_sub,
                        "videos": videos_sub,
                        "root_dir": self.root_dir,
                    }
                    torch.save(ckpt, f"{data_name}_{split}_{skip}.pt")

        self.videos = video_ids
        self.video_dicts = videos
        print("Done init dataset")

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        np.random.seed(idx)
        data_name, instance_name = self.videos[idx]
        if "voxceleb1" in data_name:
            video_path = os.path.join(self.root_dir[data_name], self.split, instance_name)
        else:
            video_path = os.path.join(self.root_dir[data_name], instance_name)

        vid_target = []

        if "voxceleb1" in data_name:
            frames_paths = sorted(glob.glob(video_path + "/*.png"))
            vid_target = [self.transform(Image.open(p).convert("RGB")) for p in frames_paths]
        else:
            frames = self.video_dicts[instance_name]
            num_frames_frontal = len(frames["frontal"])
            num_frames_extreme = len(frames["extreme"])

            if num_frames_frontal == 0 or num_frames_extreme == 0:
                return {
                    "source": placeholder,
                    "driving": placeholder,
                    "name": placeholder,
                    "video": placeholder,
                }

            frame_frontal_idx = np.random.choice(num_frames_frontal, replace=True, size=1)
            img = self.transform(
                Image.open(os.path.join(video_path, frames["frontal"][frame_frontal_idx.item()])).convert("RGB")
            )
            vid_target.append(img)

            frames["extreme"] = sorted(frames["extreme"], key=lambda x: int(x.split("/")[-1].split(".")[0]))
            for name in frames["extreme"]:
                img_path = os.path.join(video_path, name)
                img = self.transform(Image.open(img_path).convert("RGB"))
                vid_target.append(img)

        return instance_name, vid_target


class Ext256_eval_cross(Dataset):
    """Dataset of videos, each video can be represented as:

    - an image of concatenated frames
    - '.mp4' or '.gif'
    - folder with all frames
    """

    def __init__(self, data_list, skip=False, transform=None):
        self.transform = transform
        self.split = split = "test"

        video_ids = []
        videos = {}
        self.root_dir = {}

        for key, val in data_list.items():
            data_name = val["name"]
            root_dir = val["root_dir"]
            data_list = val["data_list_test"]

            video_ids_sub = []
            videos_sub = {}

            flag = False
            if os.path.exists(f"{data_name}_{split}_{skip}.pt"):
                checkpoint = torch.load(f"{data_name}_{split}_{skip}.pt")
                print(f"Load: {data_name}_{split}_{skip}.pt {len(checkpoint['video_ids'])}")
                video_ids += checkpoint["video_ids"]
                videos.update(checkpoint["videos"])
                self.root_dir[data_name] = root_dir
                # self.root_dir.update(checkpoint['root_dir'])
                flag = True

            if not flag:
                if data_name not in self.root_dir.keys():
                    self.root_dir[data_name] = root_dir
                f = open(data_list)
                file_list = f.readlines()

                for file_name in file_list:
                    if "voxceleb1" in data_name:
                        img_name = file_name.strip().split("/")[1]
                        instance_id = file_name.strip().split("/")[0]
                    elif data_name == "vfhq":
                        (
                            img_path,
                            category,
                        ) = file_name.strip().split()  # B6vmGabgzH4_0_2/00000001.png frontal
                        img_name = img_path.split("/")[-1]  # 00000001.png
                        instance_id = img_path.split("/")[-2]  # B6vmGabgzH4_0_2
                    else:  # celeb
                        (
                            img_path,
                            category,
                        ) = file_name.strip().split()  # 58DPO_8Bd88/id0/58DPO_8Bd88_4/0000001.png frontal
                        img_name = img_path.split("/")[-1]  # 0000001.png
                        instance_id = "/".join(img_path.split("/")[:-1])  # 58DPO_8Bd88/id0/58DPO_8Bd88_4

                    if [data_name, instance_id] not in video_ids_sub:
                        video_ids_sub.append([data_name, instance_id])
                    if instance_id not in list(videos_sub.keys()):
                        videos_sub[instance_id] = {}
                        if data_name == "voxceleb1" or data_name == "voxceleb1_org":
                            videos_sub[instance_id]["all"] = []
                        else:
                            videos_sub[instance_id]["frontal"] = []
                            videos_sub[instance_id]["extreme"] = []

                    if data_name == "voxceleb1" or data_name == "voxceleb1_org":
                        videos_sub[instance_id]["all"].append(img_name)
                    else:
                        videos_sub[instance_id][category].append(img_name)
                f.close()
                video_ids += video_ids_sub
                videos.update(videos_sub)

                if not os.path.exists(f"{data_name}_{split}_{skip}.pt"):
                    print(f"Save: {data_name}_{split}_{skip}.pt")
                    ckpt = {
                        "video_ids": video_ids_sub,
                        "videos": videos_sub,
                        "root_dir": self.root_dir,
                    }
                    torch.save(ckpt, f"{data_name}_{split}_{skip}.pt")

        self.videos = video_ids
        self.video_dicts = videos
        print("Done init dataset")

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        np.random.seed(idx)
        random.seed(idx)
        random_int = random.randint(0, len(self.videos) - 1)
        data_name_src, instance_name_src = self.videos[idx]
        data_name_drv, instance_name_drv = self.videos[random_int]

        if "voxceleb1" in data_name_src:
            video_path_src = os.path.join(self.root_dir[data_name_src], self.split, instance_name_src)
        else:
            video_path_src = os.path.join(self.root_dir[data_name_src], instance_name_src)

        vid_target = []

        if "voxceleb1" in data_name_src:
            frames_paths_src = sorted(glob.glob(video_path_src + "/*.png"))
        else:
            frames_src = self.video_dicts[instance_name_src]
            num_frames_frontal_src = len(frames_src["frontal"])
            num_frames_extreme_src = len(frames_src["extreme"])

            if num_frames_frontal_src == 0 or num_frames_extreme_src == 0:
                return {
                    "source": placeholder,
                    "driving": placeholder,
                    "name": placeholder,
                    "video": placeholder,
                }

            frame_frontal_idx = np.random.choice(num_frames_frontal_src, replace=True, size=1)
            img_src = self.transform(
                Image.open(os.path.join(video_path_src, frames_src["frontal"][frame_frontal_idx.item()])).convert(
                    "RGB"
                )
            )
            vid_target.append(img_src)

        if "voxceleb1" in data_name_drv:
            video_path_drv = os.path.join(self.root_dir[data_name_drv], self.split, instance_name_drv)
        else:
            video_path_drv = os.path.join(self.root_dir[data_name_drv], instance_name_drv)

        if "voxceleb1" in data_name_drv:
            frames_paths_drv = sorted(glob.glob(video_path_drv + "/*.png"))
            vid_target = [self.transform(Image.open(frames_paths_src[0]).convert("RGB"))] + [
                self.transform(Image.open(p).convert("RGB")) for p in frames_paths_drv
            ]
        else:
            frames_drv = self.video_dicts[instance_name_drv]
            num_frames_frontal_drv = len(frames_drv["frontal"])
            num_frames_extreme_drv = len(frames_drv["extreme"])

            if num_frames_frontal_drv == 0 or num_frames_extreme_drv == 0:
                return {
                    "source": placeholder,
                    "driving": placeholder,
                    "name": placeholder,
                    "video": placeholder,
                }

            frames_drv["extreme"] = sorted(frames_drv["extreme"], key=lambda x: int(x.split("/")[-1].split(".")[0]))
            for name in frames_drv["extreme"]:
                img_path = os.path.join(video_path_drv, name)
                img_drv = self.transform(Image.open(img_path).convert("RGB"))
                vid_target.append(img_drv)

        return f"{instance_name_src}+{instance_name_drv}", vid_target
