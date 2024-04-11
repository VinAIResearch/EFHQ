import os
import random

import imageio
import numpy as np
import pandas as pd
import torch
from augmentation import AllAugmentationTransform
from skimage import img_as_float32, io
from skimage.color import gray2rgb
from torch.utils.data import Dataset
from tqdm import tqdm


placeholder = torch.zeros((3, 512, 512), dtype=torch.int32)


def read_video(name, frame_shape, read_first_frame=False):
    """Read video which can be:

    - an image of concatenated frames
    - '.mp4' and'.gif'
    - folder with videos
    """

    if os.path.isdir(name):
        frames = sorted(os.listdir(name))
        # num_frames = len(frames)
        num_frames = 1 if read_first_frame else len(frames)
        video_array = []
        for idx in range(num_frames):
            try:
                img = img_as_float32(io.imread(os.path.join(name, frames[idx])))
                video_array.append(img)
            except Exception:
                continue
        video_array = np.array(video_array)
    elif name.lower().endswith(".png") or name.lower().endswith(".jpg"):
        image = io.imread(name)

        if len(image.shape) == 2 or image.shape[2] == 1:
            image = gray2rgb(image)

        if image.shape[2] == 4:
            image = image[..., :3]

        image = img_as_float32(image)

        video_array = np.moveaxis(image, 1, 0)

        video_array = video_array.reshape((-1,) + frame_shape)
        video_array = np.moveaxis(video_array, 1, 2)
    elif name.lower().endswith(".gif") or name.lower().endswith(".mp4") or name.lower().endswith(".mov"):
        # video = np.array(mimread(name,memtest=False))
        reader = imageio.get_reader(name)
        driving_video = []
        try:
            for im in reader:
                driving_video.append(im)
                if read_first_frame:
                    break
        except RuntimeError:
            pass
        reader.close()
        video = np.array(driving_video)
        if len(video.shape) == 3:
            video = np.array([gray2rgb(frame) for frame in video])
        if video.shape[-1] == 4:
            video = video[..., :3]
        video_array = img_as_float32(video)
    else:
        raise Exception("Unknown file extensions  %s" % name)

    return video_array


def read_video_extreme(name, frame_shape, read_first_frame=False):
    """Read video which can be:

    - an image of concatenated frames
    - '.mp4' and'.gif'
    - folder with videos
    """

    if os.path.isdir(name):
        frames = sorted(os.listdir(name))
        # num_frames = len(frames)
        num_frames = 1 if read_first_frame else len(frames)
        video_array = []
        for idx in range(num_frames):
            try:
                img = img_as_float32(io.imread(os.path.join(name, frames[idx])))
            except TypeError:
                img = img_as_float32(io.imread(os.path.join(name, frames[idx].decode())))
            except Exception:
                # print(os.path.join(path, frames[idx]), e)
                continue
            video_array.append(img)
        video_array = np.array(video_array)
    elif name.lower().endswith(".png") or name.lower().endswith(".jpg"):
        image = io.imread(name)

        if len(image.shape) == 2 or image.shape[2] == 1:
            image = gray2rgb(image)

        if image.shape[2] == 4:
            image = image[..., :3]

        image = img_as_float32(image)

        video_array = np.moveaxis(image, 1, 0)

        video_array = video_array.reshape((-1,) + frame_shape)
        video_array = np.moveaxis(video_array, 1, 2)
    elif name.lower().endswith(".gif") or name.lower().endswith(".mp4") or name.lower().endswith(".mov"):
        reader = imageio.get_reader(name)
        driving_video = []
        try:
            for im in reader:
                driving_video.append(im)
                if read_first_frame:
                    break
        except RuntimeError:
            pass
        reader.close()
        video = np.array(driving_video)
        # print(video_array.shape)
        if len(video.shape) == 3:
            video = np.array([gray2rgb(frame) for frame in video])
        if video.shape[-1] == 4:
            video = video[..., :3]
        video_array = img_as_float32(video)
    else:
        raise Exception("Unknown file extensions  %s" % name)

    return video_array


class FramesDataset(Dataset):
    """Dataset of videos, each video can be represented as:

    - an image of concatenated frames
    - '.mp4' or '.gif'
    - folder with all frames
    """

    def __init__(
        self,
        root_dir,
        data_list=None,
        frame_shape=(256, 256, 3),
        id_sampling=False,
        is_train=True,
        random_seed=0,
        pairs_list=None,
        augmentation_params=None,
        read_first_frame=False,
    ):
        self.root_dir = root_dir
        self.frame_shape = tuple(frame_shape)
        self.cross = False
        self.pairs_list = pairs_list
        self.id_sampling = id_sampling
        self.read_first_frame = read_first_frame

        if is_train:
            local_dir_name = os.path.join(self.root_dir, "train")
            self.root_dir = local_dir_name
        else:
            local_dir_name = os.path.join(self.root_dir, "test")
            self.root_dir = local_dir_name
        self.local_dir = local_dir_name

        if is_train:
            if os.path.exists(f"{os.path.basename(data_list).split('.txt')[0]}_{id_sampling}.pt"):
                print("Load train dataset ckpt")
                checkpoint = torch.load(f"{os.path.basename(data_list).split('.txt')[0]}_{id_sampling}.pt")
                train_video_ids = checkpoint["train_video_ids"]
                train_videos = checkpoint["train_videos"]
            else:
                f = open(data_list)
                file_list = f.readlines()
                if id_sampling:
                    train_video_ids = []
                    train_videos = {}
                    for file_name in tqdm(file_list):
                        img_name = file_name.strip().split("/")[1]
                        instance_id = file_name.strip().split("/")[0]
                        video_id = instance_id.split("#")[0]
                        if video_id not in train_video_ids:
                            train_video_ids.append(video_id)
                        if video_id not in train_videos.keys():
                            train_videos[video_id] = {}
                        if instance_id not in train_videos[video_id].keys():
                            train_videos[video_id][instance_id] = []
                        train_videos[video_id][instance_id].append(img_name)
                    f.close()
                else:
                    train_video_ids = []
                    train_videos = {}
                    for file_name in file_list:
                        img_name = file_name.strip().split("/")[1]
                        instance_id = file_name.strip().split("/")[0]
                        if instance_id not in train_video_ids:
                            train_video_ids.append(instance_id)
                        if instance_id not in train_videos.keys():
                            train_videos[instance_id] = []
                        train_videos[instance_id].append(img_name)
                    f.close()

                if not os.path.exists(f"{os.path.basename(data_list).split('.txt')[0]}_{id_sampling}.pt"):
                    checkpoint = {"train_video_ids": train_video_ids, "train_videos": train_videos}
                    torch.save(
                        checkpoint,
                        f"{os.path.basename(data_list).split('.txt')[0]}_{id_sampling}.pt",
                    )
        else:
            if os.path.exists(f"{os.path.basename(data_list).split('.txt')[0]}_{id_sampling}.pt"):
                print("Load test dataset ckpt")
                ckpt = torch.load(f"{os.path.basename(data_list).split('.txt')[0]}_{id_sampling}.pt")
                test_video_ids = ckpt["test_video_ids"]
                test_videos = ckpt["test_videos"]
            else:
                f_test = open(data_list)
                file_list = f_test.readlines()
                test_video_ids = []
                test_videos = {}
                for file_name in file_list:
                    img_name = file_name.strip().split("/")[1]
                    instance_id = file_name.strip().split("/")[0]
                    if instance_id not in test_video_ids:
                        test_video_ids.append(instance_id)
                    if instance_id not in test_videos.keys():
                        test_videos[instance_id] = []
                    test_videos[instance_id].append(img_name)
                f_test.close()

                if not os.path.exists(f"{os.path.basename(data_list).split('.txt')[0]}_{id_sampling}.pt"):
                    ckpt = {"test_video_ids": test_video_ids, "test_videos": test_videos}
                    torch.save(ckpt, f"{os.path.basename(data_list).split('.txt')[0]}_{id_sampling}.pt")

        if is_train:
            self.videos = train_video_ids
            self.video_dicts = train_videos
        else:
            self.videos = test_video_ids
            self.video_dicts = test_videos

        self.is_train = is_train
        if self.is_train:
            self.transform = AllAugmentationTransform(**augmentation_params)
        else:
            self.transform = None

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        if self.is_train and self.id_sampling:
            name = self.videos[idx]
            name_list = sorted(list(self.video_dicts[name].keys()))
            video_name = np.random.choice(name_list)
            path = os.path.join(self.root_dir, video_name)
        else:
            name = self.videos[idx]
            path = os.path.join(self.root_dir, name)
            video_name = name

        if self.is_train and os.path.isdir(path):
            if self.id_sampling:
                frames = self.video_dicts[name][video_name]
            else:
                frames = self.video_dicts[name]
            num_frames = len(frames)
            frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2))

            video_array = []
            for idx in frame_idx:
                try:
                    img = img_as_float32(io.imread(os.path.join(path, frames[idx])))
                except TypeError:
                    img = img_as_float32(io.imread(os.path.join(path, frames[idx].decode())))
                except Exception as e:
                    print(os.path.join(path, frames[idx]), e)
                    continue

                if len(img.shape) == 2:
                    img = gray2rgb(img)
                if img.shape[-1] == 4:
                    img = img[..., :3]
                video_array.append(img)
        else:
            video_array = read_video(path, frame_shape=self.frame_shape, read_first_frame=self.read_first_frame)
            num_frames = len(video_array)
            frame_idx = (
                np.sort(np.random.choice(num_frames, replace=True, size=2)) if self.is_train else range(num_frames)
            )
            video_array = video_array[frame_idx]

        if self.transform is not None:
            video_array_aug = self.transform(video_array)

        out = {}
        if self.is_train:
            source = np.array(video_array_aug[0], dtype="float32")
            driving = np.array(video_array_aug[1], dtype="float32")
            out["driving"] = driving.transpose((2, 0, 1))
            out["source"] = source.transpose((2, 0, 1))
        else:
            video = np.array(video_array, dtype="float32")
            out["video"] = video.transpose((3, 0, 1, 2))

        out["name"] = video_name

        return out


class DatasetRepeater(Dataset):
    """Pass several times over the same dataset for better i/o performance."""

    def __init__(self, dataset, num_repeats=100):
        self.dataset = dataset
        self.num_repeats = num_repeats

    def __len__(self):
        return self.num_repeats * self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset[idx % self.dataset.__len__()]


class PairedDataset(Dataset):
    """Dataset of pairs for animation."""

    def __init__(self, initial_dataset, number_of_pairs, seed=0):
        self.initial_dataset = initial_dataset
        pairs_list = self.initial_dataset.pairs_list

        np.random.seed(seed)

        if pairs_list is None:
            max_idx = min(number_of_pairs, len(initial_dataset))
            nx, ny = max_idx, max_idx
            xy = np.mgrid[:nx, :ny].reshape(2, -1).T
            number_of_pairs = min(xy.shape[0], number_of_pairs)
            self.pairs = xy.take(np.random.choice(xy.shape[0], number_of_pairs, replace=False), axis=0)
        else:
            videos = self.initial_dataset.videos
            name_to_index = {name: index for index, name in enumerate(videos)}
            pairs = pd.read_csv(pairs_list)
            pairs = pairs[np.logical_and(pairs["source"].isin(videos), pairs["driving"].isin(videos))]

            number_of_pairs = min(pairs.shape[0], number_of_pairs)
            self.pairs = []
            self.start_frames = []
            for ind in range(number_of_pairs):
                self.pairs.append(
                    (
                        name_to_index[pairs["driving"].iloc[ind]],
                        name_to_index[pairs["source"].iloc[ind]],
                    )
                )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        self.initial_dataset.read_first_frame = False
        first = self.initial_dataset[pair[0]]
        self.initial_dataset.read_first_frame = True
        second = self.initial_dataset[pair[1]]
        first = {"driving_" + key: value for key, value in first.items()}
        second = {"source_" + key: value for key, value in second.items()}

        return {**first, **second}


class FramesDataset_ExtremeHQ(Dataset):
    """Dataset of videos, each video can be represented as:

    - an image of concatenated frames
    - '.mp4' or '.gif'
    - folder with all frames
    """

    def __init__(
        self,
        data_list=None,
        name=None,
        frame_shape=(256, 256, 3),
        id_sampling=False,
        is_train=True,
        skip=False,
        random_seed=0,
        pairs_list=None,
        augmentation_params=None,
        read_first_frame=False,
    ):
        # self.root_dir = root_dir
        # self.videos = os.listdir(root_dir)
        self.frame_shape = tuple(frame_shape)
        self.pairs_list = pairs_list
        self.id_sampling = id_sampling
        self.read_first_frame = read_first_frame
        self.skip = skip
        self.cross = False

        self.mode = "train" if is_train else "test"

        train_video_ids = []
        train_videos = {}
        test_video_ids = []
        test_videos = {}
        self.root_dir = {}

        for key, val in data_list.items():
            data_name = val["name"]
            root_dir = val["root_dir"]

            if val["data_list_test"] is None and not is_train:
                continue

            if val["data_list"] is None and is_train:
                continue

            train_video_ids_sub = []
            train_videos_sub = {}
            test_video_ids_sub = []
            test_videos_sub = {}

            flag_train = False
            if os.path.exists(f"{data_name}_train_{id_sampling}_{skip}.pt"):
                checkpoint = torch.load(f"{data_name}_train_{id_sampling}_{skip}.pt")
                print(f"Load: {data_name}_train_{id_sampling}_{skip}.pt {len(checkpoint['train_video_ids'])}")
                train_video_ids += checkpoint["train_video_ids"]
                train_videos.update(checkpoint["train_videos"])
                self.root_dir[data_name] = root_dir
                # self.root_dir.update(checkpoint['root_dir'])
                flag_train = True

            if not flag_train:
                if data_name not in self.root_dir.keys():
                    self.root_dir[data_name] = root_dir

                f = open(val["data_list"])
                file_list = f.readlines()

                if id_sampling:
                    for file_name in tqdm(file_list):
                        if data_name == "voxceleb1" or data_name == "voxceleb1_org":
                            img_name = file_name.strip().split("/")[1]
                            instance_id = file_name.strip().split("/")[0]
                            video_id = instance_id.split("#")[0]
                        elif data_name == "vfhq":
                            (
                                img_path,
                                category,
                            ) = file_name.strip().split()  # Clip+dA1MOwymy4o+P0+C0+F31904-32010/00000001.png frontal
                            img_name = img_path.split("/")[-1]  # 00000001.png
                            instance_id = img_path.split("/")[0]  # Clip+dA1MOwymy4o+P0+C0+F31904-32010
                            _, vid, pid, cid, _ = instance_id.split("+")
                            video_id = f"{vid}+{pid}"  # dA1MOwymy4o+P0
                        else:  # celeb
                            (
                                img_path,
                                category,
                            ) = file_name.strip().split()  # 58DPO_8Bd88/id00000/58DPO_8Bd88_4/0000001.png frontal
                            img_name = img_path.split("/")[-1]  # 0000001.png
                            instance_id = "/".join(img_path.split("/")[:-1])  # 58DPO_8Bd88/id00000/58DPO_8Bd88_4
                            video_id = "/".join(instance_id.split("/")[:2])  # 58DPO_8Bd88/id00000

                        if [data_name, video_id] not in train_video_ids_sub:
                            train_video_ids_sub.append([data_name, video_id])
                        if video_id not in train_videos_sub.keys():
                            train_videos_sub[video_id] = {}
                        if instance_id not in train_videos_sub[video_id].keys():
                            train_videos_sub[video_id][instance_id] = {}
                            if data_name == "voxceleb1" or data_name == "voxceleb1_org":
                                train_videos_sub[video_id][instance_id]["all"] = []
                            else:
                                train_videos_sub[video_id][instance_id]["frontal"] = []
                                train_videos_sub[video_id][instance_id]["extreme"] = []

                        if data_name == "voxceleb1" or data_name == "voxceleb1_org":
                            train_videos_sub[video_id][instance_id]["all"].append(img_name)
                        else:
                            train_videos_sub[video_id][instance_id][category].append(img_name)
                    f.close()

                    train_video_ids += train_video_ids_sub
                    train_videos.update(train_videos_sub)
                else:
                    pass  # Not used

                if not os.path.exists(f"{data_name}_train_{id_sampling}_{skip}.pt"):
                    print(f"Save: {data_name}_train_{id_sampling}_{skip}.pt")
                    ckpt = {
                        "train_video_ids": train_video_ids_sub,
                        "train_videos": train_videos_sub,
                        "root_dir": self.root_dir,
                    }
                    torch.save(ckpt, f"{data_name}_train_{id_sampling}_{skip}.pt")

            flag_test = False
            if os.path.exists(f"{data_name}_test_{id_sampling}_{skip}.pt"):
                checkpoint = torch.load(f"{data_name}_test_{id_sampling}_{skip}.pt")
                print(f"Load:{data_name}_test_{id_sampling}_{skip}.pt {len(checkpoint['test_video_ids'])}")
                test_video_ids += checkpoint["test_video_ids"]
                test_videos.update(checkpoint["test_videos"])
                self.root_dir[data_name] = root_dir
                flag_test = True

            if not flag_test:
                f_test = open(val["data_list_test"])
                file_list = f_test.readlines()

                for file_name in file_list:
                    if data_name == "voxceleb1" or data_name == "voxceleb1_org":
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
                    if [data_name, instance_id] not in test_video_ids_sub:
                        test_video_ids_sub.append([data_name, instance_id])
                    if instance_id not in list(test_videos_sub.keys()):
                        test_videos_sub[instance_id] = {}
                        if data_name == "voxceleb1" or data_name == "voxceleb1_org":
                            test_videos_sub[instance_id]["all"] = []
                        else:
                            test_videos_sub[instance_id]["frontal"] = []
                            test_videos_sub[instance_id]["extreme"] = []

                    if data_name == "voxceleb1" or data_name == "voxceleb1_org":
                        test_videos_sub[instance_id]["all"].append(img_name)
                    else:
                        test_videos_sub[instance_id][category].append(img_name)
                f_test.close()

                test_video_ids += test_video_ids_sub
                test_videos.update(test_videos_sub)

                if not os.path.exists(f"{data_name}_test_{id_sampling}_{skip}.pt"):
                    print(f"Save: {data_name}_test_{id_sampling}_{skip}.pt")
                    ckpt = {
                        "test_video_ids": test_video_ids_sub,
                        "test_videos": test_videos_sub,
                        "root_dir": self.root_dir,
                    }
                    torch.save(ckpt, f"{data_name}_test_{id_sampling}_{skip}.pt")

        if is_train:
            self.videos = train_video_ids
            self.video_dicts = train_videos
        else:
            self.videos = test_video_ids
            self.video_dicts = test_videos

        self.is_train = is_train
        if self.is_train:
            self.transform = AllAugmentationTransform(**augmentation_params)
        else:
            self.transform = None

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        if not self.is_train:
            np.random.seed(idx)
        if self.is_train and self.id_sampling:
            data_name, video_id = self.videos[idx]
            name_list = sorted(list(self.video_dicts[video_id].keys()))
            instance_name = np.random.choice(name_list)
            if data_name == "voxceleb1" or data_name == "voxceleb1_org":
                path = os.path.join(self.root_dir[data_name], self.mode, instance_name)
            else:
                path = os.path.join(self.root_dir[data_name], instance_name)
        else:
            data_name, instance_name = self.videos[idx]
            if data_name == "voxceleb1" or data_name == "voxceleb1_org":
                path = os.path.join(self.root_dir[data_name], self.mode, instance_name)
            else:
                path = os.path.join(self.root_dir[data_name], instance_name)

        video_array = []
        if self.is_train:
            if self.id_sampling:
                frames = self.video_dicts[video_id][instance_name]
            else:
                frames = self.video_dicts[instance_name]

            if data_name == "voxceleb1" or data_name == "voxceleb1_org":
                num_frames = len(frames["all"])
                frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2))

                for idx in frame_idx:
                    try:
                        img = img_as_float32(io.imread(os.path.join(path, frames["all"][idx])))
                    except TypeError:
                        img = img_as_float32(io.imread(os.path.join(path, frames["all"][idx].decode())))
                    except Exception as e:
                        print(os.path.join(path, frames["all"][idx]), e)
                        continue
                    if len(img.shape) == 2:
                        img = gray2rgb(img)
                    if img.shape[-1] == 4:
                        img = img[..., :3]
                    video_array.append(img)
            else:
                num_frames_frontal = len(frames["frontal"])
                num_frames_extreme = len(frames["extreme"])

                flag = False
                if num_frames_extreme == 0 or (num_frames_frontal == 0 and self.skip):
                    return self.__getitem__(idx + 1)
                elif num_frames_frontal == 0 and not self.skip:
                    flag = True
                    frame_frontal_idx, frame_extreme_idx = np.sort(
                        np.random.choice(num_frames_extreme, replace=True, size=2)
                    )
                else:
                    frame_frontal_idx = np.sort(np.random.choice(num_frames_frontal, replace=True, size=1))
                    frame_extreme_idx = np.sort(np.random.choice(num_frames_extreme, replace=True, size=1))

                pool = (
                    [(frame_frontal_idx.item(), "extreme"), (frame_extreme_idx.item(), "extreme")]
                    if flag
                    else [
                        (frame_frontal_idx.item(), "frontal"),
                        (frame_extreme_idx.item(), "extreme"),
                    ]
                )

                for (idx, category) in pool:
                    try:
                        img = img_as_float32(io.imread(os.path.join(path, frames[category][idx])))
                    except TypeError:
                        img = img_as_float32(io.imread(os.path.join(path, frames[category][idx].decode())))
                    except Exception:
                        return {"skip": 1}
                    if len(img.shape) == 2:
                        img = gray2rgb(img)
                    if img.shape[-1] == 4:
                        img = img[..., :3]
                    video_array.append(img)
        else:
            if data_name == "voxceleb1" or data_name == "voxceleb1_org":
                video_array = read_video(path, frame_shape=self.frame_shape, read_first_frame=self.read_first_frame)
                num_frames = len(video_array)
                frame_idx = (
                    np.sort(np.random.choice(num_frames, replace=True, size=2)) if self.is_train else range(num_frames)
                )
                video_array = video_array[frame_idx]
            else:
                frames = self.video_dicts[instance_name]
                num_frames_frontal = len(frames["frontal"])
                num_frames_extreme = len(frames["extreme"])

                if num_frames_frontal == 0 or num_frames_extreme == 0:
                    return {"skip": 1}

                frame_frontal_idx = np.random.choice(num_frames_frontal, replace=True, size=1)
                img = img_as_float32(io.imread(os.path.join(path, frames["frontal"][frame_frontal_idx.item()])))
                video_array.append(img)

                frames["extreme"] = sorted(frames["extreme"], key=lambda x: int(x.split("/")[-1].split(".")[0]))

                for name in frames["extreme"]:
                    img_path = os.path.join(path, name)
                    try:
                        img = img_as_float32(io.imread(img_path))
                    except TypeError:
                        img = img_as_float32(io.imread(img_path.decode()))
                    except Exception as e:
                        print(os.path.join(img_path), e)
                        continue
                    if len(img.shape) == 2:
                        img = gray2rgb(img)
                    if img.shape[-1] == 4:
                        img = img[..., :3]
                    video_array.append(img)

        if self.transform is not None:
            video_array_aug = self.transform(video_array)

        out = {}
        if self.is_train:
            source = np.array(video_array_aug[0], dtype="float32")
            driving = np.array(video_array_aug[1], dtype="float32")
            out["source"] = source.transpose((2, 0, 1))
            out["driving"] = driving.transpose((2, 0, 1))
        else:
            video = np.array(video_array, dtype="float32")
            out["video"] = video.transpose((3, 0, 1, 2))

        out["name"] = instance_name
        out["data_name"] = data_name

        return out


class FramesDataset_ExtremeHQ_Cross(Dataset):
    """Dataset of videos, each video can be represented as:

    - an image of concatenated frames
    - '.mp4' or '.gif'
    - folder with all frames
    """

    def __init__(
        self,
        data_list=None,
        name=None,
        frame_shape=(256, 256, 3),
        id_sampling=False,
        is_train=True,
        skip=False,
        random_seed=0,
        pairs_list=None,
        augmentation_params=None,
        read_first_frame=False,
    ):
        self.frame_shape = tuple(frame_shape)
        self.pairs_list = pairs_list
        self.id_sampling = id_sampling
        self.read_first_frame = read_first_frame
        self.skip = skip
        self.cross = True

        self.mode = "test"
        test_video_ids = []
        test_videos = {}
        self.root_dir = {}

        for key, val in data_list.items():
            data_name = val["name"]
            root_dir = val["root_dir"]

            if val["data_list_test"] is None:
                continue

            test_video_ids_sub = []
            test_videos_sub = {}

            flag_test = False
            if os.path.exists(f"{data_name}_test_{id_sampling}_{skip}_cross.pt"):
                checkpoint = torch.load(f"{data_name}_test_{id_sampling}_{skip}_cross.pt")
                print(f"Load:{data_name}_test_{id_sampling}_{skip}_cross.pt {len(checkpoint['test_video_ids'])}")
                test_video_ids += checkpoint["test_video_ids"]
                test_videos.update(checkpoint["test_videos"])
                self.root_dir[data_name] = root_dir
                flag_test = True

            if not flag_test:
                f_test = open(val["data_list_test"])
                file_list = f_test.readlines()

                for file_name in file_list:
                    if data_name == "voxceleb1" or data_name == "voxceleb1_org":
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

                    if [data_name, instance_id] not in test_video_ids_sub:
                        test_video_ids_sub.append([data_name, instance_id])
                    if instance_id not in list(test_videos_sub.keys()):
                        test_videos_sub[instance_id] = {}
                        if data_name == "voxceleb1" or data_name == "voxceleb1_org":
                            test_videos_sub[instance_id]["all"] = []
                        else:
                            test_videos_sub[instance_id]["frontal"] = []
                            test_videos_sub[instance_id]["extreme"] = []

                    if data_name == "voxceleb1" or data_name == "voxceleb1_org":
                        test_videos_sub[instance_id]["all"].append(img_name)
                    else:
                        test_videos_sub[instance_id][category].append(img_name)
                f_test.close()

                test_video_ids += test_video_ids_sub
                test_videos.update(test_videos_sub)

                if not os.path.exists(f"{data_name}_test_{id_sampling}_{skip}_cross.pt"):
                    print(f"Save: {data_name}_test_{id_sampling}_{skip}_cross.pt")
                    ckpt = {
                        "test_video_ids": test_video_ids_sub,
                        "test_videos": test_videos_sub,
                        "root_dir": self.root_dir,
                    }
                    torch.save(ckpt, f"{data_name}_test_{id_sampling}_{skip}_cross.pt")

        self.videos = test_video_ids
        self.video_dicts = test_videos

        self.is_train = False
        if self.is_train:
            self.transform = AllAugmentationTransform(**augmentation_params)
        else:
            self.transform = None
        print("Done init dataset")

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        random.seed(idx)
        np.random.seed(idx)
        random_int = random.randint(0, len(self.videos) - 1)
        video_array = []

        data_name_src, instance_name_src = self.videos[idx]
        path_src = os.path.join(self.root_dir[data_name_src], instance_name_src)
        frames_src = self.video_dicts[instance_name_src]
        num_frames_frontal_src = len(frames_src["frontal"])
        num_frames_extreme_src = len(frames_src["extreme"])

        if num_frames_frontal_src == 0 or num_frames_extreme_src == 0:
            return {"skip": 1}

        frame_frontal_idx_src = np.random.choice(num_frames_frontal_src, replace=True, size=1)
        img_src = img_as_float32(
            io.imread(os.path.join(path_src, frames_src["frontal"][frame_frontal_idx_src.item()]))
        )
        video_array.append(img_src)

        data_name_drv, instance_name_drv = self.videos[random_int]
        path_drv = os.path.join(self.root_dir[data_name_drv], instance_name_drv)
        frames_drv = self.video_dicts[instance_name_drv]
        num_frames_frontal_drv = len(frames_drv["frontal"])
        num_frames_extreme_drv = len(frames_drv["extreme"])

        if num_frames_frontal_drv == 0 or num_frames_extreme_drv == 0:
            return {"skip": 1}
        frames_drv["extreme"] = sorted(frames_drv["extreme"], key=lambda x: int(x.split("/")[-1].split(".")[0]))

        for name in frames_drv["extreme"]:
            img_path = os.path.join(path_drv, name)
            try:
                img = img_as_float32(io.imread(img_path))
            except TypeError:
                img = img_as_float32(io.imread(img_path.decode()))
            except Exception as e:
                print(os.path.join(img_path), e)
                continue
            if len(img.shape) == 2:
                img = gray2rgb(img)
            if img.shape[-1] == 4:
                img = img[..., :3]
            video_array.append(img)

        if self.transform is not None:
            self.transform(video_array)

        out = {}
        video = np.array(video_array, dtype="float32")
        out["video"] = video.transpose((3, 0, 1, 2))
        out["name"] = f"{instance_name_src}+{instance_name_drv}"
        out["data_name"] = f"{data_name_src}+{data_name_drv}"

        return out
