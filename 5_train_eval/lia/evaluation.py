import argparse
import os

import lpips
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from dataset import Ext256_eval, Ext256_eval_cross, Taichi_eval, TED_eval, Vox256_eval
from networks.generator import Generator
from PIL import Image
from torch.utils import data
from tqdm import tqdm


def data_sampler(dataset, shuffle):
    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)


def sample_data(loader):
    while True:
        yield from loader


def load_image(filename, size):
    img = Image.open(filename).convert("RGB")
    img = img.resize((size, size))
    img = np.asarray(img)
    img = np.transpose(img, (2, 0, 1))  # 3 x 256 x 256

    return img / 255.0


def save_video(save_path, name, vid_target_recon, fps=10.0):
    if isinstance(name, tuple):
        name = name[0]
    if "/" in name:
        name = "_".join(name.split("/"))
    vid = (vid_target_recon.permute(0, 2, 3, 4, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    torchvision.io.write_video(os.path.join(save_path, f"{name}.mp4"), vid[0].cpu(), fps=fps)


def data_preprocessing(img_path, size):
    img = load_image(img_path, size)  # [0, 1]
    img = torch.from_numpy(img).unsqueeze(0).float()  # [0, 1]
    imgs_norm = (img - 0.5) * 2.0  # [-1, 1]

    return imgs_norm


class Eva(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        transform = torchvision.transforms.Compose(
            [
                transforms.Resize((args.size, args.size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

        self.dataset_name = args.dataset

        path = args.ckpt_path
        self.ckpt_name = os.path.basename(path)
        start_iter = str(os.path.splitext(self.ckpt_name)[0])
        if args.dataset == "vox":
            # path = 'checkpoints/vox.pt'
            dataset = Vox256_eval(transform)
        elif args.dataset == "taichi":
            # path = 'checkpoints/taichi.pt'
            dataset = Taichi_eval(transform)
        elif args.dataset == "ted":
            # path = 'checkpoints/ted.pt'
            dataset = TED_eval(transform)
        elif args.dataset == "ext":
            data_list = {
                "data_1": {
                    "name": "celebV",
                    "root_dir": "../../data/CelebV-HQ/processed/extracted_cropped_face_results_vox_final/all",
                    "data_list_test": "../../data/CelebV-HQ/info/reenactment/test_0.25.pt",
                },
                "data_2": {
                    "name": "vfhq",
                    "root_dir": "./../data/VFHQ/processed/extracted_cropped_face_results_vox/all",
                    "data_list_test": "../../data/VFHQ/info/reenactment/test_0.25.pt",
                },
            }
            dataset = Ext256_eval(data_list=data_list, transform=transform)
        elif args.dataset == "ext_cross":
            data_list = {
                "data_1": {
                    "name": "celebV",
                    "root_dir": "../../data/CelebV-HQ/processed/extracted_cropped_face_results_vox_final/all",
                    "data_list_test": "../../data/CelebV-HQ/info/reenactment/test_0.25.pt",
                },
                "data_2": {
                    "name": "vfhq",
                    "root_dir": "./../data/VFHQ/processed/extracted_cropped_face_results_vox/all",
                    "data_list_test": "../../data/VFHQ/info/reenactment/test_0.25.pt",
                },
            }
            dataset = Ext256_eval_cross(data_list=data_list, transform=transform)
        else:
            raise NotImplementedError

        if "cross" in args.dataset:
            self.save_folder = os.path.join(args.save_path, "eval_cross", start_iter, args.dataset)
            os.makedirs(os.path.join(args.save_path, "eval_cross", start_iter, args.dataset), exist_ok=True)
        else:
            self.save_folder = os.path.join(args.save_path, "eval", start_iter, args.dataset)
            os.makedirs(os.path.join(args.save_path, "eval", start_iter, args.dataset), exist_ok=True)
        os.makedirs(os.path.join(self.save_folder, "pred"), exist_ok=True)
        os.makedirs(os.path.join(self.save_folder, "gt"), exist_ok=True)

        print("==> loading model")
        self.gen = Generator(args.size, args.latent_dim_style, args.latent_dim_motion, args.channel_multiplier).cuda()
        weight = torch.load(path, map_location=lambda storage, loc: storage)["gen"]
        self.gen.load_state_dict(weight)
        self.gen.eval()

        print("==> loading data")
        self.loader = data.DataLoader(
            dataset,
            num_workers=1,
            batch_size=1,
            drop_last=False,
        )
        self.loss_fn = lpips.LPIPS(net="alex").cuda()

    def run(self):
        loss_list = []
        loss_lpips = []
        pbar = tqdm(range(len(self.loader)))
        loader = iter(self.loader)
        for idx in pbar:
            try:
                vid_name, vid = next(loader)
            except:
                print("skiping: ", idx)
                continue
            with torch.no_grad():
                vid_real = []
                vid_recon = []
                img_source = vid[0]
                if "cross" in self.args.dataset:
                    pbar = tqdm(vid[1:])
                    vid_recon.append(img_source.unsqueeze(2).cuda())
                    vid_real.append(img_source.unsqueeze(2).cuda())
                else:
                    pbar = tqdm(vid)

                for img_target in pbar:
                    img_recon = self.gen(img_source.cuda(), img_target.cuda())
                    vid_recon.append(img_recon.unsqueeze(2))
                    vid_real.append(img_target.unsqueeze(2).cuda())

                vid_recon = torch.cat(vid_recon, dim=2)
                vid_real = torch.cat(vid_real, dim=2)

                save_video(os.path.join(self.save_folder, "pred"), vid_name, vid_recon)
                if self.args.save_gt == 1:
                    save_video(os.path.join(self.save_folder, "gt"), vid_name, vid_real)

                if "cross" not in self.args.dataset:
                    loss_list.append(torch.abs(0.5 * (vid_recon.clamp(-1, 1) - vid_real.cuda())).mean().cpu().numpy())
                    loss_lpips.append(
                        self.loss_fn.forward(vid_real, vid_recon.clamp(-1, 1)).mean().cpu().detach().numpy()
                    )
                vid_real.cpu().detach()
                vid_recon.cpu().detach()

            if "cross" not in self.args.dataset:
                pbar.set_postfix(
                    {
                        "dataset": self.dataset_name,
                        "L1": np.mean(loss_list),
                        "lpips": np.mean(loss_lpips),
                    }
                )

        if "cross" not in self.args.dataset:
            print("Final reconstruction loss: %s" % np.mean(loss_list))
            print("Final lpips loss: %s" % np.mean(loss_lpips))
            with open(os.path.join(self.save_folder, "metric.txt"), "w") as f:
                f.write(f"reconstruction: {np.mean(loss_list)}\n")
                f.write(f"lpips: {np.mean(loss_lpips)}\n")
                f.close()


if __name__ == "__main__":
    # training params
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel_multiplier", type=int, default=1)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--latent_dim_style", type=int, default=512)
    parser.add_argument("--latent_dim_motion", type=int, default=20)
    parser.add_argument("--dataset", type=str, choices=["vox", "taichi", "ted", "ext", "ext_cross"], default="")
    parser.add_argument("--save_path", type=str, default="./evaluation_res")
    parser.add_argument("--ckpt_path", type=str, default="./evaluation_res")
    parser.add_argument("--save_gt", type=int, default=0)
    args = parser.parse_args()

    demo = Eva(args)
    demo.run()
