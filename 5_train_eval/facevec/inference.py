import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
from backbones import get_model
from dataset import DataLoaderX, ImageDataset
from natsort import natsorted
from tqdm import tqdm


def l2_normalize(x, axis=1, eps=1e-8):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)


@torch.no_grad()
def preproc(img):
    img = cv2.imread(img)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).float()
    img.div_(255).sub_(0.5).div_(0.5)
    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch ArcFace Training")
    parser.add_argument("--network", type=str, default="r50", help="backbone network")
    parser.add_argument("--weight", type=str, default="")
    parser.add_argument("--dir", type=str, default=None)
    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--listfile", type=str, default=None, help="filter list")
    args = parser.parse_args()
    bs = args.bs
    listfile = args.listfile
    outp = Path(args.out)
    outp.mkdir(exist_ok=True, parents=True)
    net = get_model(args.network, fp16=False)
    if ".ckpt" in args.weight:
        statedict = torch.load(args.weight)["state_dict"]
        model_statedict = {key[6:]: val for key, val in statedict.items() if key.startswith("model.")}
    else:
        model_statedict = torch.load(args.weight)
    net.load_state_dict(model_statedict)
    net.eval()
    net.cuda()
    if dir is not None and listfile is None:
        dir = Path(args.dir)
        cache_path = Path("./cached/")
        dirstr = dir.resolve().as_posix().replace("/", "@")
        cache_path = cache_path / dirstr
        if not cache_path.exists():
            image_files = natsorted(list(dir.rglob("*.[jp][pn]g")))
            cache_path.parent.mkdir(exist_ok=True, parents=True)
            with open(cache_path, "wb") as f:
                pickle.dump(image_files, f)
        else:
            with open(cache_path, "rb") as f:
                image_files = pickle.load(f)
    elif listfile is not None:
        dir = Path(args.dir)
        with open(listfile) as f:
            lines = f.readlines()
        image_files = set()
        for line in lines:
            p = line.strip()
            p = dir / p
            assert p.exists(), p
            image_files.add(p)
        image_files = natsorted(list(image_files))

    dataset = ImageDataset(image_files)
    init_fn = None

    train_loader = DataLoaderX(
        local_rank=0,
        dataset=dataset,
        batch_size=bs,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
    )

    n_batch = (len(image_files) // bs) + 1
    final_feats = []
    for (imgs, labels) in tqdm(train_loader):
        feats = net(imgs).detach().cpu().numpy()
        feats = l2_normalize(feats)
        final_feats.append(feats)
    final_feats = np.concatenate(final_feats).reshape(-1, 512)
    outp_npz = outp / "feat.npz"
    outp_path = outp / "paths.txt"
    outfp_path = outp / "fullpaths.txt"
    full_image_files = list(map(lambda x: x.resolve().as_posix(), image_files))
    relative_image_files = list(map(lambda x: x.resolve().relative_to(dir).as_posix(), image_files))
    np.savez(outp_npz.as_posix(), final_feats)
    with open(outp_path, "w") as f:
        f.write("\n".join(relative_image_files))
    with open(outfp_path, "w") as f:
        f.write("\n".join(full_image_files))
