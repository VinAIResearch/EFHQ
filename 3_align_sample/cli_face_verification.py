import os
import pickle
import random
import time
from functools import lru_cache, partial
from multiprocessing import Pool
from pathlib import Path

import albumentations as A
import cv2
import face_verification_evaltoolkit
import numpy as np
import pandas as pd
import typer
from skimage import transform as trans
from tqdm import tqdm


arcface_dst = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def get_lmk(lmks5pts, lmks68pts):
    lmk = lmks5pts
    if isinstance(lmk, str):
        lmk = eval(lmk)
    if not isinstance(lmk, list):
        lmk = lmks68pts
        if isinstance(lmk, str):
            lmk = eval(lmk)
        if not isinstance(lmk, list) or len(lmk) != 68 * 3:
            return None
        lmk = np.array(lmk).reshape(3, 68)
        lm = np.stack(lmk[:2], axis=1).reshape(68, 2).astype(np.int32)

        lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
        lm5p = np.stack(
            [
                lm[lm_idx[0], :],
                np.mean(lm[lm_idx[[1, 2]], :], 0),
                np.mean(lm[lm_idx[[3, 4]], :], 0),
                lm[lm_idx[5], :],
                lm[lm_idx[6], :],
            ],
            axis=0,
        )
        lm5p = lm5p[[1, 2, 0, 3, 4], :]

        lmk = lm5p.reshape(5, 2)
    return lmk


@lru_cache(2048)
def cached_listdir(d: Path, image_only=False):
    fs = os.listdir(d)
    if image_only:
        fs = [file for file in fs if file.lower().endswith((".png", ".jpg", ".jpeg", ".gif"))]
    return list(fs)


@lru_cache(2048)
def cached_rglob(d: Path, filter_str=None):
    fs = list(d.rglob("*.[jp][pn]g"))
    if filter_str is not None and isinstance(filter_str, str):
        fs = list(filter(lambda x: filter_str in x.resolve().as_posix(), fs))
    return fs


def estimate_norm(lmk, image_size=112, mode="arcface"):
    assert lmk.shape == (5, 2)
    assert image_size % 112 == 0 or image_size % 128 == 0
    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0
    else:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio
    dst = arcface_dst * ratio
    dst[:, 0] += diff_x
    tform = trans.SimilarityTransform()
    tform.estimate(lmk, dst)
    M = tform.params[0:2, :]
    return M


def norm_crop(img, landmark, image_size=112, mode="arcface"):
    M = estimate_norm(landmark, image_size, mode)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped


app = typer.Typer(pretty_exceptions_show_locals=False)

transform = A.Compose(
    [
        A.RandomBrightnessContrast(p=0.5),
        A.Downscale(scale_min=0.25, scale_max=0.35, p=0.5),
        A.ImageCompression(p=0.5),
    ]
)


def func(csv_file, sample_per_csv, raw_dir, outimagepath, seed, debug_mode=False):
    bin = csv_file.parent.name
    df = pd.read_csv(csv_file)
    if bin != "frontal":
        sample_per_csv *= 2
    real_samples_per_csv = min(len(df), sample_per_csv)
    df = df.sample(n=real_samples_per_csv, random_state=seed)
    id = csv_file.stem
    fail_images = []
    for row in df.itertuples():
        frameid = row.frameid
        filename = str(frameid).zfill(8)
        filepath = f"{id}/{filename}.png"

        infilepath = raw_dir / filepath
        out_path = outimagepath / id / bin / f"{filename}.png"

        if out_path.exists():
            continue

        if not infilepath.exists():
            fail_images.append(infilepath.as_posix())
            continue

        img = cv2.imread(infilepath.as_posix())
        lmk = get_lmk(None, row.lmks68pts)
        lmk_x_min = lmk[:, 0].min()
        lmk_x_max = lmk[:, 0].max()
        lmk_y_min = lmk[:, 1].min()
        lmk_y_max = lmk[:, 1].max()
        if (
            lmk_x_min < int(row.x1) - 10
            or lmk_x_max > int(row.x2) + 10
            or lmk_y_min < int(row.y1) - 10
            or lmk_y_max > int(row.y2) + 10
        ):
            continue
        if debug_mode:
            out_debug_path = outimagepath.parent / f"debug/{time.time()}.jpg"
            out_debug_path.parent.mkdir(exist_ok=True, parents=True)
            debug_img = img.copy()
            for lmk_ in lmk:
                cv2.circle(debug_img, (int(lmk_[0]), int(lmk_[1])), 1, (0, 0, 255), -1)
            cv2.imwrite(out_debug_path.as_posix(), debug_img)

        aligned = norm_crop(img, lmk)
        aligned = transform(image=aligned)["image"]
        out_path.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(out_path.as_posix(), aligned)
    if len(fail_images) > 0:
        txt_path = outimagepath.parent / "logs" / f"{csv_file.stem}.txt"
        txt_path.parent.mkdir(parents=True, exist_ok=True)

        with open(txt_path, "w") as f:
            f.write("\n".join(fail_images))


@app.command()
def align(
    csv_dir: Path = typer.Argument(..., help="Base csv folder", dir_okay=True, exists=True),
    raw_dir: Path = typer.Argument(..., help="Base raw folder", dir_okay=True, exists=True),
    outdir: Path = typer.Argument(..., help="Output dir"),
    sample_per_csv: int = typer.Option(50, help="Align how much image per csv"),
    nprocs: int = typer.Option(8, help="Num process"),
):
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    CACHE_PATH = Path("./.cache")
    CACHE_PATH.mkdir(exist_ok=True, parents=True)
    filecache = CACHE_PATH / f'{csv_dir.resolve().as_posix().replace("/", "@!@!")}.pkl'
    if filecache.exists():
        with open(filecache, "rb") as f:
            csv_files = pickle.load(f)
    else:
        csv_files = list(csv_dir.rglob("*.csv"))
        with open(filecache, "wb") as f:
            pickle.dump(csv_files, f)
    outimagepath = outdir / "images"

    with Pool(nprocs) as p:
        list(
            tqdm(
                p.imap(
                    partial(
                        func,
                        sample_per_csv=sample_per_csv,
                        raw_dir=raw_dir,
                        outimagepath=outimagepath,
                        seed=seed,
                    ),
                    csv_files,
                ),
                total=len(csv_files),
                desc="Parse CSV",
            )
        )


@app.command()
def eval(
    feat_folder: Path = typer.Argument(..., help="feat folder", exists=True, dir_okay=True),
    pair_txt_path: Path = typer.Argument(..., help="feat folder", exists=True, file_okay=True),
    nfolds: int = typer.Argument(10, help="nfolds"),
):
    npz_path = feat_folder / "feat.npz"
    filepath_to_embedidx_path = feat_folder / "paths.txt"
    with open(filepath_to_embedidx_path) as f:
        embedidx_to_filepath = list(map(lambda x: Path(x.strip()), f.readlines()))
        filepath_to_embedidx = {embedidx_to_filepath[i].as_posix(): i for i in range(len(embedidx_to_filepath))}
    embeddings = np.load(npz_path.as_posix())["arr_0"]
    issame_list = []
    final_same_embeddings = []
    final_diff_embeddings = []
    with open(pair_txt_path) as f:
        lines = f.readlines()
        for line in tqdm(lines, desc="Preparing input"):
            line = line.strip()
            p1, p2, issame = line.split("\t")
            e1 = embeddings[filepath_to_embedidx[p1]]
            e2 = embeddings[filepath_to_embedidx[p2]]
            issame = bool(int(issame))
            if issame:
                final_same_embeddings.append(e1)
                final_same_embeddings.append(e2)
            else:
                final_diff_embeddings.append(e1)
                final_diff_embeddings.append(e2)
    final_same_embeddings = np.vstack(final_same_embeddings).reshape(-1, 2, 512)
    final_diff_embeddings = np.vstack(final_diff_embeddings).reshape(-1, 2, 512)
    final_embeddings = []
    issame_list = []
    for same, diff in zip(final_same_embeddings, final_diff_embeddings):
        final_embeddings.extend([same[0], same[1]])
        final_embeddings.extend([diff[0], diff[1]])
        issame_list.append(1)
        issame_list.append(0)
    final_embeddings = np.vstack(final_embeddings)
    assert len(final_embeddings.shape) == 2
    assert final_embeddings.shape[0] == len(issame_list) * 2

    print("Start evaluating...")
    print(f"n_pairs: {len(issame_list)}")
    print(f"embedding shape: {final_embeddings.shape}")
    tars, tars_std, fars, real_fars = face_verification_evaltoolkit.get_tpr_far(
        final_embeddings, issame_list, nrof_folds=nfolds
    )
    for tar, tar_std, far, real_far in zip(tars, tars_std, fars, real_fars):
        print(f"tar={tar}; far={far}")
        # print("tar_std=", tar_std)
        # print("real_far=", real_far)


@app.command()
def sample_f2f(
    aligned_dir: Path = typer.Argument(..., help="Aligned dirs"),
    outdir: Path = typer.Argument(..., help="output txt file path", dir_okay=True),
    n_pairs: int = typer.Option(50000, help="How much pair gonna create"),
    seed: int = typer.Option(0, help="seed"),
):
    random.seed(seed)
    np.random.seed(seed)

    ids = list(cached_listdir(aligned_dir))
    cache_npy_path = outdir / "ids_frontal.npy"
    if cache_npy_path.exists():
        ids_with_frontal = np.load(cache_npy_path.as_posix()).tolist()
    else:
        ids_with_frontal = set()
        for id in tqdm(ids, desc="Counting"):
            dir_path = aligned_dir / id
            subdir = list(cached_listdir(dir_path))
            if "frontal" in subdir:
                id_path = aligned_dir / id / "frontal"
                files = cached_listdir(id_path, image_only=True)
                if len(files) < 2:
                    continue
                ids_with_frontal.add(id)
        ids_with_frontal = list(ids_with_frontal)
        cache_npy_path.parent.mkdir(exist_ok=True, parents=True)
        np.save(cache_npy_path, np.array(ids_with_frontal))

    print(f"Number of ID has frontal: {len(ids_with_frontal)}")

    sampled_pairs = set()
    pbar = tqdm(range(n_pairs), desc="Sample frontal2frontal")
    while len(sampled_pairs) < n_pairs // 2:
        id = random.choice(ids_with_frontal)
        id_path = aligned_dir / id / "frontal"
        files = cached_listdir(id_path, image_only=True)
        pair = tuple(map(lambda x: id_path / x, np.random.choice(files, 2, replace=False))) + (1,)
        l_before = len(sampled_pairs)
        sampled_pairs.add(pair)
        pbar.update(len(sampled_pairs) - l_before)

    while len(sampled_pairs) < n_pairs:
        ids = np.random.choice(ids_with_frontal, 2, replace=False)
        assert ids[0] != ids[1]
        id_paths = list(map(lambda x: aligned_dir / x / "frontal", ids))
        files1, files2 = cached_listdir(id_paths[0], image_only=True), cached_listdir(id_paths[1], image_only=True)
        pair = tuple(
            [
                id_paths[0] / np.random.choice(files1, 1)[0],
                id_paths[1] / np.random.choice(files2, 1)[0],
                0,
            ]
        )
        l_before = len(sampled_pairs)
        sampled_pairs.add(pair)
        pbar.update(len(sampled_pairs) - l_before)

    outpath = outdir / "f2f.txt"
    outdir.mkdir(exist_ok=True, parents=True)
    with open(outpath, "w") as f:
        f.write(
            "\n".join(
                map(
                    lambda x: f"{x[0].relative_to(aligned_dir).as_posix()}\t{x[1].relative_to(aligned_dir).as_posix()}\t{x[2]}",
                    sampled_pairs,
                )
            )
        )


@app.command()
def sample_f2p(
    aligned_dir: Path = typer.Argument(..., help="Aligned dirs"),
    outdir: Path = typer.Argument(..., help="output txt file path", dir_okay=True),
    n_pairs: int = typer.Option(50000, help="How much pair gonna create"),
    seed: int = typer.Option(0, help="seed"),
):
    random.seed(seed)
    np.random.seed(seed)

    ids = list(cached_listdir(aligned_dir))
    cache_npy_path = outdir / "ids_frontal_profile.npy"
    if cache_npy_path.exists():
        obj = np.load(cache_npy_path.as_posix(), allow_pickle=True)
        d = obj.item()
        ids_with_frontal_and_profile = list(d.get("fe"))
        frontal_ids = list(d.get("f"))
        profile_ids = list(d.get("e"))
    else:
        ids = list(cached_listdir(aligned_dir))
        ids_with_frontal_only = []
        ids_with_profile_only = []
        ids_with_frontal_and_profile = []
        n_frontal = 0
        n_extreme = 0
        for id in tqdm(ids, desc="Counting"):
            dir_path = aligned_dir / id
            subdirs = list(cached_listdir(dir_path))
            if len(subdirs) == 1:
                if subdirs[0] == "frontal":
                    ids_with_frontal_only.append(id)
                    n_frontal += 1
                elif "profile" in subdirs[0]:
                    ids_with_profile_only.append(id)
                    n_extreme += 1
            else:
                if "frontal" in subdirs:
                    ids_with_frontal_and_profile.append(id)
                    n_frontal += 1
                else:
                    ids_with_profile_only.append(id)
                n_extreme += 1

        ids_with_frontal_and_profile = list(set(ids_with_frontal_and_profile))
        frontal_ids = list(set(ids_with_frontal_only + ids_with_frontal_and_profile))
        profile_ids = list(set(ids_with_profile_only + ids_with_frontal_and_profile))
        d = {"fe": ids_with_frontal_and_profile, "f": frontal_ids, "e": profile_ids}
        cache_npy_path.parent.mkdir(exist_ok=True, parents=True)
        np.save(cache_npy_path, d)

    for k, v in d.items():
        print(f"Number of ID has {k}: {len(v)}")

    sampled_pairs = set()
    pbar = tqdm(range(n_pairs), desc="Sample frontal2extreme")
    while len(sampled_pairs) < n_pairs // 2:
        id = random.choice(ids_with_frontal_and_profile)
        files = cached_rglob(aligned_dir / id)
        frontal_files = list(filter(lambda x: "frontal" in x.resolve().as_posix(), files))
        profile_files = list(filter(lambda x: "frontal" not in x.resolve().as_posix(), files))
        if len(profile_files) < 1:
            continue
        pair = tuple(
            [
                np.random.choice(frontal_files, 1)[0],
                np.random.choice(profile_files, 1)[0],
                1,
            ]
        )
        l_before = len(sampled_pairs)
        sampled_pairs.add(pair)
        pbar.update(len(sampled_pairs) - l_before)

    while len(sampled_pairs) < n_pairs:
        frontal_id = random.choice(frontal_ids)
        profile_id = random.choice(profile_ids)
        if frontal_id == profile_id:
            continue
        frontal_files = cached_rglob(aligned_dir / frontal_id)
        profile_files = cached_rglob(aligned_dir / profile_id)
        frontal_files = list(filter(lambda x: "frontal" in x.resolve().as_posix(), frontal_files))
        profile_files = list(filter(lambda x: "frontal" not in x.resolve().as_posix(), profile_files))
        pair = tuple(
            [
                np.random.choice(frontal_files, 1)[0],
                np.random.choice(profile_files, 1)[0],
                0,
            ]
        )
        l_before = len(sampled_pairs)
        sampled_pairs.add(pair)
        pbar.update(len(sampled_pairs) - l_before)

    outpath = outdir / "f2p.txt"
    outdir.mkdir(exist_ok=True, parents=True)
    with open(outpath, "w") as f:
        f.write(
            "\n".join(
                map(
                    lambda x: f"{x[0].relative_to(aligned_dir).as_posix()}\t{x[1].relative_to(aligned_dir).as_posix()}\t{x[2]}",
                    sampled_pairs,
                )
            )
        )


@app.command()
def sample_p2p(
    aligned_dir: Path = typer.Argument(..., help="Aligned dirs"),
    outdir: Path = typer.Argument(..., help="output txt file path", dir_okay=True),
    n_pairs: int = typer.Option(50000, help="How much pair gonna create"),
    seed: int = typer.Option(0, help="seed"),
):
    random.seed(seed)
    np.random.seed(seed)

    ids = list(cached_listdir(aligned_dir))
    cache_npy_path = outdir / "ids_frontal_profile.npy"
    if cache_npy_path.exists():
        obj = np.load(cache_npy_path.as_posix(), allow_pickle=True)
        d = obj.item()
        ids_with_frontal_and_profile = list(d.get("fe"))
        frontal_ids = list(d.get("f"))
        profile_ids = list(d.get("e"))
    else:
        ids = list(cached_listdir(aligned_dir))
        ids_with_frontal_only = []
        ids_with_profile_only = []
        ids_with_frontal_and_profile = []
        n_frontal = 0
        n_extreme = 0
        for id in tqdm(ids, desc="Counting"):
            dir_path = aligned_dir / id
            subdirs = list(cached_listdir(dir_path))
            if len(subdirs) == 1:
                if subdirs[0] == "frontal":
                    ids_with_frontal_only.append(id)
                    n_frontal += 1
                elif "profile" in subdirs[0]:
                    ids_with_profile_only.append(id)
                    n_extreme += 1
            else:
                if "frontal" in subdirs:
                    ids_with_frontal_and_profile.append(id)
                    n_frontal += 1
                else:
                    ids_with_profile_only.append(id)
                n_extreme += 1

        ids_with_frontal_and_profile = list(set(ids_with_frontal_and_profile))
        frontal_ids = list(set(ids_with_frontal_only + ids_with_frontal_and_profile))
        profile_ids = list(set(ids_with_profile_only + ids_with_frontal_and_profile))
        d = {"fe": ids_with_frontal_and_profile, "f": frontal_ids, "e": profile_ids}
        cache_npy_path.parent.mkdir(exist_ok=True, parents=True)
        np.save(cache_npy_path, d)

    for k, v in d.items():
        print(f"Number of ID has {k}: {len(v)}")

    sampled_pairs = set()
    pbar = tqdm(range(n_pairs), desc="Sample extreme2extreme")
    while len(sampled_pairs) < n_pairs // 2:
        id = random.choice(ids_with_frontal_and_profile)
        files = cached_rglob(aligned_dir / id)
        profile_files = list(filter(lambda x: "frontal" not in x.resolve().as_posix(), files))
        if len(profile_files) < 2:
            continue
        pair = tuple(np.random.choice(profile_files, 2, replace=False).tolist() + [1])
        l_before = len(sampled_pairs)
        sampled_pairs.add(pair)
        pbar.update(len(sampled_pairs) - l_before)

    while len(sampled_pairs) < n_pairs:
        profile_ids_ = np.random.choice(profile_ids, 2, replace=False)
        profile_files1 = cached_rglob(aligned_dir / profile_ids_[0])
        profile_files2 = cached_rglob(aligned_dir / profile_ids_[1])
        profile_files1 = list(filter(lambda x: "frontal" not in x.resolve().as_posix(), profile_files1))
        profile_files2 = list(filter(lambda x: "frontal" not in x.resolve().as_posix(), profile_files2))
        pair = tuple(
            [
                np.random.choice(profile_files1, 1)[0],
                np.random.choice(profile_files2, 1)[0],
                0,
            ]
        )
        l_before = len(sampled_pairs)
        sampled_pairs.add(pair)
        pbar.update(len(sampled_pairs) - l_before)

    outpath = outdir / "p2p.txt"
    outdir.mkdir(exist_ok=True, parents=True)
    with open(outpath, "w") as f:
        f.write(
            "\n".join(
                map(
                    lambda x: f"{x[0].relative_to(aligned_dir).as_posix()}\t{x[1].relative_to(aligned_dir).as_posix()}\t{x[2]}",
                    sampled_pairs,
                )
            )
        )


if __name__ == "__main__":
    app()
