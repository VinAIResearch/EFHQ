import argparse
import copy
import json
import math
import os
from math import ceil
from multiprocessing import Pool, cpu_count

import cv2
import face_align
import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="Head pose estimation using the 6DRepNet.")
    parser.add_argument("--bin_folder", dest="bin_folder", help="Bin folder", type=str)
    parser.add_argument("--source_folder", dest="source_folder", help="Image folder", type=str)
    parser.add_argument("--type_folder", dest="type_folder", help="Train/Test", type=str)
    parser.add_argument("--save_folder", dest="save_folder", help="Save images to folder", type=str)
    parser.add_argument("--save_bin", dest="save_bin", help="Save bin csv to folder", type=str)
    parser.add_argument("--num_landmarks", dest="num_landmarks", help="5/68", type=int)
    parser.add_argument("--size", dest="size", help="512/1024", type=int)
    parser.add_argument("--num_process", dest="num_process", help="num process", type=int)
    parser.add_argument("--dataset_name", dest="dataset_name", help="VFHQ/celeb", type=str)
    args = parser.parse_args()
    return args


def chunk_into_n(lst, n):
    size = ceil(len(lst) / n)
    return list(map(lambda x: lst[x * size : x * size + size], list(range(n))))


def processing(
    list_csv,
    bin_folder,
    source_folder,
    type_folder,
    num_landmarks,
    size,
    save_folder,
    save_bin,
    dataset_name,
):
    pbar = tqdm(list_csv)
    for csv_file in pbar:
        csv_file = csv_file.strip()
        clip_id = csv_file.split(".csv")[0]
        if not os.path.exists(os.path.join(bin_folder, type_folder, csv_file)):
            continue

        df = pd.read_csv(os.path.join(bin_folder, type_folder, csv_file), dtype={"frameid": str, "idx": str})
        records = df.to_dict("records")
        new_records = []
        for _, row in enumerate(tqdm(records)):
            new_row = copy.deepcopy(row)
            frameid = str(row["frameid"])
            faceid = str(row["idx"])
            lmks5pts = row["lmks5pts"]
            lmks68pts = row["lmks68pts"]
            if isinstance(lmks5pts, float) and math.isnan(lmks5pts):
                lmks5pts = None
            if isinstance(lmks68pts, float) and math.isnan(lmks68pts):
                lmks68pts = None
            if lmks5pts is None and lmks68pts is None:
                continue
            lmks5pts = json.loads(lmks5pts) if lmks5pts is not None else None
            lmks68pts = json.loads(lmks68pts) if lmks68pts is not None else None
            if num_landmarks == 5:
                if lmks5pts is None:
                    continue
                if lmks68pts is None:
                    lmks68pts = np.zeros((68, 3))
            elif num_landmarks == 68:
                if lmks5pts is None:
                    lmks5pts = np.zeros((5, 2))
                if lmks68pts is None:
                    continue
            lmks5pts = np.array(lmks5pts).reshape(-1, 2)
            lmks68pts = np.array(lmks68pts)
            lmks68pts = np.array(lmks68pts).reshape(3, 68)
            lmks68pts = np.stack(lmks68pts[:2], axis=1).reshape(68, 2).astype(np.float32)

            # Full Image
            if dataset_name.lower() == "celeb":
                image_path = os.path.join(source_folder, type_folder, clip_id, f"{frameid}.png")
            elif dataset_name.lower() == "vfhq":
                image_path = os.path.join(
                    source_folder,
                    type_folder,
                    row["user_id"],
                    row["video_id"],
                    f"{str(frameid).zfill(8)}.png",
                )
            else:
                raise Exception(f"Invalid dataset_name: {dataset_name}, should be either 'vfhq' or 'celeb'")
            if not os.path.exists(image_path):
                continue

            img = cv2.imread(image_path)
            if num_landmarks == 5:
                img_np, lm_5pts, lm_68pts = face_align.image_align_5(img, lmks5pts, lmks68pts, size, 4096, True)
            else:
                img_np, lm_5pts, lm_68pts = face_align.image_align_68(img, lmks5pts, lmks68pts, size, 4096, True)
            if img_np is None:
                continue

            lmk_5pts, lmk_68pts = [], []
            base_outpath = os.path.join(save_folder, type_folder, clip_id)
            os.makedirs(base_outpath, exist_ok=True)
            img_outpath = os.path.join(base_outpath, f"{frameid}_{faceid}.png")
            lmk5_outpath = os.path.join(base_outpath, f"{frameid}_{faceid}.txt")
            f = open(lmk5_outpath, "w")
            for lmk in lm_5pts:
                # cv2.circle(img_np, (int(lmk[0]), int(lmk[1])), 1, (0, 0, 255), -1)
                f.write(f"{lmk[0]} {lmk[1]}\n")
                for i in range(len(lmk)):
                    lmk_5pts.append(lmk[i])
            f.close()
            for lmk in lm_68pts:
                # cv2.circle(img_np, (int(lmk[0]), int(lmk[1])), 3, (0, 0, 255), -1)
                for i in range(len(lmk)):
                    lmk_68pts.append(lmk[i])

            cv2.imwrite(img_outpath, img_np)

            new_row["lmks5pts"], new_row["lmk_68pts"] = lmk_5pts, lmk_68pts
            new_row["aligned_path"] = os.path.join(clip_id, f"{frameid}_{faceid}.png")
            new_records.append(new_row)
        new_df = pd.DataFrame(new_records)
        new_df.to_csv(os.path.join(save_bin, type_folder, csv_file), index=False)
    return


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(os.path.join(args.save_folder, args.type_folder), exist_ok=True)
    os.makedirs(os.path.join(args.save_bin, args.type_folder), exist_ok=True)

    all_csv_list = os.listdir(os.path.join(args.bin_folder, args.type_folder))
    all_csv_list = [i for i in all_csv_list if ".csv" in i]
    all_done_list = os.listdir(os.path.join(args.save_bin, args.type_folder))
    all_csv_list = [i for i in all_csv_list if i not in all_done_list]

    try:
        ncpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
    except KeyError:
        ncpus = cpu_count()

    chunked_list = chunk_into_n(all_csv_list, args.num_process)
    new_process = []
    for i in chunked_list:
        new_arg = (
            i,
            args.bin_folder,
            args.source_folder,
            args.type_folder,
            args.num_landmarks,
            args.size,
            args.save_folder,
            args.save_bin,
            args.dataset_name,
        )
        new_process.append(new_arg)
    with Pool(ncpus) as pool:
        pool.starmap(processing, new_process)
