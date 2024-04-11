import argparse
import json
import math
import os
from multiprocessing import Pool, cpu_count

from tqdm import tqdm


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="Head pose estimation using the 6DRepNet.")
    parser.add_argument("--source_folder", dest="source_folder", help="Image folder", type=str)
    parser.add_argument("--save_folder", dest="save_folder", help="Saved Image folder", type=str)
    parser.add_argument("--type_folder", dest="type_folder", help="Train/Test", type=str)
    parser.add_argument("--json_folder", dest="json_folder", help="Save images to folder", type=str)
    parser.add_argument("--num_process", dest="num_process", help="num process", type=int)
    args = parser.parse_args()
    return args


def chunk_into_n(lst, n):
    size = math.ceil(len(lst) / n)
    return list(map(lambda x: lst[x * size : x * size + size], list(range(n))))


def reorganize(video_id_list, source_folder, json_folder, save_folder, type_folder):
    for video_id in tqdm(video_id_list):
        with open(os.path.join(json_folder, type_folder, video_id + ".json")) as f:
            info = json.load(f)
            f.close()
        if len(info.keys()) == 0:
            continue
        for id_num in info.keys():
            new_id = "id" + str(id_num).zfill(8)
            if "info" not in info[id_num].keys() or len(info[id_num]["info"]) == 0:
                continue
            for frame_info in info[id_num]["info"]:
                path, c_id, frame_id, face_id, _, _, _ = frame_info
                src_img_path = os.path.join(source_folder, type_folder, c_id, f"{frame_id}_{face_id}.png")
                if not os.path.exists(src_img_path):
                    continue
                out_dir = os.path.join(save_folder, type_folder, video_id, new_id, c_id)
                os.makedirs(out_dir, exist_ok=True)
                os.system(f"cp {src_img_path} {out_dir}")


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(os.path.join(args.save_folder, args.type_folder), exist_ok=True)
    all_video_ids = os.listdir(os.path.join(args.json_folder, args.type_folder))
    all_video_ids = [i.strip().split(".json")[0] for i in all_video_ids]

    try:
        ncpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
    except KeyError:
        ncpus = cpu_count()

    chunked_list = chunk_into_n(all_video_ids, args.num_process)
    new_process = []
    for i in chunked_list:
        new_arg = (i, args.source_folder, args.json_folder, args.save_folder, args.type_folder)
        new_process.append(new_arg)
    with Pool(ncpus) as pool:
        pool.starmap(reorganize, new_process)
