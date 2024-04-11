import math
import os
import time
from argparse import ArgumentParser
from multiprocessing import Pool, cpu_count

import face_alignment
import imageio
import numpy as np
import pandas as pd
import torch
from skimage.transform import resize
from tqdm import tqdm
from ultralytics import YOLO
from util import bb_intersection_over_union, crop_bbox_from_frames_custom, join


REF_FRAME_SIZE = 360
REF_FPS = 25


def chunk_into_n(lst, n):
    size = math.ceil(len(lst) / n)
    return list(map(lambda x: lst[x * size : x * size + size], list(range(n))))


def extract_bbox(frame, refbbox, fa, yolo):
    bboxes = fa.face_detector.detect_from_image(frame[..., ::-1])
    if len(bboxes) != 0:
        bbox = max((bb_intersection_over_union(bbox, refbbox), tuple(bbox)) for bbox in bboxes)[1]
    else:
        bbox_output = yolo(frame[..., ::-1])[0].boxes.boxes.cpu()
        if len(bbox_output) == 0:
            bbox = np.array(refbbox.append(1.0))
        else:
            bboxes = [i.numpy()[:5] for i in bbox_output]
            bbox = max((bb_intersection_over_union(bbox, refbbox), tuple(bbox)) for bbox in bboxes)[1]
    return np.maximum(np.array(bbox), 0)


def process_csv(df):
    # remove = ["confused", "trash", "remove", "profile_vertical", "profile_extreme", "profile_horizontal"]
    # df = df.drop(df[df['softbin'].isin(remove)].index)

    # Area Size
    df = df[(df["y2"] - df["y1"]) * (df["x2"] - df["x1"]) >= 100 * 100]
    return df


def save_bbox_list(bbox_folder, clip_id, bbox_list):
    f = open(os.path.join(bbox_folder, clip_id + ".txt"), "w")
    print("LEFT,TOP,RIGHT,BOT", file=f)
    for bbox in bbox_list:
        print("%s,%s,%s,%s" % tuple(bbox[:4]), file=f)
    f.close()


def estimate_bbox(clip_id, clip_folder, info_path, fa, yolo, args):
    info_df = pd.read_csv(info_path, dtype={"frameid": str, "idx": str})
    info_df["frameid"] = info_df["frameid"]
    info_df = info_df.sort_values("frameid")
    info_df = process_csv(info_df)

    if len(info_df) == 0:
        print("No rows", clip_id)
        return None, None, None

    image_path_list = []
    bbox_list = []
    info_list_all = []

    for index, row in info_df.iterrows():
        img_name = str(row["frameid"]).zfill(8)
        faceid = str(row["idx"]).zfill(8)
        x1, x2 = row["x1"], row["x2"]
        y1, y2 = row["y1"], row["y2"]
        path = os.path.join(clip_folder, img_name + ".png")

        flag = True
        while flag:
            try:
                if os.path.exists(path):
                    ori_frame = imageio.imread(path)
                    # get info
                    mult = ori_frame.shape[0] / REF_FRAME_SIZE
                    frame = resize(
                        ori_frame,
                        (REF_FRAME_SIZE, int(ori_frame.shape[1] / mult)),
                        preserve_range=True,
                    )

                    # Calculate the scaling factors for width and height
                    height_scale = REF_FRAME_SIZE / ori_frame.shape[0]
                    width_scale = int(ori_frame.shape[1] / mult) / ori_frame.shape[1]

                    x1_trans, x2_trans = x1 * width_scale, x2 * width_scale
                    y1_trans, y2_trans = y1 * height_scale, y2 * height_scale

                    bbox = extract_bbox(frame, [x1_trans, y1_trans, x2_trans, y2_trans], fa, yolo)

                    bbox_list.append(bbox * mult)
                    image_path_list.append(path)
                    info_list_all.append([img_name, clip_id, faceid])
                    break
            except RuntimeError as e:
                if str(e).startswith("CUDA"):
                    print("Warning: out of memory, sleep for 2s")
                    time.sleep(2)
            except Exception as e:
                print(e)
                continue

    if len(bbox_list) != 0:
        save_bbox_list(os.path.join(args.bbox_folder, args.type), clip_id, bbox_list)
    else:
        print("No bbox", clip_id)
        return None, None, None

    return bbox_list, image_path_list, info_list_all


def store(frame_list, tube_bbox, save_folder, info_list, video_count, args):
    out, final_bbox = crop_bbox_from_frames_custom(
        frame_list,
        tube_bbox,
        image_shape=args.image_shape,
        min_size=args.min_size,
        increase_area=args.increase,
    )
    if out is None:
        return []

    for img, info in zip(out, info_list):
        frame_index, video_id, faceid = info
        folder_out = os.path.join(save_folder, video_id)
        os.makedirs(folder_out, exist_ok=True)
        imageio.imsave(os.path.join(folder_out, f"{frame_index}_{faceid}{args.format}"), img)
    return [
        {
            "bbox": "-".join(map(str, final_bbox)),
            "video_id": "#".join([video_id]),
            "height": frame_list[0].shape[0],
            "width": frame_list[0].shape[1],
        }
    ]


def run(list_id, args):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
    yolo = YOLO("yolov8n-face.pt")
    yolo.to(torch.cuda.current_device())

    pbar = tqdm(list_id)
    for clip_id in pbar:
        clip_id = clip_id.strip()
        if ".csv" in clip_id:
            clip_id = clip_id.split(".csv")[0]

        if args.dataset_name.lower() == "vfhq":  # Use bin from 1_merging
            _, video_id, pid, clip_num, frame_rlt = clip_id.split("+")

        if not os.path.exists(os.path.join(args.annotations, args.type, f"{clip_id}.csv")):
            continue
        info_path = os.path.join(args.annotations, args.type, f"{clip_id}.csv")

        if args.dataset_name.lower() == "vfhq":
            clip_folder = os.path.join(args.source_folder, args.type, video_id, clip_id)
        else:
            clip_folder = os.path.join(args.source_folder, args.type, clip_id)

        bbox_list, image_path_list, info_list_all = estimate_bbox(clip_id, clip_folder, info_path, fa, yolo, args)

        if bbox_list is None or image_path_list is None or info_list_all is None:
            print("Something None", clip_id)
            continue

        initial_bbox = None
        tube_bbox = None
        video_count = 0
        frame_list = []
        chunks_data = []
        tmp_info = []

        for path, info, bbox in zip(image_path_list, info_list_all, bbox_list):
            bbox = np.array(bbox)
            frame = imageio.imread(path)

            if initial_bbox is None:
                initial_bbox = bbox
                tube_bbox = bbox

            tube_bbox = join(tube_bbox, bbox)
            frame_list.append(frame)
            tmp_info.append(info)

            if args.dataset_name.lower() == "vfhq":
                final_folder = os.path.join(args.save_folder, "all")
            else:
                final_folder = os.path.join(args.save_folder, args.type)

            if path == image_path_list[-1]:
                chunks_data += store(
                    frame_list,
                    tube_bbox,
                    final_folder,
                    tmp_info,
                    video_count,
                    args,
                )

        if os.path.exists(os.path.join(args.save_folder, "ckpt", f"ckpt_{args.name}.txt")):
            ckpt_file = open(os.path.join(args.save_folder, "ckpt", f"ckpt_{args.name}.txt"), "a")
        else:
            ckpt_file = open(os.path.join(args.save_folder, "ckpt", f"ckpt_{args.name}.txt"), "w")
        ckpt_file.write(f"{clip_id}\n")
        ckpt_file.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--name", dest="name", help="name", type=str)
    parser.add_argument("--dataset_name", dest="dataset_name", help="dataset_name", type=str)
    parser.add_argument("--annotations", dest="annotations", help="Clips Annotation File", type=str)
    parser.add_argument("--source_folder", dest="source_folder", help="Image folder", type=str)
    parser.add_argument("--bbox_folder", dest="bbox_folder", type=str)
    parser.add_argument("--type", dest="type", help="Train/Test", type=str)
    parser.add_argument("--save_folder", dest="save_folder", help="Save images to folder", type=str)
    parser.add_argument(
        "--image_shape",
        default=(256, 256),
        type=lambda x: tuple(map(int, x.split(","))),
        help="Image shape",
    )
    parser.add_argument("--increase", default=0.1, type=float, help="Increase bbox by this amount")
    parser.add_argument("--min_size", default=256, type=int, help="Minimal allowed size")
    parser.add_argument("--format", default=".png", help="Store format (.png, .mp4)")
    parser.add_argument("--workers", default=1, type=int, help="Number of parallel workers")

    args = parser.parse_args()

    os.makedirs(args.save_folder, exist_ok=True)
    os.makedirs(os.path.join(args.save_folder, "ckpt"), exist_ok=True)
    os.makedirs(os.path.join(args.bbox_folder, args.type), exist_ok=True)
    os.makedirs(os.path.join(args.save_folder, "txt"), exist_ok=True)
    os.makedirs(os.path.join(args.save_folder, "not_found"), exist_ok=True)

    # Find all CSV files in the folder
    clip_file_list = os.listdir(os.path.join(args.annotations, args.type))
    clip_file_list = [i for i in clip_file_list if ".csv" in i]

    try:
        ncpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
    except KeyError:
        ncpus = cpu_count()

    chunked_list = chunk_into_n(clip_file_list, args.workers)
    new_process = []

    for i in chunked_list:
        new_arg = (i, args)
        new_process.append(new_arg)
    with Pool(ncpus) as pool:
        pool.starmap(run, new_process)
