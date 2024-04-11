import argparse
import json
import math
import os
import random
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np
import pandas as pd
import postprocess
from deepface import DeepFace
from deepface.commons import distance as dst
from tqdm import tqdm


def chunk_into_n(lst, n):
    size = math.ceil(len(lst) / n)
    return list(map(lambda x: lst[x * size : x * size + size], list(range(n))))


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="Head pose estimation using the 6DRepNet.")
    parser.add_argument("--bin_folder", dest="bin_folder", help="Bin folder", type=str)
    parser.add_argument("--source_folder", dest="source_folder", help="Image folder", type=str)
    parser.add_argument("--type_folder", dest="type_folder", help="Train/Test", type=str)
    parser.add_argument("--json_folder", dest="json_folder", help="Save images to folder", type=str)
    parser.add_argument("--num_process", dest="num_process", help="num process", type=int)
    args = parser.parse_args()
    return args


def verify(embedding_1, embedding_2, model_name="ArcFace", distance_metric="cosine"):
    if distance_metric == "cosine":
        distance = dst.findCosineDistance(embedding_1, embedding_2)
    elif distance_metric == "euclidean":
        distance = dst.findEuclideanDistance(embedding_1, embedding_2)
    elif distance_metric == "euclidean_l2":
        distance = dst.findEuclideanDistance(dst.l2_normalize(embedding_1), dst.l2_normalize(embedding_2))
    else:
        raise ValueError("Invalid distance_metric passed - ", distance_metric)

    threshold = dst.findThreshold(model_name, distance_metric)
    return distance <= threshold


def process_csv(df):
    remove = ["confused", "trash", "remove"]
    df = df.drop(df[df["softbin"].isin(remove)].index)

    # Area Size
    df = df[(df["y2"] - df["y1"]) * (df["x2"] - df["x1"]) >= 100 * 100]

    df["area"] = (df["y2"] - df["y1"]) * (df["x2"] - df["x1"])

    return df


def count_and_compare_occurrences(input_list):
    true_count = input_list.count(True)
    false_count = input_list.count(False)

    if true_count > false_count:
        return 1
    elif true_count <= false_count:
        return 0


def reload_json(path):
    with open(path) as f:
        dict_json = json.load(f)
        f.close()
    return dict_json


# def sample(info_df):
#     sampled_df = pd.DataFrame(columns=info_df.columns)  # Empty DataFrame to store sampled rows
#     df_copy = info_df.copy()

#     while len(sampled_df) < min(len(df_copy), 9) and len(df_copy) > 0:
#         random_row = df_copy.sample(n=1)
#         img_name = str(random_row['frameid'].iloc[0]).zfill(8)
#         path = os.path.join(IMAGE_FOLDER, clip_id, img_name + ".png")
#         if os.path.exists(path):
#             try:
#                 tmp = cv2.imread(path)
#                 sampled_df = pd.concat([sampled_df, random_row])
#             except:
#                 pass
#         df_copy = df_copy.drop(random_row.index)

#     sampled_df = sampled_df.reset_index(drop=True)
#     return sampled_df


def process(video_list, csv_list, bin_folder, source_folder, json_folder, type_folder):
    for _, video_id in enumerate(tqdm(video_list)):
        video_json = {}
        image_list = []
        all_filtered_clip = [i for i in csv_list if i.startswith(video_id)]
        for clip_id in all_filtered_clip:
            if not os.path.exists(os.path.join(bin_folder, type_folder, clip_id + ".csv")):
                continue
            path = os.path.join(os.path.join(bin_folder, type_folder, clip_id + ".csv"))
            info_df = process_csv(pd.read_csv(path, dtype={"frameid": str, "idx": str}))
            if len(info_df) == 0:
                continue
            if (info_df["softbin"] == "frontal").all():
                continue

            for index, row in info_df.iterrows():
                frame_id = str(row["frameid"]).zfill(8)
                face_id = str(row["idx"]).zfill(8)
                lmks5pts = row["lmks5pts"]
                if row["softbin"].strip() == "frontal":
                    category = "frontal"
                else:
                    category = "extreme"
                if isinstance(lmks5pts, float) and math.isnan(lmks5pts):
                    lmks5pts = None
                    continue
                lmks5pts = np.array(json.loads(lmks5pts)).reshape(-1, 2).tolist()
                bbox = [
                    math.floor(float(row["x1"])),
                    math.floor(float(row["y1"])),
                    math.ceil(float(row["x2"])),
                    math.ceil(float(row["y2"])),
                ]
                path = os.path.join(source_folder, type_folder, clip_id, frame_id + ".png")
                info = [path, clip_id, frame_id, face_id, category, bbox, lmks5pts]
                image_list.append(info)

        for img_info in image_list:
            path, c_id, frame_id, face_id, category, bbox, lmks5 = img_info
            lmks5pts = np.asarray(lmks5) if isinstance(lmks5, list) else lmks5
            if len(video_json.keys()) == 0:
                try:
                    img = cv2.imread(path)
                    face = img[
                        max(0, int(bbox[1])) : min(img.shape[1], int(bbox[3])),
                        max(0, int(bbox[0])) : min(img.shape[0], int(bbox[2])),
                    ]
                    lmks5pts -= np.array([max(0, float(bbox[0])), max(0, float(bbox[1]))])
                    aligned_face = postprocess.alignment_procedure(face, lmks5pts[0], lmks5pts[1], lmks5pts[2])
                    img_embedding = DeepFace.represent(
                        img_path=aligned_face,
                        model_name="ArcFace",
                        enforce_detection=False,
                        detector_backend="skip",
                    )[0]["embedding"]
                except Exception:
                    continue
                pid = "0"
                video_json[pid] = {
                    "info": [info],
                    f"{video_id}/{pid}/{c_id}": {
                        "info": [[path, c_id, frame_id, face_id, category]],
                        "frontal": 0,
                        "extreme": 0,
                    },
                }
                video_json["0"][f"{video_id}/{pid}/{c_id}"][category] += 1
                with open(os.path.join(json_folder, type_folder, f"{video_id}.json"), "w") as f:
                    json.dump(video_json, f)
                    f.close()
                continue

            assert os.path.exists(os.path.join(json_folder, type_folder, f"{video_id}.json"))
            video_json = reload_json(os.path.join(json_folder, type_folder, f"{video_id}.json"))

            try:
                img = cv2.imread(path)
                face = img[
                    max(0, int(bbox[1])) : min(img.shape[1], int(bbox[3])),
                    max(0, int(bbox[0])) : min(img.shape[0], int(bbox[2])),
                ]
                lmks5pts -= np.array([max(0, float(bbox[0])), max(0, float(bbox[1]))])
                aligned_face = postprocess.alignment_procedure(face, lmks5pts[0], lmks5pts[1], lmks5pts[2])
                img_embedding = DeepFace.represent(
                    img_path=aligned_face,
                    model_name="ArcFace",
                    enforce_detection=False,
                    detector_backend="skip",
                )[0]["embedding"]
            except Exception:
                continue

            video_json = reload_json(os.path.join(json_folder, type_folder, f"{video_id}.json"))
            all_pids = list(video_json.keys())
            for k in video_json.keys():
                sampled_info = random.sample(video_json[k]["info"], min(5, len(video_json[k]["info"])))
                results = []
                for ref in sampled_info:
                    path_ref, bbox_ref, lmks5_ref = ref[0], ref[-2], ref[-1]
                    lmks5_ref = np.asarray(lmks5_ref) if isinstance(lmks5_ref, list) else lmks5_ref
                    img_ref = cv2.imread(path_ref)
                    face_ref = img_ref[
                        max(0, int(bbox_ref[1])) : min(img_ref.shape[1], int(bbox_ref[3])),
                        max(0, int(bbox_ref[0])) : min(img_ref.shape[0], int(bbox_ref[2])),
                    ]
                    lmks5_ref -= np.array([max(0, float(bbox_ref[0])), max(0, float(bbox_ref[1]))])
                    aligned_face_ref = postprocess.alignment_procedure(
                        face_ref, lmks5_ref[1], lmks5_ref[0], lmks5_ref[2]
                    )
                    img_embedding_ref = DeepFace.represent(
                        img_path=aligned_face_ref,
                        model_name="ArcFace",
                        enforce_detection=False,
                        detector_backend="skip",
                    )[0]["embedding"]
                    res = verify(img_embedding, img_embedding_ref)
                    results.append(res)

                if count_and_compare_occurrences(results) == 1:
                    video_json[k]["info"].append(img_info)
                    if f"{video_id}/{k}/{c_id}" not in video_json[k].keys():
                        video_json[k][f"{video_id}/{k}/{c_id}"] = {
                            "info": [[path, c_id, frame_id, face_id, category]],
                            "frontal": 0,
                            "extreme": 0,
                        }
                    else:
                        video_json[k][f"{video_id}/{k}/{c_id}"]["info"].append([path, category])
                    video_json[k][f"{video_id}/{k}/{c_id}"][category] += 1
                    break
                elif k == all_pids[-1]:
                    pid = str(int(k) + 1)
                    video_json[str(int(k) + 1)] = {
                        "info": [img_info],
                        f"{video_id}/{pid}/{c_id}": {
                            "info": [[path, c_id, frame_id, face_id, category]],
                            "frontal": 0,
                            "extreme": 0,
                        },
                    }
                    video_json[str(int(k) + 1)][f"{video_id}/{pid}/{c_id}"][category] += 1
            with open(os.path.join(json_folder, type_folder, f"{video_id}.json"), "w") as f:
                json.dump(video_json, f)
                f.close()
    return


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(os.path.join(args.json_folder, args.type_folder), exist_ok=True)

    all_clip = os.listdir(os.path.join(args.bin_folder, args.type_folder))
    all_done = os.listdir(os.path.join(args.json_folder, args.type_folder))
    all_clip = [i.strip().split(".csv")[0] for i in all_clip if ".csv" in i]
    all_done = [i.strip().split(".json")[0] for i in all_done if ".json" in i]
    all_clip = [i for i in all_clip if i not in all_done]
    all_video_ids = list({"_".join(i.strip().split("_")[:-1]) for i in all_clip})

    try:
        ncpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
    except KeyError:
        ncpus = cpu_count()

    chunked_list = chunk_into_n(all_video_ids, args.num_process)
    new_process = []

    for i in chunked_list:
        new_arg = (
            i,
            all_clip,
            args.bin_folder,
            args.source_folder,
            args.json_folder,
            args.type_folder,
        )
        new_process.append(new_arg)
    with Pool(ncpus) as pool:
        pool.starmap(process, new_process)
