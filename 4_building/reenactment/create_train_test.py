import argparse
import json
import math
import os


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="Train/Test Split")
    parser.add_argument("--dataset_name", dest="dataset_name", help="vfhq/celeb", type=str)
    parser.add_argument("--test_pct", dest="test_pct", help="Pct for test set", type=float, default=0.25)
    parser.add_argument("--json_path", dest="json_path", help="json_path", type=str)
    parser.add_argument("--text_path", dest="text_path", help="json_path", type=str)
    parser.add_argument(
        "--annotation_folder",
        dest="annotation_folder",
        help="Folder for final annotation",
        type=str,
    )
    args = parser.parse_args()
    return args


def split(dataset_name, json_path, text_path, annotation_folder, pct=0.25):
    os.makedirs(annotation_folder, exist_ok=True)
    with open(json_path, "r") as f:
        dict_json = json.load(f)
        f.close()

    sorted_instance = sorted(
        key for key, value in dict_json.items() if (value.get("frontal", 0) >= 1 and value.get("extreme", 0) >= 5)
    )
    sample_count = math.ceil(pct * len(dict_json.keys()))
    print("Test size: ", sample_count)

    with open(text_path, "r") as f:
        info = f.readlines()
        f.close()

    test_instance = sorted_instance[-sample_count:]
    test_id = []
    if dataset_name.lower().strip() == "vfhq":
        for instance in test_instance:
            prefix, video_id, pid, clip_idx, _ = instance.strip().split("+")
            test_id.append(f"{prefix}+{video_id}+{pid}")
    else:
        for instance in test_instance:
            video_id, pid, cid = instance.strip().split("/")
            test_id.append(f"{video_id}/{pid}")
    test_id = list(set(test_id))

    train_list = []
    test_list = []

    for i in info:
        img_path, category = i.strip().split()
        if dataset_name.lower().strip() == "vfhq":
            instance = img_path.split("/")[0]
            prefix, video_id, pid, clip_idx, _ = instance.strip().split("+")
            name = f"{prefix}+{video_id}+{pid}"
        else:
            video_id, pid, clip_idx, _ = img_path.split("/")
            name = f"{video_id}/{pid}"

        if name in test_id:
            if instance in test_instance:
                test_list.append(i)
        else:
            train_list.append(i)

    with open(os.path.join(annotation_folder, f"train_{pct}.txt"), "w") as f:
        for i in train_list:
            f.write(i)
        f.close()

    with open(os.path.join(annotation_folder, f"test_{pct}.txt"), "w") as f:
        for i in test_list:
            f.write(i)
        f.close()


if __name__ == "__main__":
    args = parse_args()
    split(args.dataset_name, args.json_path, args.text_path, args.annotation_folder, args.test_pct)
