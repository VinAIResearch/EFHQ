import argparse
import json
import os

import pandas as pd


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="Train/Test Split")
    parser.add_argument("--source_folder", dest="source_folder", help="Source folder of images", type=str)
    parser.add_argument("--bin_folder", dest="bin_folder", help="bin_folder for annotation", type=str)
    parser.add_argument("--json_path", dest="json_path", help="json save path", type=str)
    parser.add_argument("--text_path", dest="text_path", help="text_path", type=str)
    args = parser.parse_args()
    return args


def metadata(bin_folder, source_folder, json_path, text_path):
    # source_folder: extracted_cropped_face_results_vox
    # type_folder: all
    # bin_folder: output/VFHQ/merge/4_binned_edited
    root = os.path.join(source_folder, "all")  # extracted_cropped_face_results_vox/all
    all_instance_id = os.listdir(root)
    dict_values = {}
    for instance_id in all_instance_id:
        _, video_id, pid, clip_idx, frame_rlt = instance_id.strip().split("+")
        all_img = os.listdir(
            os.path.join(root, instance_id)
        )  # extracted_cropped_face_results_vox/all/Clip+dA1MOwymy4o+P0+C0+F31904-32010

        if os.path.exists(os.path.join(bin_folder, "test", video_id + ".csv")):
            df_path = os.path.join(bin_folder, "test", video_id + ".csv")
        elif os.path.exists(os.path.join(bin_folder, "train", video_id + ".csv")):
            df_path = os.path.join(bin_folder, "train", video_id + ".csv")
        else:
            continue

        df = pd.read_csv(df_path, dtype={"frameid": str, "idx": str})
        for img in all_img:
            img_name = img.split(".png")[0]
            frameid, idx = img_name.split("_")
            filtered_df = df[(df["frameid"] == frameid.lstrip("0"))]
            binning = filtered_df["softbin"].iloc[0].strip()
            category = "frontal" if binning == "frontal" else "extreme"

            line = f"{os.path.join(instance_id, img)} {category}"
            if instance_id not in dict_values.keys():
                dict_values[instance_id] = {"frontal": 0, "extreme": 0, "line": [line]}
            else:
                dict_values[instance_id]["line"].append(line)
            dict_values[instance_id][category] += 1

    with open(json_path, "w") as f:
        json.dump(dict_values, f)
        f.close()

    with open(text_path, "w") as f:
        for k, v in dict_values.items():
            lines = v["line"]
            for line in lines:
                f.write(f"{line}\n")
        f.close()


if __name__ == "__main__":
    args = parse_args()
    metadata(args.bin_folder, args.source_folder, args.json_path, args.text_path)
