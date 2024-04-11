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
    parser.add_argument("--type_folder", dest="type_folder", help="Train/Test", type=str)
    args = parser.parse_args()
    return args


def metadata(bin_folder, source_folder, json_path, text_path, type_folder):
    # source_folder: extracted_cropped_face_results_vox_final
    # type_folder: all
    # bin_folder: output/CelebV-HQ/merge/4_binned_edited
    root = os.path.join(source_folder, type_folder)  # extracted_cropped_face_results_vox_final/all
    all_video = os.listdir(root)
    dict_values = {}
    for video_id in all_video:
        all_person_id = os.listdir(
            os.path.join(root, video_id)
        )  # extracted_cropped_face_results_vox_final/all/1id56m17lko
        for person_id in all_person_id:
            all_clip_id = os.listdir(
                os.path.join(root, video_id, person_id)
            )  # extracted_cropped_face_results_vox_final/all/1id56m17lko/id0
            for clip_id in all_clip_id:
                all_img = os.listdir(
                    os.path.join(root, video_id, person_id, clip_id)
                )  # extracted_cropped_face_results_vox_final/all/1id56m17lko/id0/1id56m17lko_0
                df = pd.read_csv(
                    os.path.join(bin_folder, type_folder, clip_id + ".csv"),
                    dtype={"frameid": str, "idx": str},
                )
                for img in all_img:
                    img_name = img.split(".png")[0]
                    frameid, idx = img_name.split("_")
                    filtered_df = df[(df["frameid"] == frameid) & (df["idx"] == idx)]

                    binning = filtered_df["softbin"].iloc[0].strip()
                    category = "frontal" if binning == "frontal" else "extreme"
                    line = f"{os.path.join(video_id, person_id, clip_id, img)} {category}"

                    if os.path.join(video_id, person_id, clip_id) not in dict_values.keys():
                        dict_values[os.path.join(video_id, person_id, clip_id)] = {
                            "frontal": 0,
                            "extreme": 0,
                            "line": [line],
                        }
                    else:
                        dict_values[os.path.join(video_id, person_id, clip_id)]["line"].append(line)
                    dict_values[os.path.join(video_id, person_id, clip_id)][category] += 1

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
    metadata(args.bin_folder, args.source_folder, args.json_path, args.text_path, args.type_folder)
