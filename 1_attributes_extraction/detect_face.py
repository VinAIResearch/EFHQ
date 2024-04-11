import argparse
import glob
import os

import cv2
from face_detection import RetinaFace
from tqdm import tqdm


CONF_THRESHOLD = 0.9
AREA_THRESHOLD = 256 * 256


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="Retina + Head pose estimation using the 6DRepNet.")
    parser.add_argument("--source_folder", dest="source_folder", help="Image folder", default=False, type=str)
    parser.add_argument("--save_folder", dest="save_folder", help="Save bbox to folder", default=False, type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.save_folder, exist_ok=True)
    os.makedirs(os.path.join(args.save_folder, "txt"), exist_ok=True)
    os.makedirs(os.path.join(args.save_folder, "error"), exist_ok=True)
    os.makedirs(os.path.join(args.save_folder, "ckpt"), exist_ok=True)

    detector = RetinaFace(gpu_id=0)
    all_clip_list = glob.glob(os.path.join(args.source_folder, "*"))

    if os.path.exists(os.path.join(args.save_folder, "ckpt", "ckpt.txt")):
        with open(os.path.join(args.save_folder, "ckpt", "ckpt.txt")) as f:
            clip_ckpt = f.readlines()
            f.close()
        clip_ckpt = [os.path.join(args.source_folder, i.strip()) for i in clip_ckpt]
        all_clip_list = [i for i in all_clip_list if i not in clip_ckpt]

    for clip_path in tqdm(all_clip_list):
        clip_id = clip_path.split("/")[-1]
        images_path = glob.glob(os.path.join(clip_path, "*"))
        frame_index_dict = {}
        for path in images_path:
            try:
                image_id = path.split("/")[-1].split(".png")[0]
                full_image = cv2.imread(path)
                faces = detector(full_image)
                if len(faces) == 0:
                    if os.path.exists(os.path.join(args.save_folder, "error", "error.txt")):
                        error_file = open(os.path.join(args.save_folder, "error", "error.txt"), "a")
                    else:
                        error_file = open(os.path.join(args.save_folder, "error", "error.txt"), "w")
                    error_file.write(f"{clip_id}: {image_id}.png No Face\n")
                    error_file.close()
                    continue

                face_id = 0
                for box, landmarks, score in faces:
                    if score < CONF_THRESHOLD:
                        continue
                    x_min = int(box[0])
                    y_min = int(box[1])
                    x_max = int(box[2])
                    y_max = int(box[3])

                    bbox = [max(0, x_min), max(0, y_min), x_max, y_max]
                    lmk = landmarks
                    bbox_width = abs(x_max - x_min)
                    bbox_height = abs(y_max - y_min)

                    if bbox_width * bbox_height < AREA_THRESHOLD:
                        continue

                    # Expand crop regions
                    x_min = max(0, x_min - int(0.2 * bbox_height))
                    y_min = max(0, y_min - int(0.2 * bbox_width))
                    x_max = x_max + int(0.2 * bbox_height)
                    y_max = y_max + int(0.2 * bbox_width)

                    if image_id not in frame_index_dict.keys():
                        frame_index_dict[image_id] = {}
                    frame_index_dict[image_id][str(face_id)] = {"bbox": box, "landmark": lmk}
                    face_id += 1
            except Exception as e:
                if os.path.exists(os.path.join(args.save_folder, "error", "error.txt")):
                    error_file = open(os.path.join(args.save_folder, "error", "error.txt"), "a")
                else:
                    error_file = open(os.path.join(args.save_folder, "error", "error.txt"), "w")
                error_file.write(f"{clip_id}: {image_id}.png {e}\n")
                error_file.close()
                continue

        if len(frame_index_dict.keys()) > 0:
            out_path = os.path.join(args.save_folder, "txt", clip_id + ".txt")
            with open(out_path, "w") as f:
                f.write("FRAME INDEX X1 Y1 X2 Y2 [Landmarks (5 Points)]\n")
                for image_id, v1 in frame_index_dict.items():
                    for fid, v2 in v1.items():
                        s = image_id + " " + fid.zfill(8)
                        bbox = [str(x) for x in v2["bbox"][:4]]
                        s += " " + " ".join(bbox)
                        lmk = v2["landmark"].flatten().tolist()
                        lmk = [str(x) for x in lmk]
                        s += " " + " ".join(lmk)
                        f.write(f"{s}\n")
                f.close()

        if os.path.exists(os.path.join(args.save_folder, "ckpt", "ckpt.txt")):
            ckpt_file = open(os.path.join(args.save_folder, "ckpt", "ckpt.txt"), "a")
        else:
            ckpt_file = open(os.path.join(args.save_folder, "ckpt", "ckpt.txt"), "w")
        ckpt_file.write(f"{clip_id}\n")
        ckpt_file.close()
