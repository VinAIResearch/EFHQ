import argparse
import math
import os
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from easydict import EasyDict as edict

# sys.path.append("SynergyNet/")
from model_building import SynergyNet
from tqdm import tqdm
from utils.ddfa import Normalize, ToTensor
from utils.inference import crop_img, predict_pose, predict_sparseVert


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="Head pose estimation using the Synergy.")
    parser.add_argument("--name", dest="name", help="name", default=False, type=str)
    parser.add_argument("--id_text", dest="id_text", help="Clips ID File", type=str)
    parser.add_argument("--annotation_folder", dest="annotation_folder", help="Clips Annotation folder", type=str)
    parser.add_argument("--source_folder", dest="source_folder", help="Image folder", type=str)
    parser.add_argument("--type", dest="type", help="Train/Test", type=str)
    parser.add_argument("--save_folder", dest="save_folder", help="Save to folder", type=str)
    parser.add_argument("--num_process", dest="num_process", type=int)
    args = parser.parse_args()
    return args


def chunk_into_n(lst, n):
    import math

    size = math.ceil(len(lst) / n)
    return list(map(lambda x: lst[x * size : x * size + size], list(range(n))))


def processing(clip_file_list, type_folder, source_folder, save_folder, annotation_folder, name):
    IMG_SIZE = 120
    cudnn.benchmark = True

    transform = transforms.Compose([ToTensor(), Normalize(mean=127.5, std=128)])

    checkpoint_fp = "./pretrained/best.pth.tar"
    args = edict({"arch": "mobilenet_v2", "devices_id": [0], "img_size": IMG_SIZE})

    checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)["state_dict"]
    model = SynergyNet(args)
    model_dict = model.state_dict()

    # because the model is trained by multiple gpus, prefix 'module' should be removed
    for k in checkpoint.keys():
        model_dict[k.replace("module.", "")] = checkpoint[k]

    model.load_state_dict(model_dict, strict=False)
    model = model.cuda()
    model.eval()

    pbar = tqdm(clip_file_list)
    for annot_path in pbar:
        annot_path = annot_path.strip()
        with open(os.path.join(f"{annotation_folder}/{type_folder}", f"{annot_path}.txt")) as f:
            annot_file = f.readlines()
            f.close()

        annot_file = [i.strip() for i in annot_file]
        for line in annot_file:
            if line.startswith("Video"):
                clip_video_txt = line
            if line.startswith("H"):
                clip_height_txt = line
            if line.startswith("W"):
                clip_width_txt = line
            if line.startswith("FPS"):
                clip_fps_txt = line
            if line.startswith("CROP"):
                clip_crop_bbox = line

        _, video_id, pid, clip_idx, frame_rlt = annot_path.split("+")
        frame_index_dict = dict()
        for i in range(7, len(annot_file) - 1):
            line = annot_file[i].strip().split()
            lmk = []
            for j in range(6, len(line)):
                lmk.append(float(line[j]))
            try:
                bbox_annot = [
                    float(line[2]),
                    float(line[3]),
                    float(line[2]) + float(line[4]),
                    float(line[3]) + float(line[5]),
                ]
            except Exception:
                bbox_annot = None
            frame_index_dict[line[0]] = {
                "frameid": line[0],
                "index": pid[-1],  # Same id
                "bbox": bbox_annot,
                "lmk": lmk,
                "pitch": None,
                "yaw": None,
                "roll": None,
                "lm68": None,
            }

        # Pose Estimation
        for frame_index, values in frame_index_dict.items():
            try:
                # frame = int(frame_index)
                rect = values["bbox"]
                x1, y1, x2, y2 = rect
                rect = [math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)]

                # Full Image
                image_path = os.path.join(source_folder, type_folder, video_id, annot_path, f"{frame_index}.png")
                if os.path.exists(image_path) and rect is not None:
                    roi_box = rect
                    roi_box.append(1.0)  # Confidence score
                    full_image = cv2.imread(image_path)

                    # enlarge the bbox a little and do a square crop
                    HCenter = (rect[1] + rect[3]) / 2
                    WCenter = (rect[0] + rect[2]) / 2
                    side_len = roi_box[3] - roi_box[1]
                    margin = side_len * 1.2 // 2
                    roi_box[0], roi_box[1], roi_box[2], roi_box[3] = (
                        WCenter - margin,
                        HCenter - margin,
                        WCenter + margin,
                        HCenter + margin,
                    )

                    img = crop_img(full_image, roi_box)
                    img = cv2.resize(img, dsize=(IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

                    input = transform(img).unsqueeze(0)
                    with torch.no_grad():
                        input = input.cuda()
                        param = model.forward_test(input)
                        param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

                    # inferences
                    lmks68 = predict_sparseVert(param, roi_box, transform=True)
                    angles, _ = predict_pose(param, roi_box)
                    yaw, pitch, roll = angles

                    print(
                        f"{annot_path} {frame_index}:  {angles[0]:.3f} {angles[1]:.3f} {angles[2]:.3f}",
                        flush=True,
                    )

                    frame_index_dict[frame_index]["yaw"] = yaw
                    frame_index_dict[frame_index]["pitch"] = pitch
                    frame_index_dict[frame_index]["roll"] = roll
                    frame_index_dict[frame_index]["lm68"] = lmks68

                else:
                    if os.path.exists(os.path.join(save_folder, "error", f"error_{name}.txt")):
                        error_file = open(os.path.join(save_folder, "error", f"error_{name}.txt"), "a")
                    else:
                        error_file = open(os.path.join(save_folder, "error", f"error_{name}.txt"), "w")
                    error_file.write(f"Not Exist Frames: {video_id}\t{frame_index}\n")
                    error_file.close()
            except Exception as e:
                print(e)
                if os.path.exists(os.path.join(save_folder, "error", f"error_{name}.txt")):
                    error_file = open(os.path.join(save_folder, "error", f"error_{name}.txt"), "a")
                else:
                    error_file = open(os.path.join(save_folder, "error", f"error_{name}.txt"), "w")
                error_file.write(f"{video_id}\t{frame_index}\t{e}\n")
                error_file.close()

        out_path = os.path.join(save_folder, "txt", annot_path.strip() + ".txt")
        with open(out_path, "w") as f:
            f.write(f"{clip_video_txt}\n")
            f.write(f"{clip_height_txt}\n")
            f.write(f"{clip_width_txt}\n")
            f.write(f"{clip_fps_txt}\n")
            f.write("\n")
            f.write("FRAME INDEX X0 Y0 X1 X1 [Landmarks (5 Points)] YAW PITCH ROLL [Landmarks (68 Points XYZ)]\n")
            f.write("\n")
            for k, v in frame_index_dict.items():
                bbox = [str(x) for x in v["bbox"][:4]]
                lmk = [str(x) for x in v["lmk"]]
                s = v["frameid"] + " " + str(v["index"]).zfill(8)
                s += " " + " ".join(bbox) + " " + " ".join(lmk)

                if v["yaw"] is not None:
                    s += f" {v['yaw']:.3f} {v['pitch']:.3f} {v['roll']:.3f} "
                    for lm in v["lm68"].reshape(-1, 3):
                        s += f"{lm[0]:.3f} {lm[1]:.3f} {lm[2]:.3f} "
                else:
                    print(f"{annot_path} {k}")
                f.write(f"{s.strip()}\n")
            f.write(f"{clip_crop_bbox}\n")
            f.close()
        if os.path.exists(os.path.join(save_folder, "ckpt", f"ckpt_{name}.txt")):
            ckpt_file = open(os.path.join(save_folder, "ckpt", f"ckpt_{name}.txt"), "a")
        else:
            ckpt_file = open(os.path.join(save_folder, "ckpt", f"ckpt_{name}.txt"), "w")
        ckpt_file.write(f"{annot_path}\n")
        ckpt_file.close()


if __name__ == "__main__":
    args = parse_args()
    save_folder = args.save_folder
    source_folder = args.source_folder

    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)
    os.makedirs(os.path.join(save_folder, "ckpt"), exist_ok=True)
    os.makedirs(os.path.join(save_folder, "txt"), exist_ok=True)
    os.makedirs(os.path.join(save_folder, "error"), exist_ok=True)

    with open(args.id_text) as f:
        clip_file_list = f.readlines()
        f.close()

    if os.path.exists(os.path.join(save_folder, "ckpt", f"ckpt_{args.name}.txt")):
        with open(os.path.join(save_folder, "ckpt", f"ckpt_{args.name}.txt")) as f:
            done_id_list = f.readlines()
            f.close()
        clip_file_list = [i.strip() for i in clip_file_list if i not in done_id_list]
    print("Remaining: ", len(clip_file_list), flush=True)

    try:
        ncpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
    except KeyError:
        ncpus = cpu_count()

    chunked_list = chunk_into_n(clip_file_list, args.num_process)

    new_process = []
    for i in chunked_list:
        new_arg = (
            i,
            args.type,
            args.source_folder,
            args.save_folder,
            args.annotation_folder,
            args.name,
        )
        new_process.append(new_arg)
    with Pool(ncpus) as pool:
        pool.starmap(processing, new_process)
