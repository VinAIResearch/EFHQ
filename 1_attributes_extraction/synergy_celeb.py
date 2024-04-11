import argparse
import copy
import os
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from easydict import EasyDict as edict
from FaceBoxes import FaceBoxes
from model_building import SynergyNet
from tqdm import tqdm
from utils.ddfa import Normalize, ToTensor
from utils.inference import crop_img, predict_pose, predict_sparseVert


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="Head pose estimation using the Synergy.")
    parser.add_argument("--name", dest="name", help="name", default=False, type=str)
    parser.add_argument("--id_text", help="File containing list of clipid", type=str)
    parser.add_argument(
        "--reference_folder",
        dest="reference_folder",
        help="Clips Annotation File from Retina",
        type=str,
    )
    parser.add_argument("--source_folder", dest="source_folder", help="Image folder", type=str)
    parser.add_argument("--type", dest="type", help="Train/Test", type=str, default="all")
    parser.add_argument("--save_folder", dest="save_folder", help="Save to folder", type=str)
    parser.add_argument("--num_process", dest="num_process", type=int)
    args = parser.parse_args()
    return args


def chunk_into_n(lst, n):
    import math

    size = math.ceil(len(lst) / n)
    return list(map(lambda x: lst[x * size : x * size + size], list(range(n))))


def processing(clip_file_list, type_folder, source_folder, save_folder, name, reference_folder):
    IMG_SIZE = 120
    cudnn.benchmark = True

    # preparation
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

    # face detector
    face_boxes = FaceBoxes()

    pbar = tqdm(clip_file_list)
    for clip_id in pbar:
        clip_id = clip_id.strip()
        frame_index_dict = {}
        path_to_folder = os.path.join(source_folder, type_folder, clip_id)
        if not os.path.exists(path_to_folder):
            continue

        flag = False
        if reference_folder is not None:
            path_to_ref_file = os.path.join(reference_folder, "txt", clip_id + ".txt")
            if os.path.exists(path_to_ref_file):
                with open(path_to_ref_file) as f:
                    ref_list = f.readlines()
                    f.close()
                ref_list = [i.strip() for i in ref_list]
                ref_list = [i for i in ref_list if not i.startswith("FRAME")]
                for line in ref_list:
                    try:
                        line = line.strip().split()
                        image_id = line[0]
                        faceid = line[1]

                        lmk = []
                        for j in range(6, len(line)):
                            lmk.append(float(line[j]))

                        if image_id not in frame_index_dict.keys():
                            frame_index_dict[image_id] = {}
                        bbox_annot = [
                            max(0, float(line[2])),
                            max(0, float(line[3])),
                            float(line[4]),
                            float(line[5]),
                        ]
                        frame_index_dict[image_id][str(faceid)] = {
                            "bbox": bbox_annot,
                            "lmk": lmk,
                            "yaw": None,
                            "pitch": None,
                            "roll": None,
                            "lm68": None,
                        }

                        path_to_image = os.path.join(path_to_folder, image_id + ".png")
                        if not os.path.exists(path_to_image):
                            continue
                        rect = copy.deepcopy(bbox_annot)
                        full_img = cv2.imread(path_to_image)
                        roi_box = rect
                        roi_box.append(1.0)

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

                        img = crop_img(full_img, roi_box)
                        img = cv2.resize(img, dsize=(IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

                        input_img = transform(img).unsqueeze(0)
                        with torch.no_grad():
                            input_img = input_img.cuda()
                            param = model.forward_test(input_img)
                            param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

                        # inferences
                        lmks68 = predict_sparseVert(param, roi_box, transform=True)
                        angles, _ = predict_pose(param, roi_box)
                        yaw, pitch, roll = angles

                        if image_id not in frame_index_dict.keys():
                            frame_index_dict[image_id] = {}
                        frame_index_dict[image_id][str(faceid)] = {
                            "bbox": bbox_annot,
                            "lmk": lmk,
                            "yaw": yaw,
                            "pitch": pitch,
                            "roll": roll,
                            "lm68": lmks68,
                        }
                    except Exception as e:
                        if os.path.exists(os.path.join(save_folder, "error", f"error_{name}.txt")):
                            error_file = open(os.path.join(save_folder, "error", f"error_{name}.txt"), "a")
                        else:
                            error_file = open(os.path.join(save_folder, "error", f"error_{name}.txt"), "w")
                        error_file.write(f"{clip_id} {image_id}\t{e}\n")
                        error_file.close()
            else:
                flag = True
        else:
            flag = True

        if flag:
            if not os.path.exists(path_to_folder):
                continue
            image_list = os.listdir(path_to_folder)
            image_list = [i for i in image_list if i.endswith(".jpg") or i.endswith(".png")]

            for img_name in image_list:
                path_to_image = os.path.join(path_to_folder, img_name)
                image_id = img_name.split(".png")[0] if ".png" in img_name else img_name.split(".jpg")[0]

                if not os.path.exists(path_to_image):
                    continue
                full_img = cv2.imread(path_to_image)
                try:
                    rects = face_boxes(full_img)
                    if len(rects) == 0:
                        if os.path.exists(os.path.join(save_folder, "error", "error_{name}.txt")):
                            error_file = open(os.path.join(save_folder, "error", "error_{name}.txt"), "a")
                        else:
                            error_file = open(os.path.join(save_folder, "error", "error_{name}.txt"), "w")
                        error_file.write(f"{clip_id}: {image_id}    No Face detected\n")
                        error_file.close()
                        continue

                    for idx, rect in enumerate(rects):
                        rect_annot = copy.deepcopy(rect)
                        roi_box = rect

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

                        img = crop_img(full_img, roi_box)
                        img = cv2.resize(img, dsize=(IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

                        input_img = transform(img).unsqueeze(0)
                        with torch.no_grad():
                            input_img = input_img.cuda()
                            param = model.forward_test(input_img)
                            param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

                        # inferences
                        lmks68 = predict_sparseVert(param, roi_box, transform=True)
                        angles, _ = predict_pose(param, roi_box)
                        yaw, pitch, roll = angles

                        if image_id not in frame_index_dict.keys():
                            frame_index_dict[image_id] = {}
                        frame_index_dict[image_id][str(idx)] = {
                            "bbox": [
                                str(rect_annot[0]),
                                str(rect_annot[1]),
                                str(rect_annot[2]),
                                str(rect_annot[3]),
                            ],
                            "lmk": ["None"] * 10,
                            "yaw": yaw,
                            "pitch": pitch,
                            "roll": roll,
                            "lm68": lmks68,
                        }
                except Exception as e:
                    print(f"{clip_id} {image_id} {e}")
                    if os.path.exists(os.path.join(save_folder, "error", f"error_{name}.txt")):
                        error_file = open(os.path.join(save_folder, "error", f"error_{name}.txt"), "a")
                    else:
                        error_file = open(os.path.join(save_folder, "error", f"error_{name}.txt"), "w")
                    error_file.write(f"{clip_id} {image_id}\t{e}\n")
                    error_file.close()

        if len(list(frame_index_dict.keys())) > 0:
            out_path = os.path.join(save_folder, "txt", clip_id.strip() + ".txt")
            with open(out_path, "w") as f:
                f.write("FRAME INDEX X0 Y0 X1 X1 [Landmarks (5 Points)] YAW PITCH ROLL [Landmarks (68 Points XYZ)]\n")
                for imgid, detection in frame_index_dict.items():
                    for faceid, v in detection.items():
                        bbox = [str(x) for x in v["bbox"]]
                        lmk = [str(x) for x in v["lmk"]]

                        s = str(imgid) + " " + str(faceid).zfill(8) + " " + " ".join(bbox[:4]) + " " + " ".join(lmk)
                        s += f" {v['yaw']:.3f} {v['pitch']:.3f} {v['roll']:.3f} "
                        for lm in v["lm68"].reshape(-1, 3):
                            s += f"{lm[0]:.3f} {lm[1]:.3f} {lm[2]:.3f} "
                        f.write(f"{s.strip()}\n")
                f.close()

        if os.path.exists(os.path.join(save_folder, "ckpt", f"ckpt_{name}.txt")):
            ckpt_file = open(os.path.join(save_folder, "ckpt", f"ckpt_{name}.txt"), "a")
        else:
            ckpt_file = open(os.path.join(save_folder, "ckpt", f"ckpt_{name}.txt"), "w")
        ckpt_file.write(f"{clip_id}\n")
        ckpt_file.close()


if __name__ == "__main__":
    args = parse_args()
    save_folder = args.save_folder
    source_folder = args.source_folder

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
        done_id_list = [i.strip() for i in done_id_list]
        clip_file_list = [i.strip() for i in clip_file_list if i.strip() not in done_id_list]
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
            args.name,
            args.reference_folder,
        )
        new_process.append(new_arg)
    with Pool(ncpus) as pool:
        pool.starmap(processing, new_process)
