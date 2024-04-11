import argparse
import math
import os
from multiprocessing import Pool, cpu_count

# sys.path.append("hyperIQA/")
# from hyperIQA import models
import models
import numpy as np
import torch
import torchvision
from PIL import Image
from tqdm import tqdm


def pil_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def chunk_into_n(lst, n):
    size = math.ceil(len(lst) / n)
    return list(map(lambda x: lst[x * size : x * size + size], list(range(n))))


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="Quality Control")
    parser.add_argument("--name", dest="name", help="name", default=False, type=str)
    parser.add_argument("--id_text", dest="id_text", help="Clips ID File", type=str)
    parser.add_argument("--annotation_folder", dest="annotation_folder", help="Clips Annotation folder", type=str)
    parser.add_argument("--source_folder", dest="source_folder", help="Image folder", type=str)
    parser.add_argument("--type", dest="type", help="Train/Test", type=str)
    parser.add_argument("--save_folder", dest="save_folder", help="Save hyperIQA to folder", type=str)
    parser.add_argument("--num_process", dest="num_process", type=int)
    args = parser.parse_args()
    return args


def processing(clip_file_list, annotation_folder, type_folder, source_folder, save_folder):
    model_hyper = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).cuda()
    model_hyper.train(False)
    model_hyper.load_state_dict(torch.load("./koniq_pretrained.pkl"))

    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((512, 384)),
            torchvision.transforms.RandomCrop(size=224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    pbar = tqdm(clip_file_list)
    for annot_path in pbar:
        annot_path = annot_path.strip()
        if os.path.exists(os.path.join(annotation_folder, type_folder, f"{annot_path}.txt")):
            with open(os.path.join(annotation_folder, type_folder, f"{annot_path}.txt")) as f:
                annot_file = f.readlines()
                f.close()
        else:
            continue

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
            # get the coordinates of face
            if line.startswith("CROP"):
                clip_crop_bbox = line

        _, video_id, pid, clip_idx, frame_rlt = annot_path.split("+")

        frame_index_dict = dict()
        for i in range(7, len(annot_file) - 1):
            line = annot_file[i].split()
            lmk = []
            for j in range(6, len(line)):
                lmk.append(float(line[j]))

            frame_index_dict[line[0]] = {
                "frameid": line[0],
                "index": pid[-1],  # Same id
                "bbox": [
                    float(line[2]),
                    float(line[3]),
                    float(line[2]) + float(line[4]),
                    float(line[3]) + float(line[5]),
                ],
                "lmk": lmk,
                "score": -1,
            }

        # HyperIQA
        for frame_index, _ in frame_index_dict.items():
            try:
                image_path = os.path.join(source_folder, type_folder, video_id, pid, clip_idx, f"{frame_index}.png")
                if os.path.exists(image_path):
                    pred_scores = []
                    for i in range(10):
                        img = pil_loader(image_path)
                        img = transforms(img)
                        img = torch.tensor(img.cuda()).unsqueeze(0)
                        paras = model_hyper(img)

                        # Building target network
                        model_target = models.TargetNet(paras).cuda()
                        for param in model_target.parameters():
                            param.requires_grad = False

                        # Quality prediction
                        pred = model_target(paras["target_in_vec"])
                        pred_scores.append(float(pred.item()))
                    score = np.mean(pred_scores).item()
                    frame_index_dict[frame_index]["score"] = score
                else:
                    if os.path.exists(os.path.join(save_folder, "error", f"error_{args.name}.txt")):
                        error_file = open(os.path.join(save_folder, "error", f"error_{args.name}.txt"), "a")
                    else:
                        error_file = open(os.path.join(save_folder, "error", f"error_{args.name}.txt"), "w")
                    error_file.write(f"Not Exist Frames: {video_id}\t{frame_index}\n")
                    error_file.close()
            except Exception as e:
                print(e)
                if os.path.exists(os.path.join(save_folder, "error", f"error_{args.name}.txt")):
                    error_file = open(os.path.join(save_folder, "error", f"error_{args.name}.txt"), "a")
                else:
                    error_file = open(os.path.join(save_folder, "error", f"error_{args.name}.txt"), "w")
                error_file.write(f"{video_id}\t{frame_index}\t{e}\n")
                error_file.close()
        out_path = os.path.join(save_folder, "txt", annot_path + ".txt")
        with open(out_path, "w") as f:
            f.write(f"{clip_video_txt}\n")
            f.write(f"{clip_height_txt}\n")
            f.write(f"{clip_width_txt}\n")
            f.write(f"{clip_fps_txt}\n")
            f.write("\n")
            f.write("FRAME INDEX X0 Y0 X1 Y1 [Landmarks (5 Points)] IQA\n")
            f.write("\n")
            for k, v in frame_index_dict.items():
                bbox = [str(x) for x in v["bbox"]]
                lmk = [str(x) for x in v["lmk"]]
                if v["score"] != -1:
                    line_new = v["frameid"] + " " + str(v["index"]).zfill(8)
                    line_new += " " + " ".join(bbox)
                    line_new += " " + " ".join(lmk) + " " + str(v["score"])
                else:
                    line_new = v["frameid"] + " " + str(v["index"]).zfill(8)
                    line_new += " " + " ".join(bbox)
                    line_new += " " + " ".join(lmk) + " " + "NA"
                f.write(f"{line_new}\n")
            f.write(f"{clip_crop_bbox}\n")
            f.close()
        if os.path.exists(os.path.join(save_folder, "ckpt", f"ckpt_{args.name}.txt")):
            ckpt_file = open(os.path.join(save_folder, "ckpt", f"ckpt_{args.name}.txt"), "a")
        else:
            ckpt_file = open(os.path.join(save_folder, "ckpt", f"ckpt_{args.name}.txt"), "w")
        ckpt_file.write(f"{annot_path}\n")
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

    # print(len(clip_file_list))
    if os.path.exists(os.path.join(save_folder, "ckpt", f"ckpt_{args.name}.txt")):
        with open(os.path.join(save_folder, "ckpt", f"ckpt_{args.name}.txt")) as f:
            done_id_list = f.readlines()
            f.close()
        clip_file_list = [i.strip() for i in clip_file_list if i not in done_id_list]
    print("Remaining :", len(clip_file_list))

    try:
        ncpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
    except KeyError:
        ncpus = cpu_count()

    chunked_list = chunk_into_n(clip_file_list, args.num_process)
    new_process = []
    for i in chunked_list:
        new_arg = (i, args.annotation_folder, args.type, source_folder, save_folder)
        new_process.append(new_arg)
    with Pool(ncpus) as pool:
        pool.starmap(processing, new_process)
