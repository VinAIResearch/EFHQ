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


def processing(clip_file_list, annotation_folder, type_folder, source_folder, save_folder, name):
    cuda = True
    try:
        model_hyper = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).cuda()
    except Exception:
        model_hyper = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7)
        cuda = False

    model_hyper.train(False)
    if cuda:
        model_hyper.load_state_dict(torch.load("./koniq_pretrained.pkl"))
    else:
        model_hyper.load_state_dict(torch.load("./koniq_pretrained.pkl", map_location="cpu"))

    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((512, 384)),
            torchvision.transforms.RandomCrop(size=224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    pbar = tqdm(clip_file_list)
    for clip_id in pbar:
        clip_id = clip_id.strip()
        if os.path.exists(os.path.join(annotation_folder, "txt", f"{clip_id}.txt")):
            with open(os.path.join(annotation_folder, "txt", f"{clip_id}.txt")) as f:
                annot_file = f.readlines()
                f.close()
        else:
            continue

        annot_file = [i for i in annot_file if not i.startswith("FRAME")]
        annot_file = [i.strip() for i in annot_file]

        frame_index_dict = dict()
        for i in range(len(annot_file)):
            line = annot_file[i].split()
            lmk = []
            for j in range(6, len(line)):
                lmk.append(float(line[j]))

            if line[0] not in frame_index_dict.keys():
                frame_index_dict[line[0]] = {}
            frame_index_dict[line[0]][str(line[1])] = {
                "bbox": [
                    max(0, float(line[2])),
                    max(0, float(line[3])),
                    float(line[4]),
                    float(line[5]),
                ],
                "lmk": lmk,
                "score": -1,
            }

        # HyperIQA
        for frame_index, info in frame_index_dict.items():
            for face_id, _ in info.items():
                try:
                    image_path = os.path.join(source_folder, type_folder, clip_id, f"{frame_index}.png")
                    if os.path.exists(image_path):
                        pred_scores = []
                        for i in range(10):
                            img = pil_loader(image_path)
                            img = transforms(img)
                            if cuda:
                                img = torch.tensor(img.cuda()).unsqueeze(0)
                            else:
                                img = torch.tensor(img).unsqueeze(0)
                            paras = model_hyper(img)

                            # Building target network
                            if cuda:
                                model_target = models.TargetNet(paras).cuda()
                            else:
                                model_target = models.TargetNet(paras)
                            for param in model_target.parameters():
                                param.requires_grad = False

                            # Quality prediction
                            pred = model_target(paras["target_in_vec"])
                            pred_scores.append(float(pred.item()))
                        score = np.mean(pred_scores).item()
                        frame_index_dict[frame_index][face_id]["score"] = score
                    else:
                        if os.path.exists(os.path.join(save_folder, "error", f"error_{args.name}.txt")):
                            error_file = open(os.path.join(save_folder, "error", f"error_{args.name}.txt"), "a")
                        else:
                            error_file = open(os.path.join(save_folder, "error", f"error_{args.name}.txt"), "w")
                        error_file.write(f"Not Exist Frames: {clip_id}\t{frame_index}\n")
                        error_file.close()
                except Exception as e:
                    print(e)
                    if os.path.exists(os.path.join(save_folder, "error", f"error_{args.name}.txt")):
                        error_file = open(os.path.join(save_folder, "error", f"error_{args.name}.txt"), "a")
                    else:
                        error_file = open(os.path.join(save_folder, "error", f"error_{args.name}.txt"), "w")
                    error_file.write(f"{clip_id}\t{frame_index}\t{e}\n")
                    error_file.close()

        out_path = os.path.join(save_folder, "txt", clip_id + ".txt")
        with open(out_path, "w") as f:
            f.write("FRAME INDEX X0 Y0 X1 Y1 [Landmarks (5 Points)] IQA\n")
            for frame_index, info in frame_index_dict.items():
                for face_id, v in info.items():
                    bbox = [str(x) for x in v["bbox"]]
                    lmk = [str(x) for x in v["lmk"]]
                    line_new = str(frame_index) + " " + str(face_id).zfill(8)
                    line_new += " " + " ".join(bbox)
                    line_new += " " + " ".join(lmk)
                    line_new += " " + str(v["score"])
                    f.write(f"{line_new}\n")
            f.close()

        if os.path.exists(os.path.join(save_folder, "ckpt", f"ckpt_{name}.txt")):
            ckpt_file = open(os.path.join(save_folder, "ckpt", f"ckpt_{name}.txt"), "a")
        else:
            ckpt_file = open(os.path.join(save_folder, "ckpt", f"ckpt_{name}.txt"), "w")
        ckpt_file.write(f"{clip_id}\n")
        ckpt_file.close()


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="Quality Control")
    parser.add_argument("--name", dest="name", help="name", default=False, type=str)
    parser.add_argument("--id_text", dest="id_text", help="Clips ID File", type=str)
    parser.add_argument("--annotation_folder", dest="annotation_folder", help="Retina annotation folder", type=str)
    parser.add_argument("--source_folder", dest="source_folder", help="Image folder", default=False, type=str)
    parser.add_argument("--type", dest="type", help="Train/Test", type=str)
    parser.add_argument(
        "--save_folder",
        dest="save_folder",
        help="Save hyperIQA to folder",
        default=False,
        type=str,
    )
    parser.add_argument("--num_process", dest="num_process", type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    save_folder = args.save_folder
    source_folder = args.source_folder

    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(os.path.join(save_folder, "ckpt"), exist_ok=True)
    os.makedirs(os.path.join(save_folder, "txt"), exist_ok=True)
    os.makedirs(os.path.join(save_folder, "error"), exist_ok=True)

    with open(args.id_text) as f:
        all_clip_list = f.readlines()
        f.close()

    if os.path.exists(os.path.join(save_folder, "ckpt", f"ckpt_{args.name}.txt")):
        with open(os.path.join(save_folder, "ckpt", f"ckpt_{args.name}.txt")) as f:
            done_id_list = f.readlines()
            f.close()
        all_clip_list = [i.strip() for i in all_clip_list if i not in done_id_list]
    print("Remaining :", len(all_clip_list))

    try:
        ncpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
    except KeyError:
        ncpus = cpu_count()

    chunked_list = chunk_into_n(all_clip_list, args.num_process)
    new_process = []
    for i in chunked_list:
        new_arg = (
            i,
            args.annotation_folder,
            args.type,
            args.source_folder,
            args.save_folder,
            args.name,
        )
        new_process.append(new_arg)
    with Pool(ncpus) as pool:
        pool.starmap(processing, new_process)
