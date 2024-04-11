import os

import imageio
import numpy as np
import torch
from logger import Logger, Visualizer
from torch.utils.data import DataLoader
from tqdm import tqdm


def reconstruction(
    config, inpainting_network, kp_detector, bg_predictor, dense_motion_network, checkpoint, log_dir, dataset
):
    if checkpoint is not None:
        Logger.load_cpk(
            checkpoint,
            inpainting_network=inpainting_network,
            kp_detector=kp_detector,
            bg_predictor=bg_predictor,
            dense_motion_network=dense_motion_network,
        )
    else:
        raise AttributeError("Checkpoint should be specified for mode='reconstruction'.")

    is_cross = dataset.cross
    if is_cross:
        png_dir = os.path.join(log_dir, "reconstruction_cross/png")
        log_dir = os.path.join(log_dir, "reconstruction_cross")
    else:
        png_dir = os.path.join(log_dir, "reconstruction/png")
        log_dir = os.path.join(log_dir, "reconstruction")

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    loss_list = []
    inpainting_network.eval()
    kp_detector.eval()
    dense_motion_network.eval()
    if bg_predictor:
        bg_predictor.eval()

    for it, x in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            predictions = []
            originals = []
            visualizations = []
            if "skip" in x.keys():
                continue
            data_name = x["data_name"][0]
            if data_name != "voxceleb1" and data_name != "voxceleb1_org":
                data_name = "extremeHQ"
                x["name"][0] = "_".join(x["name"][0].split("/"))
            if torch.cuda.is_available():
                x["video"] = x["video"].cuda()
            kp_source = kp_detector(x["video"][:, :, 0])
            for frame_idx in range(min(x["video"].shape[2], 1000)):
                source = x["video"][:, :, 0]
                driving = x["video"][:, :, frame_idx]
                kp_driving = kp_detector(driving)
                bg_params = None
                if bg_predictor:
                    bg_params = bg_predictor(source, driving)

                dense_motion = dense_motion_network(
                    source_image=source,
                    kp_driving=kp_driving,
                    kp_source=kp_source,
                    bg_param=bg_params,
                    dropout_flag=False,
                )
                out = inpainting_network(source, dense_motion)
                out["kp_source"] = kp_source
                out["kp_driving"] = kp_driving

                originals.append(np.transpose(driving.cpu().numpy(), [0, 2, 3, 1])[0])
                if frame_idx == 0 and is_cross:
                    predictions.append(np.transpose(driving.cpu().numpy(), [0, 2, 3, 1])[0])
                else:
                    predictions.append(np.transpose(out["prediction"].data.cpu().numpy(), [0, 2, 3, 1])[0])

                if not is_cross:
                    visualization = Visualizer(**config["visualizer_params"]).visualize(
                        source=source, driving=driving, out=out
                    )
                    visualizations.append(visualization)
                    loss = torch.abs(out["prediction"] - driving).mean().cpu().numpy()
                    loss_list.append(loss)

            image_name = x["name"][0] + ".mp4"
            predictions = np.concatenate(predictions, axis=1)
            originals = np.concatenate(originals, axis=1)

            predictions = np.clip(predictions, 0, 1)

            if not is_cross:
                os.makedirs(os.path.join(log_dir, "viz", data_name), exist_ok=True)
            os.makedirs(os.path.join(png_dir, data_name, "gt"), exist_ok=True)
            os.makedirs(os.path.join(png_dir, data_name, "preds"), exist_ok=True)

            imageio.imsave(
                os.path.join(png_dir, data_name, "preds", x["name"][0] + ".png"), (255 * predictions).astype(np.uint8)
            )
            imageio.imsave(
                os.path.join(png_dir, data_name, "gt", x["name"][0] + ".png"), (255 * originals).astype(np.uint8)
            )
            if not is_cross:
                imageio.mimsave(os.path.join(log_dir, "viz", data_name, image_name), visualizations)
                print("Reconstruction loss: %s" % np.mean(loss_list))
    if not is_cross:
        print("Reconstruction loss: %s" % np.mean(loss_list))
    return loss_list
