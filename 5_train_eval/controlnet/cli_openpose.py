import random
import time
from pathlib import Path

import config
import cv2
import einops
import numpy as np
import torch
import typer
from annotator.openpose import OpenposeDetector
from annotator.util import HWC3, resize_image
from cldm.ddim_hacked import DDIMSampler
from cldm.model import create_model, load_state_dict
from path_utils import get_filelist_and_cache
from pytorch_lightning import seed_everything
from share import *
from tqdm import tqdm


preprocessor = None


def process(
    det,
    input_image,
    prompt,
    a_prompt,
    n_prompt,
    num_samples,
    image_resolution,
    detect_resolution,
    ddim_steps,
    guess_mode,
    strength,
    scale,
    seed,
    eta,
    model,
    ddim_sampler,
):
    global preprocessor

    if det is not None:
        if not isinstance(preprocessor, OpenposeDetector):
            preprocessor = OpenposeDetector()

    with torch.no_grad():
        input_image = HWC3(input_image)

        if det is None:
            detected_map = input_image.copy()
        else:
            detected_map = preprocessor(resize_image(input_image, detect_resolution), face_only=True)
            detected_map = HWC3(detected_map)

        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, "b h w c -> b c h w").clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {
            "c_concat": [control],
            "c_crossattn": [model.get_learned_conditioning([prompt + ", " + a_prompt] * num_samples)],
        }
        un_cond = {
            "c_concat": None if guess_mode else [control],
            "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)],
        }
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = (
            [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
        )
        # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01

        samples, intermediates = ddim_sampler.sample(
            ddim_steps,
            num_samples,
            shape,
            cond,
            verbose=False,
            eta=eta,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=un_cond,
        )

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (
            (einops.rearrange(x_samples, "b c h w -> b h w c") * 127.5 + 127.5)
            .cpu()
            .numpy()
            .clip(0, 255)
            .astype(np.uint8)
        )

        results = [x_samples[i] for i in range(num_samples)]
    return [detected_map] + results


app = typer.Typer()


@app.command()
def main(
    ckpt_path: Path = typer.Argument(..., help="ckpt path", file_okay=True, exists=True),
    condition_image_dir: Path = typer.Argument(..., help="Condition_image", dir_okay=True, exists=True),
    prompt: str = typer.Argument(..., help="prompt"),
    outdir: Path = typer.Argument(..., help="out dir", dir_okay=True),
    need_detect: bool = typer.Option(False, help="If true then run open pose to draw condition image"),
    copy_condition: bool = typer.Option(False, help="If true then copy condition image to output folder"),
    negative_prompt: str = typer.Option(
        "disfigured, ugly, bad, immature, cartoon, anime, 3d, painting, b&w",
        help="negative prompt",
    ),
    added_prompt: str = typer.Option(
        "rim lighting, studio lighting, dslr, ultra quality, sharp focus, tack sharp, dof, film grain, Fujifilm XT3, crystal clear, 8K UHD, highly detailed glossy eyes, high detailed skin, skin pores",
        help="added prompt",
    ),
    inference_step: int = typer.Option(20, help="inference  step"),
    nsamples: int = typer.Option(1, help="num samples"),
    strength: float = typer.Option(1.0, help="strength"),
    guidance_scale: float = typer.Option(7.0, help="guidance_scale"),
    seed: int = typer.Option(0, help="seed for random number generator"),
    image_resolution: int = typer.Option(512, help="image_resolution"),
    eta: float = typer.Option(1.0, help="eta"),
    batch_idx: int = typer.Option(-1, help="batch_idx of chunk"),
    n_chunks: int = typer.Option(-1, help="how many chunk for this dir"),
):

    model_name = "control_v11p_sd15_openpose"
    model = create_model(f"./models/{model_name}.yaml").cpu()
    model.load_state_dict(load_state_dict("./models/v1-5-pruned.ckpt", location="cuda"), strict=False)
    model.load_state_dict(load_state_dict(ckpt_path, location="cuda"), strict=False)
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)

    if condition_image_dir.is_file():
        img_paths = [condition_image_dir]
    else:
        img_paths = get_filelist_and_cache(condition_image_dir, "*.[jp][pn]g")
    det = "Openpose" if need_detect else None
    assert strength >= 0.0 and strength <= 2.0, f"Invalid strength: {strength} (should be in [0.0,2.0])"
    assert eta >= 0.0 and eta <= 1.0, f"Invalid eta: {eta} (should be in [0.0,1.0])"
    if n_chunks == -1:
        final = img_paths
        pbar_str = "Full"
    else:
        assert batch_idx < n_chunks and batch_idx >= 0, f"Invalid batch_idx: {batch_idx} [{0}, {n_chunks-1}]"
        chunks = np.array_split(img_paths, n_chunks)
        final = chunks[batch_idx]
        pbar_str = f"Chunk {batch_idx}/{n_chunks}"
    pbar = tqdm(final)
    pbar.set_description(pbar_str)
    for img_path in pbar:
        relapath = img_path.relative_to(condition_image_dir)
        if relapath == Path("."):
            relapath = img_path.stem
        else:
            relapath = relapath.with_suffix("")
        cur_output_dir = outdir / relapath
        if cur_output_dir.exists():
            continue
        cur_output_dir.mkdir(exist_ok=True, parents=True)

        img = cv2.imread(img_path.as_posix())
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        outputs = process(
            det,
            img,
            prompt,
            added_prompt,
            negative_prompt,
            nsamples,
            image_resolution,
            image_resolution,
            inference_step,
            False,
            strength,
            guidance_scale,
            seed,
            eta,
            model,
            ddim_sampler,
        )
        for i, output in enumerate(outputs):
            if i == 0 and copy_condition:
                cur_outpath = cur_output_dir / "condition.png"
                continue
            cur_outpath = cur_output_dir / f"{time.time()}.png"
            output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            cv2.imwrite(cur_outpath.as_posix(), output)


if __name__ == "__main__":
    app()
