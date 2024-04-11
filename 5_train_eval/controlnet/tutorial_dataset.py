import json
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm


class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        target_dir = "../../data/sampled/controlnet/target_images"
        source_dir = "../../data/sampled/controlnet/condition_images"
        source_dir = Path(source_dir)
        target_dir = Path(target_dir)
        with open("../../data/sampled/controlnet/prompts.json") as f:
            prompts = json.load(f)
            for filename, prompt in tqdm(prompts.items(), total=len(prompts)):
                filename_with_suffix = filename + ".png"
                source_path = source_dir / filename_with_suffix
                target_path = target_dir / filename_with_suffix
                assert target_path.exists(), f"Target path: {target_path} not exist"
                if not source_path.exists():
                    continue

                source_img = source_path.as_posix()
                target_img = target_path.as_posix()

                self.data.append(
                    {
                        "source": source_img,
                        "target": target_img,
                        "prompt": prompt,
                    }
                )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item["source"]
        target_filename = item["target"]
        prompt = item["prompt"]

        source = cv2.imread(source_filename)
        target = cv2.imread(target_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)
