import json
from pathlib import Path

import torch
from tqdm import tqdm

DATASET_PATH = Path("/netscratch/vemburaj/Novel_View_Synthesis/datasets/DL3DV-10K-Chunks/480P")
STAGE = ["train"]

if __name__ == "__main__":
    # "train" or "test"
    for stage in STAGE:
        stage = DATASET_PATH / stage

        index = {}
        for chunk_path in tqdm(
            sorted(list(stage.iterdir())), desc=f"Indexing {stage.name}"
        ):
            if chunk_path.suffix == ".torch":
                chunk = torch.load(chunk_path)
                for example in chunk:
                    index[example["key"]] = str(chunk_path.relative_to(stage))
        with (stage / "index.json").open("w") as f:
            json.dump(index, f)
