import os
import random
import shutil
from pathlib import Path

SOURCE_DIR = Path("G:/Orni_forest/forest_changes_dataset/generated_dataset_burns/train")
DEST_DIR = Path("G:/Orni_forest/forest_changes_dataset/generated_dataset_burns/validation")

DEST_DIR.mkdir(parents=True, exist_ok=True)

groups = {}
for file_path in SOURCE_DIR.iterdir():
    if file_path.is_file():
        prefix = file_path.stem.split("_")[0]
        if prefix not in groups:
            groups[prefix] = []
        groups[prefix].append(file_path)

all_prefixes = list(groups.keys())
num_prefixes = len(all_prefixes)
sample_size = int(0.2 * num_prefixes)  # 20%
random.seed(42)
selected_prefixes = random.sample(all_prefixes, sample_size)

for prefix in selected_prefixes:
    for file_path in groups[prefix]:
        dest_file_path = DEST_DIR / file_path.name

        shutil.move(str(file_path), str(dest_file_path))

print(f"Moved {len(selected_prefixes)} prefixes (with all associated files) to {DEST_DIR}")
