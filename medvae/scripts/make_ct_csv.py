"""Build ct_data.csv (train/val/test splits) for the CT finetuning dataloader.

image_uuid = filename without the .nii.gz suffix (filenames contain dots, so we
strip the exact suffix rather than splitting on '.').  Splits come from the
`split` column of stanford_labels.csv, matched on the basename of `image_file`:
    split in {0,1,2} -> train, 3 -> val, 4 -> test.
"""
import os
import pandas as pd

CT_DIR = "/dataNAS/people/akkumar/Downloads/MedVAE/medvae/data/ct_data"
LABELS = "/dataNAS/people/lblankem/contrastive-3d/data/stanford_labels.csv"
OUT = "/dataNAS/people/akkumar/Downloads/MedVAE/medvae/data/ct_data.csv"
SUFFIX = ".nii.gz"

SPLIT_MAP = {0: "train", 1: "train", 2: "train", 3: "val", 4: "test"}

# basename(image_file) -> split label
labels = pd.read_csv(LABELS, usecols=["image_file", "split"])
uuid_to_split = {}
for image_file, split in zip(labels["image_file"], labels["split"]):
    base = os.path.basename(str(image_file))
    if base.endswith(SUFFIX):
        base = base[: -len(SUFFIX)]
    uuid_to_split[base] = int(split)

files = sorted(f for f in os.listdir(CT_DIR) if f.endswith(SUFFIX))
rows, missing = [], []
for uuid in (f[: -len(SUFFIX)] for f in files):
    if uuid in uuid_to_split:
        rows.append({"image_uuid": uuid, "split": SPLIT_MAP[uuid_to_split[uuid]]})
    else:
        missing.append(uuid)

df = pd.DataFrame(rows)
df.insert(0, "row_nr", range(len(df)))
df.to_csv(OUT, index=False)

print(f"ct_data files: {len(files)}")
print(f"matched: {len(rows)}   unmatched: {len(missing)}")
print("split counts:")
print(df["split"].value_counts())
print(f"wrote {OUT}")
if missing:
    print(f"first few unmatched: {missing[:5]}")
