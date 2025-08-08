import random, shutil, pathlib

BASE = pathlib.Path(__file__).parent.parent / "data"
SRC  = BASE / "raw"
TR   = BASE / "train"
TE   = BASE / "test"

TR.mkdir(exist_ok=True)
TE.mkdir(exist_ok=True)

files = list(SRC.glob("*.txt"))
random.shuffle(files)
n_train = int(len(files) * 0.8)

for f in files[:n_train]:
    shutil.move(str(f), TR / f.name)
for f in files[n_train:]:
    shutil.move(str(f), TE / f.name)

print(f"Split {len(files)} files into train={n_train} and test={len(files)-n_train}")