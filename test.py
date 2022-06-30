from fastai.vision.all import *
from fastai import *
from fastai.vision import *

# from fastai.callbacks.hooks import *
# from fastai.callbacks import SaveModelCallback
import matplotlib.pyplot as plt
import matplotlib.image as immg
import gc
import numpy as np
import random
from PIL import Image
import warnings

# warnings.filterwarnings("ignore")
# sns.set_style('darkgrid')
base_dir = Path("train300/")
# path.ls()
# print(path.ls())
from fastai.vision.data import SegmentationDataLoaders

fnames = get_files(base_dir / "train_images/")
fnames_mask = get_files(base_dir / " train_masks/")
dls = SegmentationDataLoaders.from_label_func(
    base_dir,
    bs=8,
    fnames=get_files(base_dir / "train_images/"),
    label_func=lambda o: base_dir / "train_masks" / f"{o.stem}.png",
    codes=np.array(["background", "hand", "mat"], dtype=str),
)

import timm
import pprint

# model = timm.create_model("resnet34")
avail_pretrained_models = timm.list_models("vit*", pretrained=True)

pprint.pprint(avail_pretrained_models)
# all_densenet_models = timm.list_models()
