import numpy as np
from RayS_Single import RayS
from gvision_model import GVisionModel, label_list_decision, gvision_classify
from PIL import Image
import torchvision.transforms.functional as TF
import os
import utils
import functools
import pathlib


image = Image.open('test_images/cat.png')
x = TF.to_tensor(image)

# drop alpha channel
x = x[:3]

# make it easier by targetting only 'cat' label
cat_decision = functools.partial(label_list_decision, label_set = ["cat"])


for query_limit in [25, 50, 100, 200, 400, 800, 1600, 3200]:
    DIR = f"output/cat_attack_relaxed"
    pathlib.Path(DIR).mkdir(parents=True, exist_ok=True)

    model = GVisionModel(decision_fn=cat_decision)
    attack = RayS(model=model, order=np.inf, early_stopping=False)
    x_adv, queries, dist, succ = attack(x, 0, target=1, query_limit=query_limit)

    exp_name = f"cat_queries={query_limit}_dist={dist}"
    utils.save_img_tensor(x_adv, f"{DIR}/{exp_name}.png")

    # save final classification labels and scores
    labels, scores = gvision_classify(x_adv)
    with open(f"{DIR}/{exp_name}_results.txt", "w") as f:
        f.write("\n".join([l + ": " + str(s)  for l, s in zip(labels, scores)]))







