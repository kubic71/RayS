import numpy as np
from RayS_Single import RayS
from gvision_model import GVisionModel, label_set_match, gvision_classify
from PIL import Image
import torchvision.transforms.functional as TF
import os
import utils
import functools
import pathlib


image = Image.open('test_images/shark.png')
x = TF.to_tensor(image)

# drop alpha channel
x = x[:3]

def shark_top5_decision(labels, scores):
    shark_label_set = ["Shark", "Fin", "Water", "Fish", "Carcharhiniformes", "Lamnidae", "Lamniformes"]
    return not label_set_match(shark_label_set, labels[:5])

for query_limit in [25, 50, 100, 200, 400, 800, 1600]:
    DIR = f"output/shark_attack_top5"
    pathlib.Path(DIR).mkdir(parents=True, exist_ok=True)

    model = GVisionModel(decision_fn=shark_top5_decision)
    attack = RayS(model=model, order=np.inf, early_stopping=False)
    x_adv, queries, dist, succ = attack(x, 0, target=1, query_limit=query_limit)

    exp_name = f"cat_queries={query_limit}_dist={dist}"
    utils.save_img_tensor(x_adv, f"{DIR}/{exp_name}.png")

    # save final classification labels and scores
    labels, scores = gvision_classify(x_adv)
    with open(f"{DIR}/{exp_name}_results.txt", "w") as f:
        f.write("\n".join([l + ": " + str(s)  for l, s in zip(labels, scores)]))







