from PIL import Image
import numpy as np


def save_img_tensor(img, destination):
    img = img.cpu().detach().numpy()

    # convert from channels-first to channels-last
    img = img.transpose(1, 2, 0)

    img = (img * 255).astype(np.uint8)
    img = Image.fromarray(img)
    img.save(destination)

