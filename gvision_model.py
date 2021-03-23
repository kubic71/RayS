import io
import os
# Imports the Google Cloud client library
from google.cloud import vision
import numpy as np 
from PIL import Image
import utils
import random

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "keys/Trema-14000fdb4eac.json"


def label_list_decision(labels, scores, label_set):
    """Decision function matching classification labels againts given label set"""
    return not label_set_match(label_set, labels)


def label_set_match(object_labels, returned_labels):
    """Return True if at least one of the object labels is contained in at least one of the returned labels"""
    for o in object_labels:
        for r in returned_labels:
            if o.lower() in r.lower():
                return True
    return False


def gvision_classify(img):
    """Return the labels and scores by calling the cloud API"""
        

    fn = "/tmp/.temp_img_" + str(random.randint(0, 1000000)) + ".png"
    utils.save_img_tensor(img, fn)

    client = vision.ImageAnnotatorClient()
    # Loads the image into memory
    with io.open(fn, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    # Performs label detection on the image file
    response = client.label_detection(image=image, max_results=100)
    labels = response.label_annotations


    descriptions = [label.description for label in labels]
    scores = [label.score for label in labels]
    return (descriptions, scores)



class GVisionModel:
    """Google Vision API wrapper for RayS_Single.py"""


    # For safety, max number of requests is by default set to conservative number
    def __init__(self, decision_fn, max_requests=10000):
        """Initialize the wrapper.

        Args:
            decision_fn ((labels, scores) -> Bool): Given labels and their confidence scores return True if the example already classifies as advesarial
            max_requests (int): Safety limit for the maximum number of API calls to GVision API (the API is expensive)
        """
        
        self.n_request = 0
        self.max_requests = max_requests
        self.decision_fn = decision_fn


    # RayS_Single only requires model.predict_label(image) method to be implemented
    def predict_label(self, img, dummy=False):
        """GVision wrapping method. Queries cloud API and returns integer label.

        Args:
            img: channels-first torch tensor image with values in [0 - 1] range
        """
        if self.n_request >= self.max_requests:
            raise Exception("Google Vision max requests exceeded")

        self.n_request += 1

        # Debug
        if dummy:
            return 0

        print("Gvision request:", self.n_request)

        descriptions, scores = gvision_classify(img)
        is_adversarial = self.decision_fn(descriptions, scores)
        if is_adversarial:
            return 1
        else:
            return 0