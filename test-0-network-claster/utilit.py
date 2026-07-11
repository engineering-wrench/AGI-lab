import numpy as np
from PIL import Image
import os

def get_most_uses_neirons(layers_results):
    most_uses_neirons = []
    for layer in layers_results:
        index_of_max_weight = np.argmax(layer)
        most_uses_neirons.append(index_of_max_weight)
    return most_uses_neirons

def compare_thing (A,B):
    diff = np.sum(np.abs(A - B))
    return diff

def get_images():
    cats = []
    dogs = []
    for filename in os.listdir('./image/'):
        img = Image.open(os.path.join('./image/', filename)).convert('L')
        img = img.resize((50, 50))
        img_array = np.array(img)
        img_array = img_array.astype(np.float32) / 255.0

        if filename.lower().startswith(('cat')):
            cats.append(img_array)
        else:
            dogs.append(img_array)
    return (cats, dogs)

def element_wise_sum (arrays):
    return [sum(values) for values in zip(*arrays)]