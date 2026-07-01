import numpy as np
from model_factory import Model
from utilit import get_most_uses_neirons, get_images, compare_thing
import matplotlib.pyplot as plt

architecture = (2500, 1250, 1000, 500)

batch = get_images ()

def test (model):
    batch_results = []
    outputs = []
    for image in batch:
        input_of_m = image.reshape((2500,))
        run_result = model.run (input_of_m)

        batch_results.append (run_result)
        outputs.append (run_result['layers_results'][len(run_result['layers_results']) - 1])
    return {
        'layer_result': batch_results,
        'outputs': outputs,
        'Weight': m.Weight,
        'bias': m.bias
    }

for i in range(100):
    data = []
    m = Model (architecture)
    data.append ( test(m) )
