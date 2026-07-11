import numpy as np
from model_factory import Model
from utilit import get_most_uses_neirons, get_images, compare_thing, element_wise_sum
import matplotlib.pyplot as plt

architecture = (2500, 2)
cats, dogs = get_images ()

def test (model, test_group):
    batch_results = []
    outputs = []
    for image in test_group:
        input_of_m = image.reshape((2500,))
        run_result = model.run (input_of_m)

        batch_results.append (run_result)
        outputs.append (run_result[len(run_result) - 1])

    return {
        'layer_result': batch_results,
        'outputs': outputs,
        'weight': model.weight,
        'bias': model.bias
    }

# create model
model = []
for i in range(10):
    model.append(Model (architecture))

groups = []

for group in [dogs,cats]:
    structure = []
    outputs = []
    impotant_way = []
    for m in model:
        data = test(m, group)

        structure.append([
            data.get('bias'), 
            data.get('weight')
            ])
        outputs.append (data.get('outputs'))

        layers_sum = element_wise_sum (data.get('layer_result'))
        impotant_way.append (get_most_uses_neirons(layers_sum))

    groups.append ((structure, outputs, impotant_way))

print (len(groups))

for i, group in enumerate(groups):
    structure, outputs, impotant_way = group
    y = []
    for el in impotant_way:
        y.append(sum(el) / len(el))
    x = np.arange(len(y))

    if i == 0:
        color = 'red'
    else:
        color = 'blue'
    
    plt.plot(x,y,color=color)

plt.show ()