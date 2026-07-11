import numpy as np


class Model ():
    '''
    Тут мы храним общую логику модели и запускаем этот класс много раз
    '''
    def __init__(self, structure: tuple):
        weight = []
        bias = []

        for in_size, out_size in zip(structure[:-1], structure[1:]):
            limit = np.sqrt(6 / (in_size + out_size)) # добавляем нормализацию весов по слою что бы не выходили слишком быльшие значения
            weight.append(np.random.uniform(-limit, limit, (in_size, out_size)))
            bias.append(np.random.uniform(-0.1, 0.1, out_size))

        self.weight = weight
        self.bias = bias

    @staticmethod
    def _activation_(x):
        # return 1 / (1 + np.exp(-x)) #NOTE - Это убийца градиентов 
        return x

    def run (self, input_array):
        layers_results = []
        for i, layer in enumerate(self.weight):
            if i == 0: # начинаем с данных ввода (умножаем вектор на первую матрицу)
                result = input_array @ layer + self.bias[i]
                result = Model._activation_(result) # применяем функцию активации FIXME: тут слишком большие числа

                # сохраняем данные
                layers_results.append (result) 

            else: # работаем с результатом вывода предыдущего слоя, принимая его за вектор

                result = result @ layer + self.bias[i]

                result = Model._activation_(result) # применяем функцию активации   

                # сохраняем данные
                layers_results.append (result)   

        return layers_results