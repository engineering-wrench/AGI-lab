import numpy as np


class Model ():
    '''
    Тут мы храним общую логику модели и запускаем этот класс много раз
    '''
    def __init__(self, structure: tuple):
        Weight = []
        bias = []

        for in_size, out_size in zip(structure[:-1], structure[1:]):
            Weight.append(np.random.random((in_size, out_size)))
            bias.append (np.random.random (out_size,))
        
        self.Weight = Weight
        self.bias = bias

    @staticmethod
    def _activation_(x):
        return 1 / (1 + np.exp(-x))

    def run (self, input_array):
        layers_results = []
        for i, layer in enumerate(self.Weight):
            if i == 0: # начинаем с данных ввода (умножаем вектор на первую матрицу)
                result = input_array @ layer + self.bias[i]
                result = self._activation_(result) # применяем функцию активации

                # сохраняем данные
                layers_results.append (result) 

            else: # работаем с результатом вывода предыдущего слоя, принимая его за вектор
                result = result @ layer + self.bias[i]
                result = self._activation_(result) # применяем функцию активации   

                # сохраняем данные
                layers_results.append (result)   
        return {
            'layers_results': layers_results
        }