import numpy as np
import matplotlib

# input_vector = [1.72, 1.23]
# weights_1 = [1.26, 0]
# weights_2 = [2.17, 0.32]

# #manually calc dot product of input_vector and weights_1
# #first_indexes_mult = input_vector[0] * weights_1[0]
# #second_indexes_mult = input_vector[1] * weights_1[1]
# #dot_product_1 = first_indexes_mult + second_indexes_mult

# dot_product_1 = np.dot(input_vector, weights_1)
# dot_product_2 = np.dot(input_vector, weights_2)

# print(f"Dot product 1 is: {dot_product_1}")
# print(f"Dot product 2 is: {dot_product_2}")

input_vector = np.array([2, 1.5])
weights_1 = np.array([1.45, -0.66])
bias = np.array([0.0])
target = 0

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def make_prediction(input_vector, weight, bias):
    layer_1 = np.dot(input_vector, weights_1) + bias
    layer_2 = sigmoid(layer_1)

    return layer_2

prediction = make_prediction(input_vector, weights_1, bias)

derivative = 2 * (prediction - target)

weights_1 = weights_1 - derivative
#adjust weights closer to target

prediction = make_prediction(input_vector, weights_1, bias)

error = (prediction - target) ** 2
print(f'The derivative is: {derivative}')

mse = np.square(prediction - target)

print(f"Prediction: {prediction} Error: {error}")