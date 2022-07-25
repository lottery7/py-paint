import numpy as np
from keras.datasets import mnist


class NeuralNetwork:
    LEARN_DATA_SIZE = 1000
    INPUT_SIZE = 784
    OUTPUT_SIZE = 10
    HIDDEN_SIZE = 100
    BATCH_SIZE = 100
    ALPHA = 0.03

    def __init__(self):
        np.random.seed(1)

        def scale(x): return (2 * x - 1)

        self.__weights = [
            0.1 * scale(np.random.rand(self.INPUT_SIZE, self.HIDDEN_SIZE)),
            0.1 * scale(np.random.rand(self.HIDDEN_SIZE, self.OUTPUT_SIZE))
        ]

        self.__test_inputs = None
        self.__test_outputs = None
        self.__learn_inputs = None
        self.__learn_outputs = None
        self.initialize_dataset()

    @staticmethod
    def relu(x):
        return (x > 0) * x

    @staticmethod
    def relu_deriv(x):
        return (x > 0)

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh_deriv(x):
        return 1 - (x**2)

    @staticmethod
    def sigmoid(x):
        y = np.exp(x)
        return y / (y + 1)

    @staticmethod
    def softmax(x):
        y = np.exp(x)
        return y / np.sum(y, axis=1, keepdims=True)

    def execute(self, data):
        layers = [None] * 3

        layers[0] = data
        layers[1] = self.tanh(np.dot(layers[0], self.__weights[0]))
        layers[2] = self.softmax(np.dot(layers[1], self.__weights[1]).reshape(1, self.OUTPUT_SIZE))

        return layers[2]

    def initialize_dataset(self):
        (learn_inputs, learn_outputs), (test_inputs, test_outputs) = mnist.load_data()
        self.__learn_inputs, learn_outputs = learn_inputs[:self.LEARN_DATA_SIZE].reshape(self.LEARN_DATA_SIZE, self.INPUT_SIZE) / 255, learn_outputs[:self.LEARN_DATA_SIZE]
        self.__test_inputs, test_outputs = test_inputs.reshape(len(test_inputs), self.INPUT_SIZE) / 255, test_outputs

        self.__learn_outputs = np.zeros(shape=(self.LEARN_DATA_SIZE, self.OUTPUT_SIZE), dtype=np.float32)
        for ind, out in enumerate(learn_outputs):
            self.__learn_outputs[ind][out] = 1.0


        self.__test_outputs = np.zeros(shape=(len(test_outputs), self.OUTPUT_SIZE), dtype=np.float32)
        for ind, out in enumerate(test_outputs):
            self.__test_outputs[ind][out] = 1.0

    def test(self):
        correct_counter = 0
        for i in range(len(self.__test_inputs)):
            correct_counter += int(
               np.argmax(self.execute(self.__test_inputs[i])) == np.argmax(self.__test_outputs[i])
            )
        
        print(f"Test-Acc: {correct_counter / len(self.__test_inputs):.3f}")

    def learn(self, epochs=300):
        for i in range(1, epochs + 1):
            correct_counter = 0
            for j in range(self.LEARN_DATA_SIZE // self.BATCH_SIZE):
                batch_start, batch_end = j * self.BATCH_SIZE, (j + 1) * self.BATCH_SIZE

                layers = [None] * 3
                layers[0] = self.__learn_inputs[batch_start:batch_end]
                layers[1] = self.tanh(np.dot(layers[0], self.__weights[0]))
                mask = np.random.randint(3, size=layers[1].shape)
                layers[1] *= mask
                layers[2] = self.softmax(np.dot(layers[1], self.__weights[1]))

                expected_outputs = self.__learn_outputs[batch_start:batch_end]

                for k in range(self.BATCH_SIZE):
                    correct_counter += (
                        np.argmax(layers[2][k]) == np.argmax(expected_outputs[k])
                    )
                
                layer_deltas = [None] * 3
                layer_deltas[2] = (expected_outputs - layers[2]) / self.BATCH_SIZE
                layer_deltas[1] = np.dot(layer_deltas[2], self.__weights[1].T) * self.tanh_deriv(layers[1])
                layer_deltas[1] *= mask

                self.__weights[1] += np.dot(layers[1].T, layer_deltas[2]) * self.ALPHA
                self.__weights[0] += np.dot(layers[0].T, layer_deltas[1]) * self.ALPHA
            
            if (i % 100 == 0 or i == epochs):
                print(
                    F"Iter: {i}",
                    f"Train-Acc: {correct_counter / self.LEARN_DATA_SIZE:.3f}",
                    end=' '
                )
                self.test()

    def load_weights_to_file(self, path):
        file = open(path, "w")
        # for i in range(2):
        #     w = self.__weights[i]
        #     for j in range(len(w)):
        #         file.write(','.join(map(str, w[j])))
        #         file.write('\n')
        #     file.write(";")
        file.write(
            ';'.join(
                '\n'.join(
                    ','.join(map(str, row))
                    for row in weights
                    )
                for weights in self.__weights
                )
        )
        file.close()
    
    def load_weights_from_file(self, path):
        file = open(path, "r")
        str_file = ''.join(file.readlines())
        file.close()

        file = str_file.split(';')
        self.__weights = [None] * len(file)
        for i in range(len(self.__weights)):
            w_str = file[i].split('\n')
            self.__weights[i] = np.zeros(shape=(len(w_str), w_str[0].count(',') + 1), dtype=np.float32)
            for j in range(len(w_str)):
                w_float = list(map(np.float64, w_str[j].split(',')))
                self.__weights[i][j] += np.array(w_float, dtype=np.float64)


if __name__ == "__main__":
    nn = NeuralNetwork()

    nn.learn()
    nn.load_weights_from_file("weights.txt")
    nn.test()