import numpy.random
import scipy.special


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate

        self.Wih = numpy.random.normal(0.0, pow(input_nodes, -0.5), (hidden_nodes, input_nodes))
        self.Who = numpy.random.normal(0.0, pow(hidden_nodes, -0.5), (output_nodes, hidden_nodes))

        self.activation_func = scipy.special.expit
        self.times = 0

    def train(self, input_list, target_list):
        input_list = numpy.array(input_list, ndmin=2).T
        target_list = numpy.array(target_list, ndmin=2).T

        hidden_output = self.activation_func(numpy.dot(self.Wih, input_list))
        final_output = self.activation_func(numpy.dot(self.Who, hidden_output))

        output_error = target_list - final_output
        hidden_error = numpy.dot(self.Who.T, output_error)

        self.Who += self.learning_rate * numpy.dot(
            (output_error * final_output * (1.0 - final_output)), hidden_output.T)

        self.Wih += self.learning_rate * numpy.dot(
            (hidden_error * hidden_output * (1.0 - hidden_output)), input_list.T)

        self.times += 1
        if self.times % 1000 == 0:
            print('完成第 %d 组训练' % (self.times / 1000))

    def query(self, input_list):
        input_list = numpy.array(input_list, ndmin=2).T
        hidden_output = self.activation_func(numpy.dot(self.Wih, input_list))
        return self.activation_func(numpy.dot(self.Who, hidden_output))

    def reset_epoch(self):
        self.times = 0


if __name__ == '__main__':
    score = []
    epochs = 3
    params = {'input_nodes': 784,
              'hidden_nodes': 200,
              'output_nodes': 10,
              'learning_rate': 0.2}
    network = NeuralNetwork(**params)

    with open('mnist_train.csv', 'r') as f:
        train_data = f.readlines()

    for e in range(epochs):
        for values in train_data:
            value = values.split(',')
            inputs = numpy.asfarray(value[1:]) / 255.0 * 0.99 + 0.01
            target = numpy.zeros(params['output_nodes']) + 0.01
            target[int(value[0])] = 0.99

            network.train(inputs, target)

        print('第%d世纪 训练完成' % (e + 1))
        network.reset_epoch()

    with open('mnist_test.csv', 'r') as f:
        test_data = f.readlines()

    for values in test_data:
        value = values.split(',')
        inputs = numpy.asfarray(value[1:]) / 255.0 * 0.99 + 0.01
        answer = numpy.argmax(network.query(inputs))

        if int(value[0]) == answer:
            score.append(1)
        else:
            score.append(0)

    score = numpy.asfarray(score)
    print('准确率：', score.sum() / score.size)
