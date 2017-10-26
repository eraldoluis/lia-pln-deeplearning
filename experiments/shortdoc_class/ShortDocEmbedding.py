import theano

class InputSimulation:

    def __init__(self, inputs, outputs, wordWindow):
        self.__net = theano.function(inputs=inputs, outputs=outputs)
        self.__wordWindow = wordWindow

    def putToFlow(self, input):
        return self.__net(self.__wordWindow(input))