import numpy as np
import activations as act


x = np.linspace(-10, 10, 100)

sigmoid = act.Sigmoid()
relu = act.ReLU()
softmax = act.Softmax()
leaky_relu = act.LeakyReLU(alpha=0.1)
tanh = act.Tanh()
prelu = act.PReLU(alpha=0.01)
elu = act.ELU(alpha=1)
swish = act.Swish(beta=1)

#y = relu.forward(x)
#y = softmax.forward(x)
#y = tanh.forward(x)
#y = leaky_relu.forward(x)
y = sigmoid.forward(x)
#y = prelu.forward(x)
#y = elu.forward(x)
#y = swish.forward(x)


act.plot_func(x, y, "Sigmoid(x)")