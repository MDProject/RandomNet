from RandomNet import *
import numpy as np

# initialize random network
net_struct = [1,3,3]
net = RandomNet(net_struct,2)
DIM = RandomNetDim(net, ensemble_num = 3)
for l in range(len(DIM)):
    print('Layer {0} dimensions: '.format(l))
    for c in range(len(DIM[l])):
        print('Channel {0} dimensions: {1}'.format(c,DIM[l][c][0]))

"""
print(net.weight)
x_input = np.random.randn(3,3)
net.forward(x_input)
print(x_input)
print(net.weight[1])
print(net.bias[1])
print(net.activation[1])

for l in len(DIM):
    print('Layer {0} '.format())
"""