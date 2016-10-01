import h5py
import numpy as np
import sys

h5File = h5py.File(sys.argv[1], "r")

W1 = np.asarray(h5File["encoder"]["W"])
b1 = np.asarray(h5File["encoder"]["b"])

b2 = np.asarray(h5File["decoder"]["b"])

weights = {}

weights["W_Encoder"] = W1
weights["b_Encoder"] = b1
weights["b_Decoder"] = b2

np.save(sys.argv[2], weights)

print "Model converted"
