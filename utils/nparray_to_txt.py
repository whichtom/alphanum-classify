import numpy as np
import sys

# loads the data
data = np.load(sys.argv[1])
print(data.shape)
np.savetxt(sys.argv[2], sys.argv[1], delimiter=" ")
