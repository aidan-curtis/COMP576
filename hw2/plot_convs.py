import matplotlib.pyplot as plt
import numpy as np
conv = np.load("conv1.weights.npz")
for i in range(32):
	plt.subplot(4, 8, i+1)
	plt.imshow(conv[:, :, 0, i],cmap='gray')
plt.show()
