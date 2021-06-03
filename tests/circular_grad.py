import numpy as np
import matplotlib.pyplot as plt
import cv2

mask = np.zeros((70, 70), dtype="uint8")
cv2.circle(mask, (35, 35), 35, 255, -1)
plt.imshow(mask, cmap='gray')
plt.show()
