import cv2
import matplotlib.pyplot as plt

depth_map = cv2.imread("output_depth.png", cv2.IMREAD_UNCHANGED)
plt.imshow(depth_map, cmap="magma")  # Change 'magma' to 'jet' or 'gray' if needed
plt.colorbar()
plt.title("Predicted Depth Map")
plt.show()
