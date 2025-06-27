import cv2
import numpy as np
from skimage.morphology import skeletonize
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

mask_path = "mask_obj.png"
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

mask_bool = mask > 127
skeleton = skeletonize(mask_bool).astype(np.uint8) * 255

ys, xs = np.where(skeleton == 255)
points = np.array(list(zip(xs, ys)))

n_points = 5  
if len(points) >= n_points:
    kmeans = KMeans(n_clusters=n_points, random_state=0).fit(points)
    centers = kmeans.cluster_centers_
else:
    centers = points  

vis_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
def draw_dotted_line(img, pt1, pt2, color=(255, 0, 0), radius=2, gap=10):
    pt1 = np.array(pt1)
    pt2 = np.array(pt2)
    dist = np.linalg.norm(pt2 - pt1)
    num_dots = int(dist // gap)

    for i in range(num_dots + 1):
        point = pt1 + (pt2 - pt1) * (i / num_dots)
        point = tuple(np.round(point).astype(int))
        cv2.circle(img, point, radius, color, -1)

start_point1 = (154, 156)
end_point1 = (533, 416)
start_point2 = (226, 412)
end_point2 = (432, 170)

vis_img_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
draw_dotted_line(vis_img_mask, start_point1, end_point1)
draw_dotted_line(vis_img_mask, start_point2, end_point2)

cv2.imwrite("dotted_line_output.png", vis_img_mask)

for x, y in centers:
    cv2.circle(vis_img, (int(x), int(y)), 3, (0, 255, 0), -1)

plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title("Original Mask")
plt.imshow(vis_img_mask, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Skeleton")
plt.imshow(skeleton, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Centerline Points")
plt.imshow(vis_img)
plt.axis('off')

plt.tight_layout()
plt.show()
