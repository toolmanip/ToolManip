import cv2

def get_depth(x, y, filename, fullimg=False):
    if fullimg:
        x = int(x /1920 * 640)
        y = int(y /1080 * 480)
    depth_img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    depth = depth_img[y, x] 
    print(f"Pixel: ({x}, {y}), Depth: {depth} mm")
    return depth

if __name__ == "__main__":
    get_depth(504, 179, "depth_test.png")