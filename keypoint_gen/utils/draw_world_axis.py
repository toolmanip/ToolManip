import numpy as np
import cv2
import os

import numpy as np
import cv2


def draw_axis_directions(toolOrObj, img, save_path):
    """
    Draw X/Y/Z world coordinate system direction arrows centered at the image center
    Parameters:
    - img_path: Image path
    - K: Camera intrinsic matrix (3x3)
    - T_world2camera: World coordinate system → Camera coordinate system homogeneous transformation matrix (4x4)
    - arrow_len_px: Arrow length (in pixels)

    Returns:
    - None (directly display or save image)
    """
    arrow_len_px = 60  # Arrow length (in pixels)

    # === Camera intrinsic parameters (Example) ===
    K_tool = np.array(
        [
            [608.4656982421875, 0.0, 322.7912902832031],
            [0.0, 608.57861328125, 250.60108947753906],
            [0.0, 0.0, 1.0],
        ]
    )
    K_obj = np.array(
        [
            [607.68212890625, 0.0, 320.1272277832031],
            [0.0, 606.2548828125, 242.69110107421875],
            [0.0, 0.0, 1.0],
        ]
    )
    # === Load homogeneous matrices ===
    script_dir = os.path.dirname(__file__)
    extri_obj_file_path = os.path.join(script_dir, "../../D435/extrinsics_obj.npy")
    extri_tool_file_path = os.path.join(script_dir, "../../D435/extrinsics_tool.npy")
    T_tool_extri = np.load(extri_tool_file_path)
    T_obj_extri = np.load(
        extri_obj_file_path
    )  # Camera extrinsic parameters (needs adjustment after camera is fixed)

    T_tool_plate2Robot = np.array(
        [[0, 0, 1, 450], [0, -1, 0, 0], [1, 0, 0, 319.5], [0, 0, 0, 1]]
    )
    T_obj_plate2Robot = np.array(
        [[1, 0, 0, -480], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    )

    if toolOrObj == "tool":
        T_plate2Robot = T_tool_plate2Robot
        T_extri = T_tool_extri
    elif toolOrObj == "object":
        T_plate2Robot = T_obj_plate2Robot
        T_extri = T_obj_extri

    T_world2camera = T_extri @ T_plate2Robot

    R = T_world2camera[:3, :3]  # Extract only the rotation part

    h, w = img.shape[:2]

    center_2d = np.array([w // 2, h // 2])

    # Three world unit vectors in different directions
    x_axis = np.array([1, 0, 0]).reshape(3, 1)
    y_axis = np.array([0, 1, 0]).reshape(3, 1)
    z_axis = np.array([0, 0, 1]).reshape(3, 1)

    # Transform to camera coordinate system directions
    x_dir = R @ x_axis
    y_dir = R @ y_axis
    z_dir = R @ z_axis

    # Project direction to image plane (only direction, controlled by pixel length)
    def get_arrow_end(center, dir_cam, scale=60):
        vec2d = dir_cam[:2].flatten()
        vec2d = vec2d / np.linalg.norm(vec2d)  # Unit vector
        return (center + vec2d * scale).astype(int)

    x_end = get_arrow_end(center_2d, x_dir, arrow_len_px)
    y_end = get_arrow_end(center_2d, y_dir, arrow_len_px)
    z_end = get_arrow_end(center_2d, z_dir, arrow_len_px)

    # Draw arrows
    cv2.arrowedLine(
        img, center_2d, tuple(x_end), (0, 0, 255), 2, tipLength=0.2
    )  # X - Red
    cv2.arrowedLine(
        img, center_2d, tuple(y_end), (0, 255, 0), 2, tipLength=0.2
    )  # Y - Green
    cv2.arrowedLine(
        img, center_2d, tuple(z_end), (255, 0, 0), 2, tipLength=0.2
    )  # Z - Blue
    # Add text labels
    cv2.putText(
        img, "X", tuple(x_end + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
    )
    cv2.putText(
        img, "Y", tuple(y_end + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
    )
    cv2.putText(
        img, "Z", tuple(z_end + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2
    )

    cv2.imwrite(save_path, img)
    return img
