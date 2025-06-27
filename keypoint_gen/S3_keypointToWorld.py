import cv2
import numpy as np
import os
from utils.updateURL import RequireURL
import json
from utils.getDepth import get_depth


# Get pixel coordinates corresponding to keypoint corner coordinates (a,b)
def get_pixel_coord_from_grid(x1, y1, x2, y2, m, n, row_idx, col_idx):
    """
    Given segmentation area and grid size, return pixel coordinates of the (row_idx, col_idx)th point
    row_idx: Starting from 1, indicates which row (vertical)
    col_idx: Starting from 1, indicates which column (horizontal)
    """
    step_x = (x2 - x1) // (n - 1)
    step_y = (y2 - y1) // (m - 1)
    x = x1 + (col_idx - 1) * step_x
    y = y1 + (row_idx - 1) * step_y
    return x, y


def Calculate_Keypoint_Pixel_Coord(ToolOrObj, KeypointName):
    """
    Return corresponding pixel coordinates based on keypoint name
    ToolOrObj: "Tool" or "Object"
    KeypointName: "A", "B", "C", "S1", "S2"
    """
    script_dir = os.path.dirname(__file__)
    # Keypoint information
    VLM_Keypoint_Path = os.path.join(script_dir, "../VLM/VLM2_Output.json")
    ToolOrObj_URL_Path = os.path.join(script_dir, f"../VLM/Dotted_{ToolOrObj}_URL.json")
    with open(ToolOrObj_URL_Path, "r") as f:
        dataROI = json.load(f)
        ToolOrObj_x1 = dataROI[f"Dotted {ToolOrObj} Img x1"]
        ToolOrObj_y1 = dataROI[f"Dotted {ToolOrObj} Img y1"]
        ToolOrObj_x2 = dataROI[f"Dotted {ToolOrObj} Img x2"]
        ToolOrObj_y2 = dataROI[f"Dotted {ToolOrObj} Img y2"]
        ToolOrObj_Row = dataROI[f"Dotted {ToolOrObj} Img Row"]
        ToolOrObj_Column = dataROI[f"Dotted {ToolOrObj} Img Column"]
    with open(VLM_Keypoint_Path, "r") as f:
        dataKeypoint = json.load(f)
    KeypointCoord = dataKeypoint[f"Keypoint {KeypointName}"]
    Row_Keypoint, Column_Keypoint = KeypointCoord.strip("()").split(",")
    Row_KPT = int(Row_Keypoint)
    Column_KPT = int(Column_Keypoint)

    Pixel_x, Pixel_y = get_pixel_coord_from_grid(
        ToolOrObj_x1,
        ToolOrObj_y1,
        ToolOrObj_x2,
        ToolOrObj_y2,
        ToolOrObj_Row,
        ToolOrObj_Column,
        Row_KPT,
        Column_KPT,
    )
    return Pixel_x, Pixel_y


def camera_to_pixel(X, Y, Z, K):
    # 1. Camera coordinates to pixel coordinates
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = (X / Z) * fx + cx
    v = (Y / Z) * fy + cy
    return np.array([int(u), int(v)])


def pixel_to_camera(u, v, depth, K):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    Z = depth
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return np.array([X, Y, Z])


def camera_to_worldPlate(cam_point, T):
    """
    T: 4x4 extrinsic matrix
    """
    trans_matrix = np.eye(4)
    trans_matrix[:3, 3] = cam_point
    T_wrt_world = np.linalg.inv(T) @ trans_matrix
    worldPlate_point = T_wrt_world[:3, 3]
    return worldPlate_point


def worldPlate_to_worldRobot(worldPlate_point, T):
    """
    T: 4x4 extrinsic matrix
    """
    trans_matrix = np.eye(4)
    trans_matrix[:3, 3] = worldPlate_point
    T_wrt_world = T @ trans_matrix
    world_point = T_wrt_world[:3, 3]
    return world_point


def pixel_to_worldPlate(u, v, depth, K, T):
    cam_point = pixel_to_camera(u, v, depth, K)
    world_point = camera_to_worldPlate(cam_point, T)
    return world_point


def keypointToPixelCoord():
    # Calculate keypoint pixel coordinates: ToolOrObj={Tool, Object} ; IF ToolOrObj="Tool", KeypointName={A, B, C}  IF ToolOrObj="Object", KeypointName={S1, S2}
    Pixel_xA, Pixel_yA = Calculate_Keypoint_Pixel_Coord(
        ToolOrObj="tool", KeypointName="A"
    )
    Pixel_xB, Pixel_yB = Calculate_Keypoint_Pixel_Coord(
        ToolOrObj="tool", KeypointName="B"
    )
    Pixel_xC, Pixel_yC = Calculate_Keypoint_Pixel_Coord(
        ToolOrObj="tool", KeypointName="C"
    )
    Pixel_xS1, Pixel_yS1 = Calculate_Keypoint_Pixel_Coord(
        ToolOrObj="object", KeypointName="S1"
    )
    Pixel_xS2, Pixel_yS2 = Calculate_Keypoint_Pixel_Coord(
        ToolOrObj="object", KeypointName="S2"
    )

    Pixel_List = {
        "A": (Pixel_xA, Pixel_yA),
        "B": (Pixel_xB, Pixel_yB),
        "C": (Pixel_xC, Pixel_yC),
        "S1": (Pixel_xS1, Pixel_yS1),
        "S2": (Pixel_xS2, Pixel_yS2),
    }
    script_dir = os.path.dirname(__file__)
    # Keypoint information
    VLM_Keypoint_Path = os.path.join(
        script_dir, "../lang-segment-anything/Keypoints_PixelCoord_List.npy"
    )
    np.save(VLM_Keypoint_Path, Pixel_List)
    print(f"All the keypoint pixel corrdinates: {Pixel_List}")


if __name__ == "__main__":
    # calculate keypoints pixel coordinates
    keypointToPixelCoord()

    # calculate keypoints camera&board world & robot world coordinates
    # Camera intrinsic parameters (fixed throughout)
    K_obj = np.array(
        [
            [607.68212890625, 0.0, 320.1272277832031],
            [0.0, 606.2548828125, 242.69110107421875],
            [0.0, 0.0, 1.0],
        ]
    )
    K_tool = np.array(
        [
            [608.4656982421875, 0.0, 322.7912902832031],
            [0.0, 608.57861328125, 250.60108947753906],
            [0.0, 0.0, 1.0],
        ]
    )

    # Get the directory of the current Python script
    script_dir = os.path.dirname(__file__)
    extri_obj_file_path = os.path.join(script_dir, "../D435/extrinsics_obj.npy")
    extri_tool_file_path = os.path.join(script_dir, "../D435/extrinsics_tool.npy")
    T_obj_extri = np.load(
        extri_obj_file_path
    )  # Camera extrinsic parameters (needs adjustment after camera is fixed)
    T_tool_extri = np.load(
        extri_tool_file_path
    )  # Camera extrinsic parameters (needs adjustment after camera is fixed)

    T_obj_plate2Robot = np.array(
        [[1, 0, 0, -480], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    )

    T_tool_plate2Robot = np.array(
        [[0, 0, 1, 450], [0, -1, 0, 0], [1, 0, 0, 319.5], [0, 0, 0, 1]]
    )

    depth_tool = os.path.join(script_dir, "../D435/captures/depth_tool.png")
    depth_object = os.path.join(script_dir, "../D435/captures/depth_obj.png")

    # Load the saved points
    loaded_points_path = os.path.join(
        script_dir, "../lang-segment-anything/Keypoints_PixelCoord_List.npy"
    )
    loaded_points = np.load(loaded_points_path, allow_pickle=True).item()
    keypoints = ["A", "B", "C", "S1", "S2"]
    keyP_inCamera_List = []
    keyP_inWorldPlate_List = []
    keyP_inWorldRobot_List = []

    VLM2OutputPath = os.path.join(script_dir, "../VLM/VLM2_Output.json")
    with open(VLM2OutputPath, "r") as f:
        VLM2Data = json.load(f)
    keypointA = VLM2Data["Keypoint A"]
    keypointB = VLM2Data["Keypoint B"]
    keypointC = VLM2Data["Keypoint C"]
    keypointS1 = VLM2Data["Keypoint S1"]
    keypointS2 = VLM2Data["Keypoint S2"]
    keypointsRowColumn = [keypointA, keypointB, keypointC, keypointS1, keypointS2]
    S1S2_Same = False
    if keypointS1 == keypointS2:
        print("S1=S2")
        S1S2_Same = True

    for i in range(len(keypoints)):
        u = loaded_points[f"{keypoints[i]}"][0]
        v = loaded_points[f"{keypoints[i]}"][1]
        if i < 3:

            d = get_depth(int(u), int(v), depth_tool)
            [Camera_x, Camera_y, Camera_z] = pixel_to_camera(u, v, d, K_tool)
            keyP_inCamera = np.array([Camera_x, Camera_y, Camera_z])
            print(f"tool keypoint {i+1},row/column = {keypointsRowColumn[i]}")
            print(
                f"tool keypoint {i+1} camera coordinates: X={Camera_x:.2f}, Y={Camera_y:.2f}, Z={Camera_z:.2f}"
            )
            keyP_inWorldPlate = pixel_to_worldPlate(u, v, d, K_tool, T_tool_extri)
            print(
                f"tool keypoint {i+1} world coordinates (Plate): X={keyP_inWorldPlate[0]:.2f}, Y={keyP_inWorldPlate[1]:.2f}, Z={keyP_inWorldPlate[2]:.2f}"
            )
            keyP_inWorldRobot = worldPlate_to_worldRobot(
                keyP_inWorldPlate, T_tool_plate2Robot
            )
            print(
                f"tool keypoint {i+1} world coordinates (Robot): X={keyP_inWorldRobot[0]:.2f}, Y={keyP_inWorldRobot[1]:.2f}, Z={keyP_inWorldRobot[2]:.2f}"
            )

        else:
            d = get_depth(int(u), int(v), depth_object)
            [Camera_x, Camera_y, Camera_z] = pixel_to_camera(u, v, d, K_obj)
            keyP_inCamera = np.array([Camera_x, Camera_y, Camera_z])
            print(f"object keypoint {i+1},row/column = {keypointsRowColumn[i]}")
            print(
                f"object keypoint {i+1} camera coordinates: X={Camera_x:.2f}, Y={Camera_y:.2f}, Z={Camera_z:.2f}"
            )
            keyP_inWorldPlate = pixel_to_worldPlate(u, v, d, K_obj, T_obj_extri)
            print(
                f"object keypoint {i+1} world coordinates (Plate): X={keyP_inWorldPlate[0]:.2f}, Y={keyP_inWorldPlate[1]:.2f}, Z={keyP_inWorldPlate[2]:.2f}"
            )
            keyP_inWorldRobot = worldPlate_to_worldRobot(
                keyP_inWorldPlate, T_obj_plate2Robot
            )
            # if S1 is same as S2, adjust S2 Z-=10
            if i == 4:
                if S1S2_Same == True:
                    keyP_inWorldRobot[2] = keyP_inWorldRobot[2] - 10
            print(
                f"object keypoint {i+1} world coordinates (Robot): X={keyP_inWorldRobot[0]:.2f}, Y={keyP_inWorldRobot[1]:.2f}, Z={keyP_inWorldRobot[2]:.2f}"
            )
            print(keyP_inWorldRobot)

        # Calculate camera coordinate

        keyP_inCamera_List.append(keyP_inCamera)
        keyP_inWorldPlate_List.append(keyP_inWorldPlate)
        keyP_inWorldRobot_List.append(keyP_inWorldRobot)

    keyP_inCamera_Lists = np.stack(keyP_inCamera_List)
    keyP_inWorld_Lists = np.stack(keyP_inWorldPlate_List)
    keyP_inWorldRobot_Lists = np.stack(keyP_inWorldRobot_List)

    keyP_inCamera = os.path.join(
        script_dir, "../lang-segment-anything/keyP_inCamera.npy"
    )
    keyP_inWorld = os.path.join(script_dir, "../lang-segment-anything/keyP_inWorld.npy")
    keyP_inWorldRobot = os.path.join(
        script_dir, "../lang-segment-anything/keyP_inWorldRobot.npy"
    )

    np.save(keyP_inCamera, keyP_inCamera_Lists)
    np.save(keyP_inWorld, keyP_inWorld_Lists)
    np.save(keyP_inWorldRobot, keyP_inWorldRobot_Lists)
