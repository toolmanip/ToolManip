import numpy as np
import math
from scipy.spatial.transform import Rotation as R
import os
import json


# Construct homogeneous transformation matrix T based on Euler angles (xyz order) and position vector
def make_transform_from_euler_xyz(rx_deg, ry_deg, rz_deg, tx, ty, tz):
    """
    Construct homogeneous transformation matrix T based on Euler angles xyz order and position vector

    Parameters:
    - rx_deg, ry_deg, rz_deg: Rotation angles around x, y, z axes (unit: degrees)
    - tx, ty, tz: Translation vector (unit: meters)

    Returns:
    - 4x4 homogeneous transformation matrix numpy.ndarray
    """
    # Convert degrees to radians
    rx, ry, rz = np.radians([rx_deg, ry_deg, rz_deg])

    # Construct rotation matrices around each axis (Euler angle order: xyz)
    Rz = np.array(
        [[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]]
    )
    Ry = np.array(
        [[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]]
    )
    Rx = np.array(
        [[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]]
    )

    # Compose rotation matrix (xyz order)
    R_mat = Rz @ Ry @ Rx

    # Construct homogeneous transformation matrix
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = [tx, ty, tz]
    return np.round(T, 3)


# calculate tool function point B relative to gripper center point A,return tx, ty,tz，
# 4.18 EEF_current= Robot EEF pose when grasping tool point (provided by SAM6D tool grasping pose detection)
def calculateToolTranslation(pointB_world, T_TG, EEF_current):
    # Convert EEF grasping pose to homogeneous matrix
    T_BT = make_transform_from_euler_xyz(
        rx_deg=EEF_current[3],
        ry_deg=EEF_current[4],
        rz_deg=EEF_current[5],
        tx=EEF_current[0] / 1000,
        ty=EEF_current[1] / 1000,
        tz=EEF_current[2] / 1000,
    )
    print("T_BT:", T_BT)
    # Step 1: T_WB: gripper transformation in world coordinate system
    T_WB = T_BT @ T_TG

    # Step 2: T_BG: inverse transformation from world to gripper
    T_BG = np.linalg.inv(T_WB)

    # Step 3: Transform B_world to gripper coordinate system
    B_world_h = np.append(pointB_world / 1000, 1)  # Homogeneous form
    print("B_world_h:", B_world_h)
    B_in_A_h = (
        T_BG @ B_world_h
    )  # Coordinates of functional point B in gripper A coordinate system (homogeneous)

    tx, ty, tz = B_in_A_h[:3]

    print(
        "✅ Position offset of functional point B in tool grasping point A coordinate system (i.e., tx, ty, tz in T_GE):"
    )
    print(f"tx = {tx:.4f}, ty = {ty:.4f}, tz = {tz:.4f}")
    return tx, ty, tz


# 4.18 Calculate tool functional point coordinate system direction relative to gripper; EEF_current= Robot EEF pose when grasping tool point (provided by SAM6D tool grasping pose detection), return tool's rx, ry, rz
def calculateToolVectorDirection(
    pointA, pointB, pointC, T_TG, EEF_current, x_dirChoose
):
    # Step 1: Z-axis = BC (tool forward direction)
    z_dir = pointB - pointC
    z_dir_world = z_dir / np.linalg.norm(z_dir)

    # Step 2: X-axis lies in the plane defined by A, B, C — use AC vector
    if x_dirChoose == 0:
        x_ref = pointA - pointB
    elif x_dirChoose == 1:
        x_ref = pointC - pointA

    x_dir_world = np.cross(np.cross(z_dir_world, x_ref), z_dir_world)
    x_dir_world /= np.linalg.norm(x_dir_world)

    # Step 3: Y-axis = Z × X
    y_dir_world = np.cross(z_dir_world, x_dir_world)
    y_dir_world /= np.linalg.norm(y_dir_world)

    # Step 4: Build tool-to-world rotation matrix
    R_tool_in_world = np.column_stack((x_dir_world, y_dir_world, z_dir_world))

    # Step 5: Get world-to-gripper rotation
    T_BT = make_transform_from_euler_xyz(
        rx_deg=EEF_current[3],
        ry_deg=EEF_current[4],
        rz_deg=EEF_current[5],
        tx=EEF_current[0],
        ty=EEF_current[1],
        tz=EEF_current[2],
    )
    T_WG = T_BT @ T_TG
    R_gripper_in_world = T_WG[:3, :3]

    # Step 6: Transform tool rotation to gripper frame
    R_tool_in_gripper = R_gripper_in_world.T @ R_tool_in_world
    x_dir_gripper = R_tool_in_gripper[:, 0]
    y_dir_gripper = R_tool_in_gripper[:, 1]
    z_dir_gripper = R_tool_in_gripper[:, 2]

    # Step 7: Output
    print("✅ Tool coordinate system basis vectors in Gripper coordinate system:")
    print("x_dir =", x_dir_gripper)
    print("y_dir =", y_dir_gripper)
    print("z_dir =", z_dir_gripper)

    return x_dir_gripper, y_dir_gripper, z_dir_gripper


# A = np.array([0, 0, 0]), B = np.array([1, 1, 1]), z_dir=vec B to A
def generate_coord_xyz_in_plane(pointS1, pointS2, pointS3):
    """
    Given 3 points S1, S2, S3 in space:
    - Z-axis is defined as S2 - S1
    - X-axis lies in the plane (S1, S2, S3), toward S3 as much as possible
    - Y-axis is orthogonal, forming a right-handed coordinate system
    """
    # Step 1: Define z-axis
    z_vec = pointS2 - pointS1
    z_dir = z_vec / np.linalg.norm(z_vec)

    # Step 2: Use S3 - S1 as reference vector lying in the plane
    # x_ref = pointS3 - pointS1
    x_ref = pointS1 - pointS3
    print("x_ref:", x_ref)
    # Step 3: Project x_ref onto the plane perpendicular to z_dir
    x_proj = np.cross(np.cross(z_dir, x_ref), z_dir)
    x_dir = x_proj / np.linalg.norm(x_proj)

    # Step 4: y_dir = z × x to complete right-handed system
    y_dir = np.cross(z_dir, x_dir)
    y_dir = y_dir / np.linalg.norm(y_dir)

    return x_dir, y_dir, z_dir


# Tool coordinate system unit vectors in world coordinate system
# Three direction vectors (combined by columns into rotation matrix)


# Calculate target coordinate system direction in world coordinate system: rx,ry,rz
def get_obj_direction(pointS1, pointS2, pointS3_ref, multipleCase):
    Obj_x_axis, Obj_y_axis, Obj_z_axis = generate_coord_xyz_in_plane(
        pointS1, pointS2, pointS3_ref
    )

    # Obj_x_axis = [0, -1, 0]
    # Obj_y_axis = [0, 0, 1]
    # Obj_z_axis = [-1, 0, 0]

    # change to set ry in the same plane with rz rather than rx
    if multipleCase == 1:
        Obj_x_axis_ori = Obj_x_axis
        Obj_y_axis_ori = Obj_y_axis
        Obj_x_axis = Obj_y_axis_ori
        Obj_y_axis = -Obj_x_axis_ori
    # do not change the direction of Obj_x_axis and Obj_y_axis
    else:
        Obj_x_axis = Obj_x_axis
        Obj_y_axis = Obj_y_axis
    print("obj x_dir:", Obj_x_axis)
    print("obj y_dir:", Obj_y_axis)
    print("obj z_dir:", Obj_z_axis)

    # Combine by columns into rotation matrix R
    R_mat = np.column_stack((Obj_x_axis, Obj_y_axis, Obj_z_axis))
    rotation = R.from_matrix(R_mat)
    euler_deg = rotation.as_euler("xyz", degrees=True)
    print("Euler angles (rx, ry, rz), unit °:", np.round(euler_deg, 3))
    return euler_deg


"""
***Note***
Tool Z-axis direction is CB
Object's S1 is the alignment point for tool functional point B (position to reach)
Object target Z-axis direction is S1 to S2
First need SAM6D to get tool grasping pose EEF_current, then calculate tool Z-axis direction
"""
# EEF pose provided by SAM6D
script_dir = os.path.dirname(__file__)
EEFpose = os.path.join(script_dir, "../lang-segment-anything/EEF_Pose.npy")
pose_array = np.load(EEFpose)
x, y, z, rx, ry, rz = pose_array
EEF_current = [x, y, z, rx, ry, rz]


KeypointWorld_coord_path = os.path.join(
    script_dir, "../lang-segment-anything/keyP_inWorldRobot.npy"
)
KeypointWorld_coord = np.load(KeypointWorld_coord_path)
pointA = KeypointWorld_coord[0]
pointB = KeypointWorld_coord[1]
pointC = KeypointWorld_coord[2]
pointS1 = KeypointWorld_coord[3]
pointS2 = KeypointWorld_coord[4]

VLM2OutputPath = os.path.join(script_dir, "../VLM/VLM2_Output.json")
with open(VLM2OutputPath, "r") as f:
    VLM2Data = json.load(f)
S2_Z_height = VLM2Data["Keypoint S2 Direction"]

Obj_euler_dir_list = []  # two cases of the possible object pose,
# multilpeCase= {0,1}; change/not change rx and ry direction respectively to contain different align poses
for multilpeCase in range(2):
    # object z dir is vertical to world xy plane (S1=S2)
    if pointS1[0] == pointS2[0]:
        print("✅S1-S2 are same, z is vertical, set object axis manually")
        if multilpeCase == 0:
            rx_dir_obj = [0, -1, 0]
            ry_dir_obj = [-1, 0, 0]
            rz_dir_obj = [0, 0, -1]
        elif multilpeCase == 1:
            rx_dir_obj = [-1, 0, 0]
            ry_dir_obj = [0, 1, 0]
            rz_dir_obj = [0, 0, -1]
        R_mat = np.column_stack((rx_dir_obj, ry_dir_obj, rz_dir_obj))
        rotation = R.from_matrix(R_mat)
        euler_deg = rotation.as_euler("xyz", degrees=True)
        rx_obj = euler_deg[0]
        ry_obj = euler_deg[1]
        rz_obj = euler_deg[2]

    elif S2_Z_height == "same" or S2_Z_height == "down" or S2_Z_height == "up":
        if S2_Z_height == "same":
            print("✅z_S2= z_S1")
            pointS2[2] = pointS1[
                2
            ]  # Set S2's z-axis coordinate consistent with S1 to make the plane of S1,S2,S3 perpendicular to the z_world axis
            pointS3_ref = [0, 0, pointS1[2]]
        elif S2_Z_height == "down":
            print("✅z_S2< z_S1")
            pointS2[2] = pointS1[2] - 30
            pointS3_ref = [pointS1[0], pointS1[1], pointS1[2] - 30]
        elif S2_Z_height == "up":
            print("✅z_S2 > z_S1")
            pointS2[2] = pointS1[2] + 30
            pointS3_ref = [pointS1[0], pointS1[1], pointS1[2] + 30]
        print("pointS1:", pointS1)
        print("pointS2:", pointS2)
        print("pointS3_ref:", pointS3_ref)
        rx_obj, ry_obj, rz_obj = get_obj_direction(
            pointS1, pointS2, pointS3_ref, multilpeCase
        )
    Obj_euler_dir = [rx_obj, ry_obj, rz_obj]
    Obj_euler_dir_list.append(Obj_euler_dir)


results = []


for i in range(len(Obj_euler_dir_list)):

    obj_pose = Obj_euler_dir_list[i]
    print(f"Case{i+1}]:obj direction pose=", np.round(obj_pose, 3))
    # (Fixed) Gripper pose relative to EEF: rotate counter-clockwise 60 degrees around z-axis, z-offset 200mm  (degree, m)
    T_TG = make_transform_from_euler_xyz(
        rx_deg=0, ry_deg=0, rz_deg=60, tx=0, ty=0, tz=0.2
    )
    # ((Finished 4.18) SAM6D-given, Final alignment Pose (relative to world coord) (only consider how target pose rotates and translates around world axes)
    T_BE = make_transform_from_euler_xyz(
        rx_deg=obj_pose[0],
        ry_deg=obj_pose[1],
        rz_deg=obj_pose[2],
        tx=pointS1[0] / 1000,
        ty=pointS1[1] / 1000,
        tz=pointS1[2] / 1000,
    )  # Vertical downward grasping start point
    # print("\n✅object Pose: T_BE:\n",T_BE)

    tool_tx, tool_ty, tool_tz = calculateToolTranslation(pointB, T_TG, EEF_current)

    # different cases to difine the x axis of tool coord system(right or left)
    for tool_x_axis_choose in range(2):

        tool_x_axis, tool_y_axis, tool_z_axis = calculateToolVectorDirection(
            pointA, pointB, pointC, T_TG, EEF_current, tool_x_axis_choose
        )
        R_mat = np.column_stack((tool_x_axis, tool_y_axis, tool_z_axis))
        # Convert to Euler angles (unit: degrees)
        rotation = R.from_matrix(R_mat)
        tool_euler_deg = rotation.as_euler("xyz", degrees=True)
        # SAM6D-given, Tool holding system pose relative to Gripper system: rotate clockwise 90 degrees around z-axis, x-offset -13cm, y-3cm, z=0
        T_GE = make_transform_from_euler_xyz(
            rx_deg=tool_euler_deg[0],
            ry_deg=tool_euler_deg[1],
            rz_deg=tool_euler_deg[2],
            tx=tool_tx,
            ty=tool_ty,
            tz=tool_tz,
        )

        # Coordinate relationship chain: T_BT @ T_TG @ T_GE = T_BE ---> T_BT = T_BE @ inv(T_TG @ T_GE)
        T_TG_GE = T_TG @ T_GE  # tool coord relative to EEF coord
        T_TG_GE_inv = np.linalg.inv(T_TG_GE)  # EEF coord relative to tool coord
        # --------- Calculate robot end-effector pose T_BT ----------
        T_BT = (
            T_BE @ T_TG_GE_inv
        )  # T_BT: robot end-effector target pose, T_BT object relative to world @ EEF coord relative to tool coord
        # print("\n✅ :")
        # print(np.round(T_BT, 4))

        # Extract rotation matrix R and position vector t
        R_mat = T_BT[:3, :3]
        t_vec = T_BT[:3, 3]
        # Extract Euler angles (XYZ order), unit in degrees
        euler_deg = R.from_matrix(R_mat).as_euler("xyz", degrees=True)

        # Output
        print(f"✅ ✅ ✅ ✅ ✅ Case{i+1}-{tool_x_axis_choose+1}:")
        print("✅ Robot end-effector position (unit: mm):")
        print(
            f"x = {t_vec[0]*1000:.2f}, y = {t_vec[1]*1000:.2f}, z = {t_vec[2]*1000:.2f}"
        )
        print("✅ Robot end-effector pose (Euler angles xyz order, unit: degrees):")
        print(
            f"rx = {euler_deg[0]:.2f}°, ry = {euler_deg[1]:.2f}°, rz = {euler_deg[2]:.2f}°\n\n"
        )
        results.append(
            {
                "case_label": f"Case{i+1}-{tool_x_axis_choose+1}",
                "position": t_vec.copy(),
                "euler": euler_deg.copy(),
            }
        )

# Find the result with maximum z value
best_result = max(results, key=lambda r: r["position"][2])
x = best_result["position"][0] * 1000
y = best_result["position"][1] * 1000
z = best_result["position"][2] * 1000
rx = best_result["euler"][0]
ry = best_result["euler"][1]
rz = best_result["euler"][2]
print("🎯 Robot Final Pose:")
print(f"▶ {best_result['case_label']}")
print("▶ Robot end-effector position (unit: mm):")
print("x = {:.2f}, y = {:.2f}, z = {:.2f}".format(x, y, z))
print("▶ Robot end-effector pose (Euler angles xyz order, unit: degrees):")
print("rx = {:.2f}°, ry = {:.2f}°, rz = {:.2f}°".format(rx, ry, rz))
data = [x, y, z, rx, ry, rz]  # can be list or np.array
data = np.array(data)  # ensure it's a NumPy array

robot_final_pose_path = os.path.join(
    script_dir, "../lang-segment-anything/robot_final_pose.npy"
)
np.save(robot_final_pose_path, data)
print("✅ Saved to robot_final_pose.npy")
