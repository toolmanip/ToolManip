from PIL import Image
from Lang_SAM.lang_sam import LangSAM
import numpy as np
import json

# Prerequisite: VLM1 outputs descriptions of tools and target objects

# Using LangSAM to mask the target area
# Open VLM1 results: description of the target
with open("/home/guyuwei/ToolManip/VLM/VLM1_Output.json", "r") as f:
    data = json.load(f)

tool_Name = data["tool"]["name"]
tool_Color = data["tool"]["color"]
tool_Position = data["tool"]["position"]

stepAction = data["task_steps"]["step1"]
targetObj = data["object"]["name"]
targetObj_Color = data["object"]["color"]
targetObj_Position = data["object"]["position"]

model = LangSAM()

# >>>>>>>>>>>>>>>INPUT<<<<<<<<<<<<<<<<<<<<<
tool_img = Image.open("../D435/captures/rgb_tool.png").convert("RGB")
obj_img = Image.open("../D435/captures/rgb_obj.png").convert("RGB")
text_prompt_Tool = (
    f"{tool_Name} with color of {tool_Color} and position of {tool_Position}"
)

text_prompt_Object = (
    f"{targetObj} with color of {targetObj_Color} and position of {targetObj_Position}"
)
# >>>>>>>>>>>>>>>INPUT<<<<<<<<<<<<<<<<<<<<<

results_Tool = model.predict([tool_img], [text_prompt_Tool])
results_Object = model.predict([obj_img], [text_prompt_Object])
print(results_Tool[0]["boxes"])
print("Tool ROI is saved")
print(results_Object[0]["boxes"])
print("Object ROI is saved")
np.save("x1y1x2y2_Tool.npy", results_Tool[0]["boxes"])

np.save("x1y1x2y2_Object.npy", results_Object[0]["boxes"])

# Post-process: Run S2 keyimgDots
