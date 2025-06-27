import cv2
import numpy as np
import os
from utils.updateURL import RequireURL
import json


if __name__ == "__main__":
    # #Get the directory of the current Python script
    script_dir = os.path.dirname(__file__)
    # #Construct the full file path
    # SampleimagePath = os.path.join(script_dir, 'keypointSampleImg.jpg')
    # SampleURL = RequireURL(SampleimagePath)
    # print(f"Sample image URL: {SampleURL}")
    ToolimagePath = os.path.join(script_dir, "../D435/captures/rgb_tool.png")
    ObjimagePath = os.path.join(script_dir, "../D435/captures/rgb_obj.png")
    ToolURL = RequireURL(ToolimagePath)
    ObjURL = RequireURL(ObjimagePath)
    # print(f"dotted img url: {ToolURL}")
    # Assemble into structured JSON data
    task_data = {"Original Tool Img URL": ToolURL, "Original Object Img URL": ObjURL}

    # Save URL as JSON file
    os.makedirs("VLM", exist_ok=True)
    originalImgURLPath = os.path.join(script_dir, "../VLM/Original_Img_URL.json")
    with open(originalImgURLPath, "w") as f:
        json.dump(task_data, f, indent=4)
