from openai import OpenAI
import re
import json
import os

client = OpenAI(
    base_url='', # Your OpenAI API base URL here, e.g., 'https://api.openai.com/v1'
    api_key='' # Your API Key here
)

script_dir = os.path.dirname(__file__)
dottedObjURLPath= os.path.join(script_dir, './img_url/Dotted_object_URL.json')

with open(dottedObjURLPath, "r") as f:
    dataURL = json.load(f)
dottedImg_Object = dataURL["Dotted object Img URL"]

dottedToolURLPath= os.path.join(script_dir, './img_url/Dotted_tool_URL.json')
with open(dottedToolURLPath, "r") as f:
    dataURL = json.load(f)
dottedImg_Tool = dataURL["Dotted tool Img URL"]

VLM1OutputPath= os.path.join(script_dir, './output/VLM1_Output.json')
with open(VLM1OutputPath, "r") as f:
    VLM1Data = json.load(f)

VLM2OutputPath= os.path.join(script_dir, './output/VLM2_Output.json')
with open(VLM2OutputPath, "r") as f:
    VLM2Data = json.load(f)
keypointA = VLM2Data["Keypoint A"]
keypointB = VLM2Data["Keypoint B"]
keypointS1 = VLM2Data["Keypoint S1"]
keypointS2 = VLM2Data["Keypoint S2"]


Task = VLM1Data["task"]
Tool = VLM1Data["tool"]["name"]
targetObj = VLM1Data["object"]["name"]
stepAction_1 = VLM1Data["task_steps"]["step1"]
stepAction_2 = VLM1Data["task_steps"]["step2"]
stepAction_3 = VLM1Data["task_steps"]["step3"]


# Visual Prompt Understanding

input_prompt = (
    f"You need to instruct a robot with 2-finger gripper to finish {Task}, here are descriptions:\n "
    "There are two images:\n"
    "1. The **first image** is a *tool scenario* labeled with white dots. There are  keypoints that have been already defined :\n "
    f"   - Grasping point: **{keypointA}**\n"
    f"   - Function point: **{keypointB}**\n"

    f"2. The **second image** shows a working scenario, where the task is to **{Task}**  using the tool from image 1.\n, there are two keypoints:start point and end point of operation\n"
  

    "3. We estabilsh robot motion and force control primitives as follows:"
    " (1) Motion primitives: \n"
    " a. **Move({tool,obj},(x1,y1),{0,1})**:{tool,obj} indicate the target point(x1,y1) is on the tool img or object img;Mode={0,1}, if 0:robot EEF free move from current point to (x2,y2) and align the direction; else if 1, robot move to make the current functional point to (x2,y2) and align direction\n"
    " b. **Linear(D,L)**: linear move from current point along direction D=(ax,ay,az)  where (0,0,1) represent the z+ direction, with moving distace L cm,  \n"
    " c. **Rotate((x1,y1),(x0,y0),r, theta)**: Robot EEF point start  at point (x1,y1), and rotate around circle center point (x0,y0), with  radius of r and rotate angle theta degree \n "
    " d. **Grip(a)**: Close gripper with Grip(1) and open gripper with Grip(0)\n"

    " (2) Force control primitives: \n"
    " a. **Off**: robot without any force control, only motion control\n"
    " b. **Constant(Fc, Tc)**: constant force control, robot moving with constant force and torque of Fc(N)and Tc(N·m)\n"
    " c. **Thresh(Fl, Tl)**: Max force control, if force and torque exceed Fl(N)and Tl(N·m), robot stop moving\n"

    "Your task is:"
    # f"The robot has finished the task step {stepAction_1}, and the tool and the robot is combined; Watch tool image and working scenario image, and given the motion primitive and force control primitive steps only for the task step {stepAction_2} not all the task"
    f"Watch tool image and working scenario image, and given the motion primitive and force control primitive for all the steps {stepAction_1}, {stepAction_2} and {stepAction_3} step by step.\n"
    "📥 **Expected output format sample:**\n"
    "-Step1 Action-1: Motion:Move(tool, (6, 2), 0), Force:Off\n"
    "-Step1 Action-2: Motion:Grip(1), Force:Off\n"
    "-Step2 Action-1: Motion:Move(tool, (6, 2), 0), Force:Off\n"
    "- ...\n"
    "Reminder:"
    "1.ALL the keypoints have been already defined, please directly use them and do not change these points\n"
    "2.The total steps number is 3.\n" 
    # "2. After the gripper finish the grasp action, add a **move up 3cm** action to aviod any collision with table, if the action is not grasp, ignore this.\n"
    "3. After the gripper finish the grasp action, all the moving actions start with the functional point of the tool rather than EEF of robot"
    "4. The robot moving primitive could only move from current EEF point, so the start point of next action is the target postion of last action\n"
    "5. Output primitives without description"
)   


response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": input_prompt,
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": dottedImg_Tool}},
                {"type": "image_url", "image_url": {"url": dottedImg_Object}}
            ]
        }
    ],
    max_tokens=800,
)
print(response.choices[0].message.content)

response_text = response.choices[0].message.content

def extract_inline_field(field_title, text):
    """Extracts a specific field from the response text."""
    pattern = rf"{field_title}:\s*(.+)"
    match = re.search(pattern, text)
    return match.group(1).strip() if match else ""

# extract fields from the response text
Step1_Action1 = extract_inline_field("Step1 Action-1", response_text)
Step1_Action2 = extract_inline_field("Step1 Action-2", response_text)
Step2_Action1 = extract_inline_field("Step2 Action-1", response_text)
Step2_Action2 = extract_inline_field("Step2 Action-2", response_text)
Step3_Action1 = extract_inline_field("Step3 Action-1", response_text)
Step3_Action2 = extract_inline_field("Step3 Action-2", response_text)



TaskCheck = extract_inline_field("Task Check Criteria", response_text)

# structure the extracted data into a dictionary
VLM3Output = {
    "Step1_Action1":Step1_Action1,
    "Step1_Action2":Step1_Action2,  
    "Step2_Action1":Step2_Action1,  
    "Step2_Action2":Step2_Action2,
    "Step3_Action1":Step3_Action1,
    "Step3_Action2":Step3_Action2
}

# save the task data to a JSON file
VLM3OutputPath= os.path.join(script_dir, './output/VLM3_Output.json')
with open(VLM3OutputPath, "w") as f:
    json.dump(VLM3Output, f, indent=4)