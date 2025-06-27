from openai import OpenAI
import re
import json
import os
client = OpenAI(
    base_url='', # Your OpenAI API base URL here, e.g., 'https://api.openai.com/v1'
    api_key='' # Your API Key here
)
script_dir = os.path.dirname(__file__)
originalImgURLPath= os.path.join(script_dir, './img_url/Original_Img_URL.json')
with open(originalImgURLPath, "r") as f:
    dataURL = json.load(f)
OriToolImg = dataURL["Original Tool Img URL"]
OriObjImg = dataURL["Original Object Img URL"]


task_description = "Sweep the board"

# CoT-1:Tool understanding

input_prompt = "These are the pictures of the given tools. " \
                "Describe the tools in the scenario, including their spatial positions, color, and their functions. (tool, position, color, function)" 
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
                {"type": "image_url", "image_url":{"url":  OriToolImg}},
            ]
        }
    ],
    max_tokens=300,
)
# print(response.choices[0].message.content)

# CoT-2:Task Planning

input_prompt =  f"You are a robot with 2 finger gripper on the End joint, and is going to {task_description} based on the scenario above." \
                " You need to choose tools in the first picture to do this task.And the second picture is the object scenario. Please response " \
                "Expected output format:: " \
                "-Tool Name:\n" \
                "-Tool Color:\n" \
                "-Tool Position: \n" \
                "-Object Name:\n" \
                "-Object Color:\n" \
                "-Object Position\n" \
                "Task steps description (step1, step2 … ). Following the steps of 1.grasp tool. 2.Align the tool with target object. 3.Move tool to finish task. \n" \
                "-Task Step 1: \n"\
                "-Task Step 2: \n"\
                "-Task Step 3: \n"\
                "In terms of step description, Use Verb + object + detailed onjects. Eg: move down 5cm with 10n force.\n"\
                "Finally, make a simple criteria to check if the task is completed. Eg: the object is in the target area.\n"\
                "Task Check Criteria: \n"\
                "**Reminder:**\n"\
                "Keep all the seperate output information on the same line with their name.Such as  Task Step 1: description..."



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
                {"type": "image_url", "image_url": OriToolImg},
                {"type": "image_url", "image_url": OriObjImg}
            ]
        }
    ],
    max_tokens=500,
)
# print(response.choices[0].message.content)

response_text = response.choices[0].message.content

def extract_inline_field(field_title, text):
    '''extract inline field from text'''
    pattern = rf"{field_title}:\s*(.+)"
    match = re.search(pattern, text)
    return match.group(1).strip() if match else ""

# extract fields from the response text
ToolName = extract_inline_field("Tool Name", response_text)
ToolColor = extract_inline_field("Tool Color", response_text)
ToolPosition = extract_inline_field("Tool Position", response_text)

ObjectName = extract_inline_field("Object Name", response_text)
ObjectColor = extract_inline_field("Object Color", response_text)
ObjectPosition = extract_inline_field("Object Position", response_text)

Step1 = extract_inline_field("Task Step 1", response_text)
Step2 = extract_inline_field("Task Step 2", response_text)
Step3 = extract_inline_field("Task Step 3", response_text)

TaskCheck = extract_inline_field("Task Check Criteria", response_text)

# structure the extracted data into a dictionary
task_data = {
    "task":task_description,
    "tool": {
        "name": ToolName,
        "color": ToolColor,
        "position": ToolPosition
    },
    "object": {
        "name": ObjectName,
        "color": ObjectColor,
        "position": ObjectPosition
    },
    "task_steps":{
        "step1": Step1,
        "step2": Step2,
        "step3": Step3,
    },
    "task_check_criteria": TaskCheck
}

# save the task data to a JSON file
VLM1_OutputPath= os.path.join(script_dir, './output/VLM1_Output.json')

with open(VLM1_OutputPath, "w") as f:
    json.dump(task_data, f, indent=4)