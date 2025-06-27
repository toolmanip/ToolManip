from openai import OpenAI
import json
import re
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
    data = json.load(f)
task = data["task"]
tool = data["tool"]["name"]
object = data["object"]["name"]
tool_color = data["tool"]["color"]
tool_position = data["tool"]["position"]
GraspAction = data["task_steps"]["step1"]
targetObj =data["object"]["name"]

# Visual Prompt Understanding

input_prompt = (

    f"1. The **first image** shows a tool (**{tool}**)  labeled with dots.\n"
    f"2. The **second image** shows a working scenario with target object {object} of color {tool_color} and position {tool_position}, where the task is to **{task}**  using the tool from image 2.\n\n"

    "🔧 **Your task is:**\n"
    "1. On the tool image (image 1), based on dot labels identify:\n"
    f"   - **Keypoint A**: a point suitable for robotic grasping to perform **{GraspAction}**.\n"
    f"   - **Keypoint B**: the functional point on the **{tool}** used to **{task}**.\n"
    "   - **direction 1**: the direction of Z axis on the functional point.\n\n"

    "2. On the scenario image (image 2), determine:\n"
    "   - **Keypoint S1**: start point to start the operation where the functional point on the tool need to align with start point \n"
    "   - **Keypoint S2**: end point that defines the final position of the operation"
    "   - **direction 2**: the direction of Z axis on the start point. Where direction 1 must align with direction 2 before operation. \n"


    "📌 **Important Constraint:**\n"
    "- The tool's action will succeed **only if** its functional point **B** aligns with the object start point **S1**,\n"
    "  and the tool's internal action direction 1 is aligned with the **object's desired  direction 2**.\n"
    # "- Once direction 1 and 2  are aligned, the robot can move the tool along with this direction to apply force in the intended direction.\n"

    "📥 **Expected output format:\n"
    "- Total points on the tool and object imgs: \n"
    "- Keypoint A on Tool:\n"
    "- Keypoint B on Tool: \n"
    "-direction_1 on Tool: \n"
    # "- Describe each part of the tool such as the handle, head, or tip and so on based on the tool function, NOTE that the silver part is blade\n"
    "- Keypoint S1 in Scenario: \n"
    "- Keypoint S2 in Scenario: \n"
    "- direction_2 on object\n"
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
    '''Extracts a specific field from the response text.'''
    pattern = rf"{field_title}:\s*(.+)"
    match = re.search(pattern, text)
    return match.group(1).strip() if match else ""

# extract keypoints from the response text
Keypoint_A = extract_inline_field("Keypoint A on Tool", response_text)
Keypoint_B = extract_inline_field("Keypoint B on Tool", response_text)
Keypoint_S1 = extract_inline_field("Keypoint S1 in Scenario", response_text)
Keypoint_S2 = extract_inline_field("Keypoint S2 in Scenario", response_text)
direction_1 = extract_inline_field("direction_1 on Tool:", response_text)
direction_2 = extract_inline_field("direction_2 on object:", response_text)

# structure the extracted data into a dictionary
VLM2Output = {
    "Keypoint A":Keypoint_A,
    "Keypoint B":Keypoint_B,  
    "Keypoint S1":Keypoint_S1,
    "Keypoint S2":Keypoint_S2,
    "direction 1:":direction_1,
    "direction 2:":direction_2
}

# save the task data to a JSON file
VLM2OutputPath= os.path.join(script_dir, './output/VLM2_Output.json')
with open(VLM2OutputPath, "w") as f:
    json.dump(VLM2Output, f, indent=4)
