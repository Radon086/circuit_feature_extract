from transformers import AutoProcessor, AutoModelForMultimodalLM
from visual_util import comp_detection, image_pipeline
from agents import agent_component, agent_wire, agent_netlist, agent_trivial_fix
from matirx import component_matrix, wire_matrix, netlist_matrix
from langgraph.graph import StateGraph, START, END
from state import TopoGenState
from nodes import *
from functools import partial
from string import Template

MODEL_PATH = "./model/gemma-4/gemma-4-12B-it"
model = AutoModelForMultimodalLM.from_pretrained(MODEL_PATH, dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained(MODEL_PATH)

img = './imgs/app_circuits/l298_8.png'

txt_prompt = Template( 
    """
    ### Task:
    "list all the application circuit in the imaga and return its boundingbox range.

    ### "Rules:" 
    "- No explanation" 
    "- No analysis" 
    "- No commentary" 
    """
)

prompt = [
    {
        "role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": txt_prompt}
        ]
    }
]

def remove_model_thinktags(response):
    if response.endswith("<turn|>"):
        response = response[:-7]
    if "<channel|>" in response:
        index = response.find("<channel|>")
        response = response[index+10:]
    return response

inputs = processor.apply_chat_template(
    prompt,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
).to(model.device)


input_len = inputs["input_ids"].shape[-1]

# Generate output
outputs = model.generate(**inputs, max_new_tokens=512, temperature = 0.1)
response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
processor.parse_response(response)
response = remove_model_thinktags(response)
print("Response:", response)