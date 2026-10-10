from string import Template

find_circuit_region_prompt = """
This is a preprocessed circuit diagram. Image preprocessing has removed many thin strokes and redundant visual details, but it may also have partially removed the boundaries of the rectangular IC packages.

Your task is to identify the remaining rectangular IC packages and estimate the bounding box of each complete package.

Instructions:
- Look for the large rectangular structures representing IC packages.
- The rectangle boundaries may be incomplete, and some internal circuit symbols may remain visible.
- Do not treat small rectangles, isolated line segments, or external components as target IC packages.
- Exclude external wires and components from each bounding box.
- There may be multiple IC packages in the image. Return one bounding box for each detected package.
- Estimate the full outer boundary of each package, including portions that have been removed by preprocessing.

Return ONLY the bounding boxes, one per line, in this format:
y1,x1,y2,x2

Coordinates must be normalized to the range 0–1000, relative to the image height and width respectively.
Do not include explanations, your reasoning or any other text.
"""
find_IC_PIN_prompt = Template(
    """
    You are analyzing a circuit diagram for semantic extraction.

    The supplied bounding box identifies the target IC region in the original circuit image.

    Bounding box in [x1, y1, x2, y2] format:
    $box_str

    Analyze the target IC and its visible pin connections.

    For each pin or terminal, report:
    1. Pin number or label, if visible.
    2. Pin location on the IC boundary (top, bottom, left, or right).
    3. The wire or net visibly connected to it.
    4. Any connected external component or labeled signal, if identifiable.
    5. Confidence and visual evidence.

    Json Format(Slot/Type/Purpose):
    1. pin_number/string or null/Visible pin number or label
    2. location/string/	top, bottom, left, or right
    3. connected_net/string or null/Visible net label, such as +VS, C, or V_en
    4. connected_component/string or null/Connected external component, if identifiable
    5. confidence/string/high, medium, or low

    Rules:
    - Use the original image to inspect wires and labels around the IC boundary.
    - Trace connections only when supported by visible evidence.
    - Do not invent missing pin numbers, labels, or convnections.
    - If a connection cannot be determined from the image, mark it as uncertain.
    - Distinguish directly observed connections from inferred ones.

    Return the result as JSON.
    """
)