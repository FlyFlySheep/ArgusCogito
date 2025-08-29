def rgb_analysis_prompt():
    prompt = """You are Argus, a Detective Lane, an analyst with expertise in detecting hidden animals in natural environments.
                Your specialty is identifying camouflaged, mimicking, or partially obscured animals,
                especially those that are long-shaped or buried in the environment.
                Analyze the provided RGB image carefully , and look for:
                Unusual shapes or patterns that may indicate an animal
                Subtle color or texture differences that suggest camouflage
                Environmental context that supports animal presence (e.g., sand, foliage, rocks)
                Do not assume there is no animal present. Instead, highlight suspicious areas and explain your reasoning based on visual clues."""
    return prompt

def depth_analysis_prompt(ans_rgb_comprehensive):
    prompt= f"""Based on the RGB image analysis: {ans_rgb_comprehensive}
                Analyze the depth map to identify potential hidden or camouflaged animals. Focus on:
                - **Depth Layers**: Determine if there are clear foreground/midground/background layers and where the animal (if any) might be located.
                - **Depth Distribution**: Check for any object with abnormal depth compared to its surroundings.
                - **Geometric Anomalies**: Look for depth discontinuities, bumps, or unusual shapes.
                - **Camouflage Clues**: Identify any object that visually blends with the background but shows a different depth value.
                Avoid concluding "no animal present." Highlight suspicious areas and explain your reasoning based on depth clues."""
    
    return prompt

def focus_prompt(type):
    if type == "left_middle_right":
        prompt = """Divide the image into three vertical regions: left, middle, and right. 
                    For each region, carefully examine whether there is any animal or human present (excluding plants). 
                    For each region, clearly state whether a biological entity is present and describe the visual clues that support your judgment, 
                    such as color, texture, and shape."""
    
    elif type == "up_middle_down":
        prompt = """Divide the image into three horizontal regions: top, middle, and bottom. 
                    For each region, carefully examine whether there is any animal or human present (excludi
                    For each region, clearly state whether a biological entity is present and describe the visual clues that support your judgment, 
                    such as color, texture, and shape."""
    
    elif type == "bbox":
        prompt = """Analyze the entire image to identify potential animals or humans (excluding plants). Focus on detecting biological entities, and when a biological entity is identified, 
                    ensure the bounding box covers the entire visible structure of that organism, including key features such as the head, limbs, body, and tail (if visible). 
                    If there are multiple candidates, provide a bounding box for the candidate that blends well with the background in terms of color and texture, but ensure each box fully covers the biological entity. 
                    If uncertain, provide a bounding box that most likely contains the complete biological entity. If no obvious biological entities are detected in the image, 
                    provide a bounding box around the area that most closely resembles potential biological structures (e.g., shapes, colors, or textures similar to animals or humans). 
                    This should be the area that is visually most likely to contain hidden or camouflaged entities, even in the face of uncertainty. Note that each image should contain a camouflaged biological entity or person.
                    Example output format:
                    ```json
                    [
                        {"bbox_2d": [x1, y1, x2, y2]}
                    ]
                    ```"""
    
    return prompt