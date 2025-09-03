def camo_rgb_analysis_prompt(target_type: str):

    if target_type not in ['camo_animal', 'camo_plant']:
        raise ValueError("target_type must be 'camo_animal' or 'camo_plant'")

    specialty_object = "animals" if target_type == 'camo_animal' else "plants"

    prompt = f"""Argus, you are Detective Lane, an analyst with expertise in detecting hidden {specialty_object} in their natural environment.
                Your specialty is identifying camouflaged, mimicking, or partially obscured {specialty_object},
                especially those that are long-shaped or buried in the environment.
                Analyze the provided RGB image carefully, and look for:
                - Unusual shapes or patterns that may indicate {specialty_object}
                - Subtle color or texture differences that suggest camouflage
                - Environmental context that supports {specialty_object} presence."""
    return prompt

def camo_depth_analysis_prompt(target_type: str, ans_rgb_comprehensive: str):

    if target_type not in ['camo_animal', 'camo_plant']:
        raise ValueError("target_type must be 'camo_animal' or 'camo_plant'")

    prompt = f"""Based on the RGB image analysis: {ans_rgb_comprehensive}
                Analyze the depth map to identify potential hidden or camouflaged {target_type}. Focus on:
                - **Depth Layers**: Determine if there are clear foreground/midground/background layers and where the {target_type} (if any) might be located.
                - **Depth Distribution**: Check for any object with abnormal depth compared to its surroundings.
                - **Geometric Anomalies**: Look for depth discontinuities, bumps, or unusual shapes.
                - **Camouflage Clues**: Identify any object that visually blends with the background but shows a different depth value."""
    return prompt

def camo_focus_prompt(target_type: str, focus_type: str):

    if target_type not in ['camo_animal', 'camo_plant']:
        raise ValueError("target_type must be 'camo_animal' or 'camo_plant'")
    
    if focus_type not in ["left_middle_right", "up_middle_down"]:
        raise ValueError("Invalid focus_type. Choose from 'left_middle_right', 'up_middle_down'.")
    
    if focus_type in ["left_middle_right", "up_middle_down"]:
        region_division = "three vertical regions: left, middle, and right" if focus_type == "left_middle_right" else "three horizontal regions: top, middle, and bottom"
        
        if target_type == 'camo_animal':
            target_phrase = "any animal or human present (excluding plants)"
        else:
            target_phrase = "any plant present (excluding animals and humans)"

        return f"""Divide the image into {region_division}. 
                   For each region, carefully examine whether there is {target_phrase}. 
                   For each region, clearly state whether {target_phrase} is present and describe the visual clues that support your judgment, 
                   such as color, texture, and shape."""

    elif focus_type == "bbox":
        if target_type == 'camo_animal':
            return """Analyze the entire image to identify potential animals or humans (excluding plants). Focus on detecting biological entities, and when a biological entity is identified, 
                        ensure the bounding box covers the entire visible structure of that organism, including key features such as the head, limbs, body, and tail (if visible). 
                        If there are multiple candidates, provide a bounding box for the candidate that blends well with the background in terms of color and texture, but ensure each box fully covers the biological entity. 
                        If uncertain, provide a bounding box that most likely contains the complete biological entity. If no obvious biological entities are detected in the image, 
                        provide a bounding box around the area that most closely resembles potential biological structures (e.g., shapes, colors, or textures similar to animals or humans). 
                        Example output format:
                        ```json
                        [
                            {"bbox_2d": [x1, y1, x2, y2]}
                        ]
                        ```"""
        else: # plant
            return """  Analyze the entire image to identify potential plants (excluding animals and humans). Focus on detecting botanical entities, and when a plant is identified, ensure the bounding box covers the entire visible structure of that organism, including key features such as leaves, stems, branches, or roots (if visible).
                        If there are multiple candidate regions, provide a bounding box for the one that blends most effectively with the surrounding environment in terms of color, texture, and shape—such as a plant camouflaged among rocks, sand, or urban debris—but ensure the box fully encloses all visible parts of the plant.
                        If uncertain, provide a bounding box that most likely contains the complete plant structure based on subtle visual cues like organic symmetry, repeating patterns, edge irregularities, or localized changes in surface texture.
                        Example output format:
                        ```json
                        [
                            {"bbox_2d": [x1, y1, x2, y2]}
                        ]
                        ```"""
        
def polyp_rgb_analysis_prompt():
    prompt = """You are Argus, a medical imaging analyst with expertise in detecting subtle and camouflaged polyps in colonoscopy images. 
                Your specialty is identifying polyps that are flat, small, partially obscured, or blend into the surrounding mucosa.
                Analyze the provided RGB endoscopic image carefully, and look for:
                - Unusual shapes or contours that may indicate a polyp (e.g., slight elevation, irregular borders)
                - Subtle color variations (e.g., redder, yellower, or darker regions compared to surrounding tissue)
                - Texture differences (e.g., increased granularity, vascular patterns, or surface irregularities)
                - Environmental context such as folds, shadows, mucus, or debris that might obscure a polyp
                Do not assume there is no polyp present. Instead, highlight suspicious areas and explain your reasoning based on visual clues such as chromatic deviation, morphological asymmetry, or textural anomaly."""
    return prompt


def polyp_depth_analysis_prompt(ans_rgb_comprehensive):
    prompt = f"""Based on the RGB image analysis: {ans_rgb_comprehensive}
                Analyze the depth map (or topographic reconstruction) of the endoscopic scene to identify potential hidden or flat polyps. Focus on:
                - **Depth Layers**: Determine if there are distinct tissue layers and whether any region shows slight protrusion or depression relative to the mucosal surface.
                - **Depth Distribution**: Check for areas with abnormal elevation or indentation compared to the surrounding flat mucosa.
                - **Geometric Anomalies**: Look for subtle bumps, dome-like structures, or irregular contours that disrupt the smooth intestinal wall.
                - **Camouflage Clues**: Identify regions that visually blend with the background in color but show a different 3D profile (e.g., a slightly raised lesion with similar hue).
                Avoid concluding "no polyp present." Highlight suspicious areas and explain your reasoning based on depth discontinuities, curvature changes, or volumetric deviations."""
    return prompt


def polyp_focus_prompt(type):
    if type == "left_middle_right":
        prompt = """Divide the image into three vertical regions: left, middle, and right. 
                    For each region, carefully examine whether there is any polyp or abnormal tissue growth present (excluding normal folds, blood vessels, or debris). 
                    For each region, clearly state whether a suspicious lesion is present and describe the visual clues that support your judgment, 
                    such as color variation (e.g., NBI-like enhancement), texture change (e.g., pit pattern), shape irregularity, or subtle elevation."""
    
    elif type == "up_middle_down":
        prompt = """Divide the image into three horizontal regions: top, middle, and bottom. 
                    For each region, carefully examine whether there is any polyp or abnormal mucosal lesion present (excluding artifacts like bubbles or mucus). 
                    For each region, clearly state whether a biological anomaly is present and describe the visual clues that support your judgment, 
                    such as chromatic contrast, surface texture, border definition, or structural protrusion."""

    elif type == "bbox":
        prompt = """Analyze the entire endoscopic image to identify potential colorectal polyps. Focus on detecting subtle lesions, including flat adenomas, sessile serrated polyps, or diminutive growths that may blend with the mucosa. 
                    When a polyp is identified, ensure the bounding box tightly covers the entire visible extent of the lesion, including its base, margins, and any central depression or vessel pattern. 
                    If multiple candidates exist, prioritize the one with suspicious morphological features (e.g., irregular border, non-uniform color, Kudo pit pattern) even if partially occluded. 
                    If uncertain, provide a bounding box around the region most likely to contain a true polyp — this should be the area showing the strongest deviation in color, texture, or elevation from the surrounding tissue. 
                    Note: Every image contains at least one clinically relevant polyp. Do not return empty results. 
                    Example output format:
                    ```json
                    [
                        {"bbox_2d": [x1, y1, x2, y2]}
                    ]
                    ```"""
    return prompt
