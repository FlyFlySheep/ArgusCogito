import re, json
from PIL import Image, ImageDraw
from omegaconf import DictConfig
from src.arglog import *
from src.prompt import *
from src.qwen_chat import *

class qwen2_5_focus():

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.chat = chat(cfg)

    def resize_image(self, image: Image.Image):

        orig_width, orig_height = image.size
        long_edge = max(orig_width, orig_height)
        scale = self.cfg.target_long_edge / long_edge 
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return resized_image, new_width, new_height
    
    def get_bboxes(self, input_string: str):
        
        json_match = re.search(r"```json\s*([\s\S]*?)\s*```", input_string, re.DOTALL)
        if not json_match:
            return [] 

        json_str = json_match.group(1).strip()
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return []

        all_bboxes = []
        data_list = [data] if isinstance(data, dict) else data if isinstance(data, list) else []

        for item in data_list:
            if isinstance(item, dict):
                bbox = item.get("bbox_2d")
                if isinstance(bbox, list) and len(bbox) == 4:
                    all_bboxes.append(bbox)
        return all_bboxes


    def focus(self, processor, model, basename, image=None, knowledge=None):
        
        if self.cfg.focus_type not in ["left_middle_right", "up_middle_down"]:
            raise ValueError("type must be left_middle_right, up_middle_down")
       
        messages = []
        write_log_with_timestamp(self.cfg.output.log_file, f"{basename} Focus on {self.cfg.focus_type}")

        if knowledge is not None:
            messages.append({"role": "user", "content": knowledge})

        messages.append({"role": "user", "content": [{"type": "image", "image": image}]})
        messages.append({"role": "user", "content": focus_prompt(self.cfg.focus_type)})

        step1_response = self.chat.run_chat_generation(messages, processor, model)
        messages.append({"role": "assistant", "content": step1_response})

        messages.append({"role": "user", "content": focus_prompt('bbox')})
        
        result = self.chat.run_chat_generation(messages, processor, model)
        write_log_with_timestamp(self.cfg.output.log_file, f"{basename} Focus result for {self.cfg.focus_type}: {result}") # Log snippet
        
        bboxes = self.get_bboxes(result)

        if self.cfg.save_bboxes:
            file_path = os.path.join(self.cfg.output.save_bboxes, f"{basename}_bboxes.json")

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(bboxes, f, ensure_ascii=False, indent=4)
        return bboxes