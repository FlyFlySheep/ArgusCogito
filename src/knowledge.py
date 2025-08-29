from src.prompt import *
from src.arglog import *
from src.qwen_chat import *
from omegaconf import DictConfig


import torch
import json
from PIL import Image

class knowledge_factory():
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.chat = chat(cfg)


    def build_prompt_ask(self, type, processor, model, RGB_summary = None, image = None):
            messages = []

            if type not in ["RGB", "Depth"]:
                raise ValueError("type must be 'RGB', 'Depth'")

            if type == "RGB":
                content_list = [
                    {"type": "text", "text": f"This is {type} Image:"},
                    {"type": "image", "image": image},
                    {"type": "text", "text": rgb_analysis_prompt()}
                ]

                messages = [{"role": "user", "content": content_list}]
            
            elif type == "Depth":
                summary = depth_analysis_prompt(RGB_summary)
                content_list = [
                    {"type": "text", "text": f"This is {type} Image:"},
                    {"type": "image", "image": image}, 
                    {"type": "text", "text": summary}
                ]

                messages = [{"role": "user", "content": content_list}]

            return self.chat.run_chat_generation(messages, processor, model)
       
    def get_knowledge(self, base_name, rgb_image, depth_image, processor=None, model=None):

        write_log_with_timestamp(self.cfg.output.log_file, f"Image Loaded, {base_name} ")

        rgb_knowledge = self.build_prompt_ask("RGB", processor, model, image = rgb_image)
        write_log_with_timestamp(self.cfg.output.log_file, f"[{base_name} RGB Image Analysis]: {rgb_knowledge}")

        depth_knowledge = self.build_prompt_ask("Depth", processor, model, RGB_summary = rgb_knowledge, image = depth_image)
        write_log_with_timestamp(self.cfg.output.log_file, f"[{base_name} Depth Image Analysis]: {depth_knowledge}")

        result = {
            "rgb_knowledge": rgb_knowledge,
            "depth_knowledge": depth_knowledge,
        }

        if self.cfg.save_knowlwdge:
            file_path = os.path.join(self.cfg.output.save_knowledge, f"{base_name}_knowledge.json")

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=4)

        return result
