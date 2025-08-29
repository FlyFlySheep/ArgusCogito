import os
import sys
from PIL import Image
import json
import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from src.argfocus import *
from src.knowledge import *
from src.segment import *

import hydra
from omegaconf import DictConfig
from hydra.core.global_hydra import GlobalHydra

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def task(basename, rgb_image, depth_image):

    pack = (basename, rgb_image, depth_image)
    return pack

@hydra.main(version_base=None, config_path="config", config_name="config")

def main(cfg: DictConfig) -> None:
    ArgusCogito_focus = qwen2_5_focus(cfg)
    ArgusCogito_kfc = knowledge_factory(cfg)
    
    try:
        qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            cfg.model.qwen2_vl_path,
            torch_dtype="auto",
            device_map="auto"
        )
        qwen_processor = AutoProcessor.from_pretrained(cfg.model.qwen2_vl_path)
    except Exception as e:
        return
    
    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()

    from sam2.sam2_image_predictor import SAM2ImagePredictor
    try:
        sam2_predictor = SAM2ImagePredictor.from_pretrained(cfg.model.sam2_ckpt)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        sam2_predictor.model.to(device)
    except Exception as e:
        raise

    tokenizer, model, image_processor, _ = load_pretrained_model(
        cfg.model.llava_ckpt,
        None,
        "llava_llama3",
        device_map='auto',
        torch_dtype="float16",
        attn_implementation='eager',
    )

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        model.resize_token_embeddings(len(tokenizer))

    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    model.tie_weights()
    model = PeftModel.from_pretrained(model, cfg.model.sam4mllm_ckpt)

    os.makedirs(cfg.output.save_bboxes, exist_ok=True)
    os.makedirs(cfg.output.save_file, exist_ok=True)
    os.makedirs(cfg.output.save_knowledge, exist_ok=True)

    if cfg.image:
        base_name = cfg.image
        rgb_image_path = cfg.input.RGB_image + base_name + cfg.rgb_image_extention
        depth_image_path = cfg.input.Depth_image + base_name + cfg.depth_image_extention
        packs = [task(base_name, rgb_image_path, depth_image_path)]
        print(f"Will process: single ")

    else:
        rgb_files = {
            os.path.splitext(f)[0]: os.path.join(cfg.input.RGB_image, f)
            for f in os.listdir(cfg.input.RGB_image)
            if f.endswith(cfg.rgb_image_extention)
        }

        depth_files = {
            os.path.splitext(f)[0]: os.path.join(cfg.input.Depth_image, f)
            for f in os.listdir(cfg.input.Depth_image)
            if f.endswith(cfg.depth_image_extention)
        }

        common_basenames = set(rgb_files.keys()) & set(depth_files.keys())
        packs = [
            task(basename, rgb_files[basename], depth_files[basename])
            for basename in common_basenames
        ]
        print(f"Will process: {len(packs)} ")

    for base_name, rgb_image_path, depth_image_path in packs:

        print(base_name)
        rgb_image = Image.open(rgb_image_path)
        depth_image = Image.open(depth_image_path)

        if not cfg.use_bboxes_origin:
            if cfg.use_knowledge:
                if cfg.use_knowledge_origin:
                    file_path = os.path.join(cfg.input.knowledge_origin, f"{base_name}_knowledge.json")
                    if os.path.exists(file_path):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            knowledge = json.load(f)
                    else:
                        knowledge = {}
                else:
                    knowledge = ArgusCogito_kfc.get_knowledge(
                        base_name, rgb_image, depth_image, qwen_processor, qwen_model
                    )

                bboxes = ArgusCogito_focus.focus(
                    qwen_processor, qwen_model, base_name, rgb_image, knowledge
                )
            else:
                bboxes = ArgusCogito_focus.focus(
                    qwen_processor, qwen_model, base_name, rgb_image
                )
        else:
            bbox_file_path = os.path.join(cfg.output.save_bboxes, f"{base_name}_bboxes.json")
            with open(bbox_file_path, 'r') as f:
                bboxes = json.load(f)

        ArgusCogito_Seg = Sam4Segment(cfg, image_processor, model, tokenizer)

        final_mask = np.zeros((rgb_image.height, rgb_image.width), dtype=np.uint8)

        for idx, bbox in enumerate(bboxes):
            mask = ArgusCogito_Seg.segment_one_bbox(
                bbox, idx, rgb_image, sam2_predictor, base_name
            )
            final_mask = np.maximum(final_mask, (mask > 0).astype(np.uint8) * 255)

        final_mask_pil = Image.fromarray(final_mask, mode="L")
        final_mask_path = os.path.join(ArgusCogito_Seg.cfg.output.save_file, f"{base_name}_final_mask.png")
        final_mask_pil.save(final_mask_path)
        print(f"[Saved] Final Mask: {final_mask_path}")

if __name__ == "__main__":
    main()
