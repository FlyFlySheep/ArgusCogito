import torch, os, cv2
import numpy as np
from omegaconf import DictConfig
from typing import Tuple, List
from scipy.ndimage import binary_opening

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX
from PIL import Image, ImageDraw

class Sam4Segment:
    def __init__(self, cfg: DictConfig, image_processor, model, tokenizer):
        self.cfg = cfg
        self.image_processor = image_processor
        self.model = model
        self.tokenizer = tokenizer

    def sel_points(self, rand_points, all_probs_2, neg_thres=0.2, pos_thres=0.8):
        
        sel_points, sel_labels = [], []
        for (x, y), score in zip(rand_points, all_probs_2):
            if score[0] > neg_thres:
                sel_points.append((x, y)), sel_labels.append(0)
            elif score[1] > pos_thres:
                sel_points.append((x, y)), sel_labels.append(1)

        return np.array(sel_points), np.array(sel_labels)
    

    def point_predict(self, bbox, image, s_phrase, answer_counts='1'):

        original_size = image.size 
        sw, sh = [int(x) for x in original_size]
        scale_x = 1000 / sw
        scale_y = 1000 / sh
        image_tensor = process_images([image], self.image_processor, self.model.config)
        image_tensor = [_image.to(dtype=torch.float16) for _image in image_tensor]
        image_sizes = [image.size]

        text_output = bbox
        [x1, y1, x2, y2] = bbox

        x1_1000 = int(x1 * scale_x)
        y1_1000 = int(y1 * scale_y)
        x2_1000 = int(x2 * scale_x)
        y2_1000 = int(y2 * scale_y)
        width = x2_1000 - x1_1000
        height = y2_1000 - y1_1000

        grid_size = int(np.ceil(np.sqrt(10)))
        points_per_dim = grid_size
        x_grid = np.linspace(0.1, 0.9, points_per_dim)
        y_grid = np.linspace(0.1, 0.9, points_per_dim)

        grid_points = np.array([(x, y) for x in x_grid for y in y_grid])

        if len(grid_points) > 10:
            indices = np.random.choice(len(grid_points), 10, replace=False)
            grid_rand_points = grid_points[indices]
        else:
            grid_rand_points = grid_points

        rand_points = grid_rand_points * np.array([width, height]) + np.array([x1_1000, y1_1000])
        rand_points = rand_points.astype(int)

        rand_points[:, 0] = np.clip(rand_points[:, 0], x1_1000, x2_1000 - 1)
        rand_points[:, 1] = np.clip(rand_points[:, 1], y1_1000, y2_1000 - 1)
        text_output = f"[{x1_1000}, {y1_1000}, {x2_1000}, {y2_1000}]"

        points_txt = ' '.join([f"({p[0]:03d},{p[1]:03d})" for p in rand_points])
        question_points = f'Check if the points listed below are located on the object with coordinates [{text_output}]:\n{points_txt}'

        prompt_question = self.tokenizer.apply_chat_template([
            {"role": "system",
             "content": "You are a helpful language and vision assistant.You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."},
            {"role": "user",
             "content": f'<image>\nPlease provide the bounding box coordinate of the region this sentence describes ({answer_counts}):\n"{s_phrase}."'},
            {"role": "assistant", "content": text_output},
            {"role": "user", "content": question_points},
        ], tokenize=False, add_generation_prompt=True)

        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX,
                                          return_tensors="pt").unsqueeze(0).to('cuda:0')

        output_2 = self.model.generate(
            input_ids,
            images=[x.half() for x in image_tensor],
            image_sizes=image_sizes,
            max_new_tokens=30,
            output_logits=True,
            return_dict_in_generate=True,
        )

        yesno_probs = torch.stack(output_2['logits'], dim=1).softmax(dim=-1)
        yesno_probs = yesno_probs[0, :30, [2822, 9642]].float().cpu().numpy()

        points_sel, labels_sel = self.sel_points(rand_points, yesno_probs, neg_thres=0.9, pos_thres=0.75)

        return points_sel, labels_sel

    def segment_one_bbox(self, bbox, bbox_idx, rgb_image, sam2_predictor, base_name):

        img_w, img_h = rgb_image.size
        scale_x = 1000 / img_w
        scale_y = 1000 / img_h
        rgb_image_np = np.array(rgb_image)
        sam2_predictor.set_image(rgb_image_np)

        if self.cfg.use_sam4mllm:
            all_points = []
            all_labels = []

            for r in self.cfg.rounds:
                round_id = r.id
                words = r.words

                sel_points, sel_labels = self.point_predict(bbox, rgb_image, words)
                print(sel_points)

                points_original = sel_points.copy().astype(float)
                points_original[:, 0] = sel_points[:, 0] / scale_x
                points_original[:, 1] = sel_points[:, 1] / scale_y
                points_original = points_original.astype(int)
                
                points_original[:, 0] = np.clip(points_original[:, 0], 0, img_w - 1)
                points_original[:, 1] = np.clip(points_original[:, 1], 0, img_h - 1)
                
                all_points.append(points_original)
                print(all_points,'all_points')
                all_labels.append(sel_labels)

                print(round_id,'finished')

            if all_points and all_labels:

                try:
                    combined_points = np.concatenate(all_points, axis=0)
                    combined_labels = np.concatenate(all_labels, axis=0)

                    if len(combined_points) > 0:
                        points_str = [f"{int(p[0])},{int(p[1])}" for p in combined_points]
                        _, unique_indices = np.unique(points_str, return_index=True)
                        combined_points = combined_points[unique_indices]
                        combined_labels = combined_labels[unique_indices]
                        
                except Exception as e:
                    print(f"Data Error: {e}")

        best_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        
        try:
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
                
                if self.cfg.use_sam4mllm:
                    masks, _, _ = sam2_predictor.predict(
                        point_coords=combined_points,
                        point_labels=combined_labels,
                        box=np.array(bbox, dtype=np.float32),
                        multimask_output=False
                    )
                else:
                    masks, _, _ = sam2_predictor.predict(
                        box=np.array(bbox, dtype=np.float32),
                        multimask_output=False
                    )
            
            if masks is not None and len(masks) > 0:
                mask = masks[0]
                
                if isinstance(mask, torch.Tensor):
                    mask = mask.detach().cpu().numpy()
                
                if mask.ndim == 3:
                    mask = mask.squeeze(0)
                
                if mask.shape != (img_h, img_w):
                    mask = cv2.resize(mask.astype(np.float32), (img_w, img_h), interpolation=cv2.INTER_LINEAR)
                

                best_mask = (mask > 0.5).astype(np.uint8) * 255

                best_mask = (binary_opening(best_mask > 0.5) * 255).astype(np.uint8)
                
        except Exception as e:
            print(f"[Error {base_name}] SAM2 Failed: {e}")

        mask_binary_path = os.path.join(self.cfg.output.save_file, f"{base_name}_mask_idx{bbox_idx}.png")
        
        try:
            Image.fromarray(best_mask).convert("L").save(mask_binary_path)
        except Exception as e:
            print(f"Mask Saved Error: {e}")

        try:
            draw_image = rgb_image.copy().convert('RGBA')
            
            mask_rgba = np.zeros((img_h, img_w, 4), dtype=np.uint8)
            mask_rgba[..., 1] = best_mask 
            mask_rgba[..., 3] = (best_mask * 0.3).astype(np.uint8) 
            mask_img = Image.fromarray(mask_rgba, mode='RGBA')
            
            draw_image = Image.alpha_composite(draw_image, mask_img)
            
            draw = ImageDraw.Draw(draw_image)
            draw.rectangle(bbox, outline='red', width=5)
            
            if self.cfg.use_sam4mllm:
                colors = {0: (0, 0, 255), 1: (255, 0, 0)}
                
                for points, labels in zip(all_points, all_labels):
                    for (x, y), label in zip(points, labels):
                        color = colors.get(label, (255, 255, 255))
                        draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill=color)
            
                vis_save_path = os.path.join(self.cfg.output.save_file, f"{base_name}_sam4mllm_idx{bbox_idx}.png")
            else:
                vis_save_path = os.path.join(self.cfg.output.save_file, f"{base_name}_idx{bbox_idx}.png")
            
            draw_image.convert('RGB').save(vis_save_path)
            print(f"[Save] Visiable Image: {vis_save_path}")
            
        except Exception as e:
            print(f"Error Save Visiable Image: {e}")

        if self.cfg.use_sam4mllm:
            try:
                mask_points_vis_path = os.path.join(self.cfg.output.save_file, f"{base_name}_mask_with_points_idx{bbox_idx}.png")
                img_draw = rgb_image.copy().convert("RGB")
                draw = ImageDraw.Draw(img_draw)
                
                for points, labels in zip(all_points, all_labels):
                    for (x, y), label in zip(points, labels):
                        color = (0, 255, 0) if label == 1 else (255, 0, 0)
                        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color, outline=color)
                
                if np.sum(best_mask > 0) > 0:
                    mask_pil = Image.fromarray(best_mask, mode="L")
                    green_mask = Image.new("RGBA", mask_pil.size, (0, 255, 0, 128))
                    blended = Image.blend(mask_pil.convert("RGBA"), green_mask, alpha=0.5)
                    img_draw.paste(blended, (0, 0), blended)
                
                img_draw.save(mask_points_vis_path)
                
            except Exception as e:
                print(f"Error Save Visiable Image: {e}")
                
        return best_mask
