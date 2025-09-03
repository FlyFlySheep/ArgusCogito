import torch
from omegaconf import DictConfig

class chat():

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def run_chat_generation(self, messages, processor, model):
        with torch.no_grad():
            image_inputs = []
            for message in messages:
                content = message.get('content')
                if isinstance(content, list):
                    for part in content:
                        if part.get('type') == 'image':
                            image_inputs.append(part['image'])

            text = processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )

            if image_inputs:
                inputs = processor(
                    text=[text], 
                    images=image_inputs,
                    return_tensors="pt"
                ).to(model.device)
            else:
                inputs = processor(
                    text=[text],
                    return_tensors="pt"
                ).to(model.device)

            output_ids = model.generate(**inputs, max_new_tokens= self.cfg.max_new_tokens)
            
            input_ids_len = inputs.input_ids.shape[1]
            trimmed = output_ids[:, input_ids_len:]
            answer = processor.batch_decode(trimmed, skip_special_tokens=True)[0]

            del output_ids, trimmed, inputs, text, image_inputs
            torch.cuda.empty_cache()
            return answer