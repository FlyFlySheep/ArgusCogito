# ArgusCogito: Chain-of-Thought for Cross-Modal Synergy and Omnidirectional Reasoning in Camouflaged Object Segmentation  

## ⚙️ Environment Requirements  
- Python >= 3.9  
- PyTorch >= 2.0 (with GPU support)  
- Dependencies:  
  ```bash
  pip install -r requirements.txt
  ```  

  Main dependencies:  
  - transformers  
  - peft  
  - Pillow  
  - hydra-core  

---

## 🚀 How to use

### Modify the config.yaml

```yaml
image: null  # image to process only

# If you specify an image, only this image will be inferred (only basename, no extention)

dataset:  # Name of the dataset used [camo_plant, camo_animal, polyp]

use_knowledge: True
use_knowledge_origin: False # use the knowledge already have
use_bboxes_origin: False

use_sam4mllm: True # use sam4mllm to provide point hints
save_knowlwdge: True
save_bboxes: True

focus_type: left_middle_right

target_long_edge: 1000
max_new_tokens: 1024

input:
  knowledge_origin: # The folder for knowledge of the images
  RGB_image: # dataset rgb image
  Depth_image: # dataset depth image

output:
  log_file: # .log file for logging
  save_file: # folder for saving visible images and masks
  save_knowledge: # folder for saving knowledge
  save_bboxes: # folder for saving bboxes

model:
  qwen2_vl_path: # model path
  sam2_ckpt: facebook/sam2.1-hiera-large # dafault path for sam2
  llava_ckpt: # model path
  sam4mllm_ckpt: # sam4mllm+ path


rounds:
  - id: 1
    words: ['animal or human']
  - id: 2
    words: ['animal or human']


rgb_image_extention: .jpg
depth_image_extention: .png
```

### 3. Running
```bash
python main.py
```

Only want to process one image：
```bash
python main.py image=example_001
```

## Checkpoints

- `llava-next`：[llava-next](https://huggingface.co/lmms-lab/llama3-llava-next-8b)
- `sam4mllm+`：[sam4mllm+](https://drive.google.com/drive/folders/1ytEfGRa6bxThTXQn5MLVKKy4jsxxBo6M)
- `Qwen2.5-VL-7B-Instruct`: [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- `sam2.1-hiera-large`:  Directly pulled by using command `facebook/sam2.1-hiera-large`

## Acknowledge

This project borrows some code from [sam4mllm](https://github.com/AI-Application-and-Integration-Lab/SAM4MLLM), thanks for their admiring contributions~!

## Reference

```bibtex
@misc{tan2025arguscogitochainofthoughtcrossmodalsynergy,
      title={ArgusCogito: Chain-of-Thought for Cross-Modal Synergy and Omnidirectional Reasoning in Camouflaged Object Segmentation}, 
      author={Jianwen Tan and Huiyao Zhang and Rui Xiong and Han Zhou and Hongfei Wang and Ye Li},
      year={2025},
      eprint={2508.18050},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.18050}, 
}
