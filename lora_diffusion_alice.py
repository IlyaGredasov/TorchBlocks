# accelerate launch --num_processes=1 --mixed_precision=bf16 `
# >>   diffusers/examples/text_to_image/train_text_to_image_lora_sdxl.py `
# >>   --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" `
# >>   --train_data_dir="Datasets/AlicePack" `
# >>   --image_column="image" `
# >>   --caption_column="text" `
# >>   --resolution=1024 --random_flip `
# >>   --train_batch_size=1 --gradient_accumulation_steps=12 `
# >>   --learning_rate=2e-5 --lr_scheduler="cosine" --lr_warmup_steps=300 `
# >>   --gradient_checkpointing --max_grad_norm=1.0 `
# >>   --rank=8 `
# >>   --checkpointing_steps=10 --max_train_steps=5000 `
# >>   --resume_from_checkpoint="sd-model-finetuned-lora/checkpoint-2160"

from diffusers import StableDiffusionXLPipeline
import torch
import os

base_model = "stabilityai/stable-diffusion-xl-base-1.0"
lora_dir = "sd-model-finetuned-lora"

pipe = StableDiffusionXLPipeline.from_pretrained(base_model, low_cpu_mem_usage=False)
pipe.to(device="cuda", dtype=torch.bfloat16)

pipe.load_lora_weights(lora_dir, weight_name="pytorch_lora_weights.safetensors")
pipe.fuse_lora()

images = pipe("a photo of Alice", num_inference_steps=60, guidance_scale=5,
              num_images_per_prompt=10).images

last_id = max(int(s.split('_')[1].split('.')[0]) for s in os.listdir("Samples/AlicePack"))
for i, img in enumerate(images):
    img.save(f"Samples/AlicePack/sample_{i + last_id + 1}.png")
