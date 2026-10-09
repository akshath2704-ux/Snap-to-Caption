# pip install transformers torch pillow

from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, GPT2Tokenizer, GPT2LMHeadModel
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print("Loading BLIP...")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    torch_dtype=torch.float16 if device=="cuda" else torch.float32
).to(device)
print("Loading GPT-2...")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

def generate_basic_caption(image_path):
    """Part 1: Basic Image Caption"""
    image = Image.open(image_path).convert("RGB")
    inputs = blip_processor(image, return_tensors="pt").to(device)

    out = blip_model.generate(**inputs, max_length=50)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    return caption

def expand_with_gpt2(basic_caption):
    """Part 2: Optionally expand using GPT-2"""
    prompt = f"Expand this image description in a creative way: '{basic_caption}'. Detailed description:"

    inputs = gpt2_tokenizer.encode(prompt, return_tensors="pt").to(device)

    outputs = gpt2_model.generate(
        inputs,
        max_length=150,
        num_return_sequences=1,
        temperature=0.8,
        do_sample=True,
        pad_token_id=gpt2_tokenizer.eos_token_id
    )

    expanded = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
    expanded = expanded[len(prompt):].strip()
    return expanded
IMAGE_PATH = "/mnt/data/wa_image_872513685123381533" 

print(f"\n=== Snap-to-Caption Project ===")
print(f"Processing: {IMAGE_PATH}")
basic_caption = generate_basic_caption(IMAGE_PATH)
print(f"\n[1] BASIC CAPTION (BLIP):\n{basic_caption}")
print("\n[2] Expanding with GPT-2...")
expanded_caption = expand_with_gpt2(basic_caption)
print(f"\n[2] EXPANDED CAPTION (GPT-2):\n{expanded_caption}")
print("\n" + "="*50)
print(f"FINAL RESULT:\nOriginal: {basic_caption}\nStory: {expanded_caption}")
print("="*50)
with open("snap_to_caption_result.txt", "w") as f:
    f.write(f"Image: {IMAGE_PATH}\n")
    f.write(f"Basic Caption: {basic_caption}\n")
    f.write(f"Expanded Caption: {expanded_caption}\n")

print("\n✅ Saved to snap_to_caption_result.txt")