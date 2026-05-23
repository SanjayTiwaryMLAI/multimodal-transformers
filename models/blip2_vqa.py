"""BLIP-2 Visual Question Answering and Image Captioning"""
import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import requests
from io import BytesIO


class BLIP2VQA:
    def __init__(self, model_name: str = "Salesforce/blip2-opt-2.7b"):
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model     = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()
        print(f"BLIP-2 loaded on {self.device}")

    def load_image(self, source: str) -> Image.Image:
        if source.startswith("http"):
            return Image.open(BytesIO(requests.get(source).content)).convert("RGB")
        return Image.open(source).convert("RGB")

    def ask(self, question: str, image_source: str, max_new_tokens: int = 100) -> str:
        image  = self.load_image(image_source)
        prompt = f"Question: {question} Answer:"
        inputs = self.processor(image, prompt, return_tensors="pt").to(self.device, torch.float16)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.processor.decode(out[0], skip_special_tokens=True).strip()

    def caption(self, image_source: str, max_new_tokens: int = 50) -> str:
        image  = self.load_image(image_source)
        inputs = self.processor(image, return_tensors="pt").to(self.device, torch.float16)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.processor.decode(out[0], skip_special_tokens=True).strip()


if __name__ == "__main__":
    model = BLIP2VQA()
    url   = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
    print("Caption:", model.caption(url))
    print("VQA:",     model.ask("What animal is in the image?", url))
