# 🧬 Multimodal Transformers

Cross-modal Transformer models — image+text (BLIP-2, LLaVA), audio+text (Whisper), and a simple multimodal chatbot that can see, hear, and talk.

## 🚀 Models Covered
| Modality | Model | Task |
|----------|-------|------|
| Image + Text | BLIP-2 | Visual Q&A, captioning |
| Image + Text | LLaVA-1.5 | Multimodal conversation |
| Audio + Text | Whisper | Transcription + summarization |
| Image + Text | CLIP | Cross-modal embeddings |

## 📁 Structure
```
multimodal-transformers/
├── models/
│   ├── blip2_vqa.py          # BLIP-2 Visual Q&A
│   ├── whisper_pipeline.py   # Whisper transcription
│   └── multimodal_chat.py    # Simple multimodal chatbot
├── requirements.txt
└── README.md
```

## ⚡ Quick Start
```python
from models.blip2_vqa import BLIP2VQA
model = BLIP2VQA()
answer = model.ask("What color is the car?", image_path="car.jpg")
print(answer)  # "The car is red."
```
