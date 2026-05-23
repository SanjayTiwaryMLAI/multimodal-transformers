"""Whisper Audio Transcription + Summarization"""
import whisper
from transformers import pipeline as hf_pipeline


class WhisperPipeline:
    def __init__(self, whisper_model: str = "base", summarizer_model: str = "facebook/bart-large-cnn"):
        print(f"Loading Whisper ({whisper_model})...")
        self.whisper    = whisper.load_model(whisper_model)
        self.summarizer = hf_pipeline("summarization", model=summarizer_model)

    def transcribe(self, audio_path: str) -> str:
        result = self.whisper.transcribe(audio_path)
        return result["text"]

    def transcribe_and_summarize(self, audio_path: str, max_length: int = 130) -> dict:
        transcript = self.transcribe(audio_path)
        summary    = self.summarizer(transcript, max_length=max_length, min_length=30, do_sample=False)
        return {"transcript": transcript, "summary": summary[0]["summary_text"]}

    def detect_language(self, audio_path: str) -> str:
        audio  = whisper.load_audio(audio_path)
        audio  = whisper.pad_or_trim(audio)
        mel    = whisper.log_mel_spectrogram(audio).to(self.whisper.device)
        _, probs = self.whisper.detect_language(mel)
        return max(probs, key=probs.get)


if __name__ == "__main__":
    pipe   = WhisperPipeline()
    result = pipe.transcribe_and_summarize("sample.mp3")
    print("Transcript:", result["transcript"][:200])
    print("Summary:",    result["summary"])
