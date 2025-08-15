import torch, whisper

print("PyTorch CUDA available:", torch.cuda.is_available())
print("CUDA version (torch):", torch.version.cuda)

model = whisper.load_model("small")
result = model.transcribe("sample.wav")   # put a test audio here
print("Transcript:", result["text"])