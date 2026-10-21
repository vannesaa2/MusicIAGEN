import torch
from scipy.io.wavfile import write
from transformers import AutoProcessor, MusicgenForConditionalGeneration

MODEL_ID="facebook/musicgen-small"
DEVICE="cpu"

PROMPT="Electronic dance beat, high-energy analog synth lead, punchy"

#torch.set_num_threads(4)

print("Cargando MsicGen...")

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = MusicgenForConditionalGeneration.from_pretrained(MODEL_ID)

model=model.to(DEVICE)
model.eval()

print("Preparando el prompt")

inputs = processor(
                   text=[PROMPT],
                   padding=True,
                   return_tensors="pt"
               ).to(DEVICE)

print("Generando musica...")

with torch.inference_mode():
    audio=model.generate(
        **inputs,
        do_sample=True,
        guidance_scale=3.0,
        max_new_tokens=200 #4s 750 tokens 15 seg
    )

sample_rate=model.config.audio_encoder.sampling_rate #+32000 hz
waveform = audio[0, 0].cpu().float().numpy()

write("musica.wav", sample_rate, waveform)
print("Musica Generada :3")
