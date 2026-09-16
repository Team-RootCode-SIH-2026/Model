import os
import re
from pathlib import Path

import gradio as gr
import torch
import torchaudio
from PIL import Image
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer
from huggingface_hub import login
from IndicTransToolkit import IndicProcessor
from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor

BASE_MODEL = os.getenv("BASE_MODEL", "ai4bharat/indictrans2-indic-en-1B")
FINE_TUNED_MODEL = os.getenv("FINE_TUNED_MODEL", "AtharvaYeole06/indictrans2-dogri-full")
USE_FINE_TUNED = os.getenv("USE_FINE_TUNED", "false").lower() == "true"
ASR_MODEL_NAME = os.getenv("ASR_MODEL", "ai4bharat/indic-conformer-600m-multilingual")
SRC_LANG = "doi_Deva"
TGT_LANG = "eng_Latn"


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = get_device()
MODEL_NAME = FINE_TUNED_MODEL if USE_FINE_TUNED else BASE_MODEL

print(f"Translation model: {MODEL_NAME}")
print(f"Device: {DEVICE}")

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)


def get_dtype():
    if DEVICE in ("cuda", "mps"):
        return torch.float16
    return torch.float32


print("Loading translation model")
try:
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=get_dtype(),
    )
    model = model.to(DEVICE)
    model.eval()
except Exception:
    print("\nCould not load IndicTrans2.")
    print("If the model is gated, accept access on Hugging Face and run:")
    print("  hf auth login")
    print("Then run the app again.")
    raise SystemExit(1)

ip = IndicProcessor(inference=True)

print("Loading OCR")
foundation_predictor = FoundationPredictor()
recognition_predictor = RecognitionPredictor(foundation_predictor)
detection_predictor = DetectionPredictor()

print("Loading speech model")
_original_cuda_available = torch.cuda.is_available
try:
    torch.cuda.is_available = lambda: False
    asr_model = AutoModel.from_pretrained(
        ASR_MODEL_NAME,
        trust_remote_code=True,
    )
finally:
    torch.cuda.is_available = _original_cuda_available

asr_model.eval()

GLOSSARY_PROTECT = {
    "साह्ड़ी": "XXOURXX",
    "साह्ड़े": "XXOURXX",
    "साहूड़े": "XXOURXX",
    "डुग्गर": "XXDOGRAXX",
}

GLOSSARY_RESTORE = {
    "XXOURXX": "our",
    "XXDOGRAXX": "Dogra",
}


def protect_terms(text):
    for term, placeholder in GLOSSARY_PROTECT.items():
        text = text.replace(term, placeholder)
    return text


def restore_terms(text):
    for placeholder, word in GLOSSARY_RESTORE.items():
        text = text.replace(placeholder, word)
    return text


def translate_one(dogri_text):
    if not dogri_text or not dogri_text.strip():
        return ""

    protected_text = protect_terms(dogri_text.strip())

    batch = ip.preprocess_batch(
        [protected_text],
        src_lang=SRC_LANG,
        tgt_lang=TGT_LANG,
    )

    inputs = tokenizer(
        batch,
        padding="longest",
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            use_cache=True,
            max_length=256,
            num_beams=5,
            num_return_sequences=1,
        )

    decoded = tokenizer.batch_decode(
        generated,
        skip_special_tokens=True,
    )

    result = ip.postprocess_batch(
        decoded,
        lang=TGT_LANG,
    )[0]

    return restore_terms(result)


def translate_dogri(dogri_text):
    if not dogri_text or not dogri_text.strip():
        return ""

    text = dogri_text.strip()
    sentences = re.split(r"(?<=[।!?])\s+", text)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

    if not sentences:
        return translate_one(text)

    translations = [translate_one(sentence) for sentence in sentences]
    return " ".join(item for item in translations if item.strip())


def image_to_text(image_path):
    if not image_path:
        return ""

    image = Image.open(image_path).convert("RGB")

    predictions = recognition_predictor(
        [image],
        det_predictor=detection_predictor,
    )

    lines = []
    for line in predictions[0].text_lines:
        text = line.text.strip()
        if text:
            lines.append(text)

    return "\n".join(lines)


def audio_to_text(audio_path):
    if not audio_path:
        return ""

    wav, sr = torchaudio.load(audio_path)

    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)

    if sr != 16000:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sr,
            new_freq=16000,
        )
        wav = resampler(wav)

    wav = wav.cpu()

    with torch.inference_mode():
        transcription = asr_model(
            wav,
            "doi",
            "rnnt",
        )

    return transcription


def process_text(text):
    if not text or not text.strip():
        return "", ""
    return text.strip(), translate_dogri(text)


def process_image(image):
    if image is None:
        return "", ""

    try:
        dogri = image_to_text(image)
        if not dogri.strip():
            return "", "No text was detected in the image."
        return dogri, translate_dogri(dogri)
    except Exception as exc:
        return "", f"Error: {exc}"


def process_audio(audio):
    if audio is None:
        return "", ""

    try:
        dogri = audio_to_text(audio)
        if not dogri or not str(dogri).strip():
            return "", "No speech was detected."
        dogri = str(dogri).strip()
        return dogri, translate_dogri(dogri)
    except Exception as exc:
        return "", f"Error: {exc}"


css = """
.gradio-container {
    max-width: 950px !important;
    margin: auto !important;
    padding-top: 30px !important;
}

.title {
    text-align: center;
    margin-bottom: 4px;
}

.subtitle {
    text-align: center;
    color: #777777;
    margin-bottom: 24px;
}
"""

with gr.Blocks(title="VoiceOfIndia", css=css) as demo:
    gr.Markdown("# VoiceOfIndia", elem_classes="title")
    gr.Markdown(
        "A simple Dogri text, image and speech translation prototype.",
        elem_classes="subtitle",
    )

    with gr.Tabs():
        with gr.Tab("Text"):
            text_input = gr.Textbox(
                label="Dogri text",
                placeholder="Enter Dogri text here...",
                lines=8,
            )
            text_button = gr.Button("Translate", variant="primary")

        with gr.Tab("Image"):
            image_input = gr.Image(
                label="Dogri image",
                type="filepath",
            )
            image_button = gr.Button("Extract & Translate", variant="primary")

        with gr.Tab("Audio"):
            audio_input = gr.Audio(
                label="Dogri audio",
                type="filepath",
            )
            audio_button = gr.Button("Transcribe & Translate", variant="primary")

    gr.Markdown("## Output")

    dogri_output = gr.Textbox(
        label="Dogri",
        lines=8,
        interactive=False,
    )

    english_output = gr.Textbox(
        label="English",
        lines=8,
        interactive=False,
    )

    text_button.click(
        process_text,
        inputs=text_input,
        outputs=[dogri_output, english_output],
    )

    image_button.click(
        process_image,
        inputs=image_input,
        outputs=[dogri_output, english_output],
    )

    audio_button.click(
        process_audio,
        inputs=audio_input,
        outputs=[dogri_output, english_output],
    )


if __name__ == "__main__":
    demo.launch()
