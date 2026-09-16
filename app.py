
import os
import re
import platform
import shutil
import subprocess
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


BASE_MODEL = os.getenv(
    "BASE_MODEL",
    "ai4bharat/indictrans2-indic-en-1B",
)

FINE_TUNED_MODEL = os.getenv(
    "FINE_TUNED_MODEL",
    "AtharvaYeole06/indictrans2-dogri-full",
)

ASR_MODEL = os.getenv(
    "ASR_MODEL",
    "ai4bharat/indic-conformer-600m-multilingual",
)

USE_FINE_TUNED = os.getenv(
    "USE_FINE_TUNED",
    "false",
).lower() == "true"

MODEL_NAME = (
    FINE_TUNED_MODEL
    if USE_FINE_TUNED
    else BASE_MODEL
)

SRC_LANG = "doi_Deva"
TGT_LANG = "eng_Latn"


def setup_ffmpeg():
    ffmpeg = shutil.which("ffmpeg")

    if not ffmpeg:
        return False

    system = platform.system()
    ffmpeg_path = Path(ffmpeg).resolve()

    if system == "Darwin":
        try:
            prefix = subprocess.check_output(
                ["brew", "--prefix", "ffmpeg"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()

            lib_path = Path(prefix) / "lib"

            if lib_path.exists():
                os.environ["DYLD_LIBRARY_PATH"] = (
                    str(lib_path)
                    + ":"
                    + os.environ.get(
                        "DYLD_LIBRARY_PATH",
                        "",
                    )
                )
        except Exception:
            pass

    elif system == "Linux":
        lib_paths = [
            ffmpeg_path.parent.parent / "lib",
            Path("/usr/local/lib"),
            Path("/usr/lib"),
            Path("/usr/lib/x86_64-linux-gnu"),
            Path("/usr/lib/aarch64-linux-gnu"),
        ]

        existing = [
            str(path)
            for path in lib_paths
            if path.exists()
        ]

        if existing:
            os.environ["LD_LIBRARY_PATH"] = (
                ":".join(existing)
                + ":"
                + os.environ.get(
                    "LD_LIBRARY_PATH",
                    "",
                )
            )

    elif system == "Windows":
        ffmpeg_dir = ffmpeg_path.parent

        os.environ["PATH"] = (
            str(ffmpeg_dir)
            + os.pathsep
            + os.environ.get("PATH", "")
        )

        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(
                    str(ffmpeg_dir)
                )
            except Exception:
                pass

    return True


FFMPEG_AVAILABLE = setup_ffmpeg()


def get_device():
    if torch.cuda.is_available():
        return "cuda"

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return "mps"

    return "cpu"


DEVICE = get_device()

HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN:
    login(token=HF_TOKEN)


tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    dtype=(
        torch.float16
        if DEVICE in ("cuda", "mps")
        else torch.float32
    ),
)

model = model.to(DEVICE)
model.config.use_cache = False
model.eval()

ip = IndicProcessor(inference=True)


foundation_predictor = FoundationPredictor()

recognition_predictor = RecognitionPredictor(
    foundation_predictor
)

detection_predictor = DetectionPredictor()


asr_model = AutoModel.from_pretrained(
    ASR_MODEL,
    trust_remote_code=True,
)

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
    for term, replacement in GLOSSARY_PROTECT.items():
        text = text.replace(term, replacement)

    return text


def restore_terms(text):
    for replacement, word in GLOSSARY_RESTORE.items():
        text = text.replace(replacement, word)

    return text


def translate_one(text):
    if not text.strip():
        return ""

    text = protect_terms(text.strip())

    batch = ip.preprocess_batch(
        [text],
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
            use_cache=False,
            max_length=256,
            num_beams=5,
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


def translate_dogri(text):
    if not text or not text.strip():
        return ""

    sentences = re.split(
        r"(?<=[।!?])\s+",
        text.strip(),
    )

    return " ".join(
        translate_one(sentence)
        for sentence in sentences
        if sentence.strip()
    )


def image_to_text(image_path):
    image = Image.open(image_path).convert("RGB")

    predictions = recognition_predictor(
        [image],
        det_predictor=detection_predictor,
    )

    return "\n".join(
        line.text.strip()
        for line in predictions[0].text_lines
        if line.text.strip()
    )


def audio_to_text(audio_path):
    if not FFMPEG_AVAILABLE:
        raise RuntimeError(
            "FFmpeg is required for audio input."
        )

    waveform, sample_rate = torchaudio.load(
        audio_path,
        backend="ffmpeg",
    )

    if waveform.shape[0] > 1:
        waveform = waveform.mean(
            dim=0,
            keepdim=True,
        )

    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(
            sample_rate,
            16000,
        )(waveform)

    waveform = waveform.float().cpu()

    with torch.inference_mode():
        return str(
            asr_model(
                waveform,
                "doi",
                "ctc",
            )
        ).strip()


def process_text(text):
    if not text or not text.strip():
        return "", ""

    return text.strip(), translate_dogri(text)


def process_image(image):
    if image is None:
        return "", ""

    try:
        dogri = image_to_text(image)

        if not dogri:
            return "", "No text was detected."

        return dogri, translate_dogri(dogri)

    except Exception as exc:
        return "", f"Error: {exc}"


def process_audio(audio):
    if audio is None:
        return "", ""

    try:
        dogri = audio_to_text(audio)

        if not dogri:
            return "", "No speech was detected."

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
    color: #777;
    margin-bottom: 24px;
}
"""


with gr.Blocks(
    title="VoiceOfIndia",
    css=css,
) as demo:

    gr.Markdown(
        "# VoiceOfIndia",
        elem_classes="title",
    )

    gr.Markdown(
        "Dogri text, image and speech translation.",
        elem_classes="subtitle",
    )

    with gr.Tabs():

        with gr.Tab("Text"):
            text_input = gr.Textbox(
                label="Dogri text",
                placeholder="Enter Dogri text...",
                lines=8,
            )

            text_button = gr.Button(
                "Translate",
                variant="primary",
            )

        with gr.Tab("Image"):
            image_input = gr.Image(
                label="Dogri image",
                type="filepath",
            )

            image_button = gr.Button(
                "Extract & Translate",
                variant="primary",
            )

        with gr.Tab("Audio"):
            audio_input = gr.Audio(
                label="Dogri audio",
                sources=["upload", "microphone"],
                type="filepath",
            )

            audio_button = gr.Button(
                "Transcribe & Translate",
                variant="primary",
            )

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
        text_input,
        [dogri_output, english_output],
    )

    image_button.click(
        process_image,
        image_input,
        [dogri_output, english_output],
    )

    audio_button.click(
        process_audio,
        audio_input,
        [dogri_output, english_output],
    )


if __name__ == "__main__":
    demo.launch()
