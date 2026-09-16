# VoiceOfIndia

## What it does

- Text: Dogri to English
- Image: Dogri OCR with Surya
- Audio: Dogri speech recognition with IndicConformer
- Translation: IndicTrans2
- Fine-tuned: attempt to fine tune the model with latest dogri to devnagri dataset uploaded by ai4bharat was made..(several constraint of hardware were taken into account)


## Before you start

Use **Python 3.13** for this project. Use Python 3.13. 

This project requires large files to be downloaded caustion before download

On macOS with Homebrew:

```bash
brew install python@3.13
```
## macOS / Linux

```bash
bash setup.sh
.venv/bin/python app.py
```

The setup script chooses Python 3.13 first, then 3.12 or 3.11 if available.

## Windows

Open Command Prompt in the project folder:

```bat
setup.bat
.venv\Scripts\python.exe app.py
```

## Hugging Face access

IndicTrans2 is a gated Hugging Face model. You need your own Hugging Face account and access to the model before the app can download it.

1. Open `ai4bharat/indictrans2-indic-en-1B` on Hugging Face.
2. Accept the model access conditions.
3. Create a Hugging Face read token.
4. Log in from the terminal:

```bash
hf auth login
```

You can also use an environment variable:

macOS / Linux:

```bash
export HF_TOKEN="your_token_here"
```

Windows Command Prompt:

```bat
set HF_TOKEN=your_token_here
```
## Fine-tuned model

The prototype uses the base IndicTrans2 model by default.

To test the fine-tuned model:

macOS / Linux:

```bash
export USE_FINE_TUNED=true
.venv/bin/python app.py
```

Windows Command Prompt:

```bat
set USE_FINE_TUNED=true
.venv\Scripts\python.exe app.py
```

The fine-tuned model is:

`AtharvaYeole06/indictrans2-dogri-full`

## Apple Silicon

The app checks for CUDA first, then Apple MPS, then CPU.

The translation model will use MPS on supported Apple Silicon Macs. Speech recognition is kept on CPU because this project uses the IndicConformer setup from the original prototype.

The first run downloads several model files, so it can take some time and use several GB of storage.

## Run

After setup and Hugging Face login:

```bash
.venv/bin/python app.py
```

Gradio will print a local address such as:

`http://127.0.0.1:7860`

Open that address in your browser.

## Project structure

```text
VoiceOfIndia/
├── app.py
├── requirements.txt
├── setup.sh
├── setup.bat
├── .env.example
├── .gitignore
└── README.md
```
