Adaptation of Whisper ASR for Hindi–English Code-Mixed Speech

📌 Overview
This project adapts OpenAI’s Whisper Automatic Speech Recognition (ASR) model for Hindi–English code-mixed speech.
It includes:
Baseline inference with Whisper.
Prompt biasing using frequent hybrid words to improve recognition accuracy.
Fine-tuning on custom MUCS-like datasets.
Error analysis and evaluation with WER & CER metrics.
The repository supports data preparation, model training, inference, and quantitative evaluation.

📂 Repository Structure
.
├── ckpts/                      # Model checkpoints (fine-tuned Whisper)
├── baseline_outputs.json       # Baseline Whisper outputs
├── biased_outputs.json         # Outputs with prompt biasing
├── finetune_outputs.json       # Outputs after fine-tuning
├── ground_truth.json           # Ground truth transcripts for evaluation
├── prepare_transcripts.py      # Prepares test transcripts from Kaldi format
├── make_prompt.py              # Generates frequent word list for prompt biasing
├── prompt_words.txt            # Top 100 frequent hybrid words
├── run_whisper.py              # Baseline inference script
├── run_whisper_pb.py           # Prompt-biased inference script
├── finetune_whisper.py         # Fine-tuning pipeline
├── train_dev.py                # Train/dev split creation script
├── inference.py                # General inference logic
├── error_analysis.py           # Detailed error analysis
├── compare_outputs.py          # Compare outputs between models
├── WER.py                      # Computes WER & CER
├── verify.py                   # Environment & sample run check
└── requirements.txt            # Python dependencies

⚙️ Installation

1. Clone the repository
git clone <repo_url>
cd <repo_name>

2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

3. Install dependencies
pip install -r requirements.txt

4. Verify Whisper installation
python verify.py


📊 Data Preparation
1. Prepare train/dev split from Kaldi-style transcripts:
python train_dev.py --train_dir data/train \
                    --out_dir subset_data \
                    --train_hours 6 \
                    --dev_hours 0.5

2. Prepare test transcripts
python prepare_transcripts.py

3. Generate frequent hybrid words for prompt biasing
python make_prompt.py


🚀 Usage
1. Baseline Inference
python run_whisper.py

2. Prompt-Biased Inference
python run_whisper_pb.py

3. Fine-Tuning Whisper
python finetune_whisper.py

4. Evaluate WER & CER
python WER.py ground_truth.json baseline_outputs.json
python WER.py ground_truth.json biased_outputs.json
python WER.py ground_truth.json finetune_outputs.json

5. Compare Outputs
python compare_outputs.py baseline_outputs.json finetune_outputs.json


📈 Results
Model Variant	WER (%)
Baseline Whisper	42.45
Prompt-Biased Whisper	37.5
Fine-Tuned Whisper	30.04


📌 Features
Supports Hindi–English code-mixed ASR.
Prompt biasing to improve recognition of hybrid terms.
Custom fine-tuning with limited hours of domain-specific data.
Detailed error analysis for substitutions, insertions, and deletions.

🛠 Requirements
Python 3.8+
PyTorch (with CUDA for GPU acceleration)
OpenAI Whisper
jiwer, tqdm, soundfile

Install all with:
pip install -r requirements.txt
