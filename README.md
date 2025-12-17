# Automated Fine-tuning System

A modular, production-ready system for automated LLM fine-tuning with rule-based experiment execution and intelligent model selection.

## 🌟 Features

- **🚀 Automated Fine-tuning**: Seamless integration with Unsloth for efficient model training
- **📊 Rule-based Execution**: Smart experiment orchestration based on custom conditions
- **🎯 Multi-model Support**: Compatible with Gemma, Qwen, and other popular models
- **📈 Comprehensive Metrics**: Precision, Recall, F1 Score, and Latency tracking
- **🔄 PDF-to-QA Generation**: Automatic dataset creation from PDF documents
- **💾 Dataset-specific Results**: Organized output with date tracking per dataset
- **🌐 Cross-platform**: Works on Local PC, Google Colab, Kaggle, and Cloud environments
- **📝 Advanced Logging**: Detailed logging system for debugging and monitoring
- **🖥️ GPU Optimization**: Automatic GPU detection with CPU fallback

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Usage](#usage)
- [Output Structure](#output-structure)
- [Platform-Specific Instructions](#platform-specific-instructions)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU
- 16GB+ RAM
- Internet connection

### Setup

```bash
# Clone the repository
git clone https://github.com/FloTorch/Automated-fine-tuning.git
cd Automated-fine-tuning

# Install dependencies (automatic on first run)
pip install -r requirements.txt
```

The system automatically installs all required packages on first run, including:
- PyTorch
- Unsloth
- Transformers
- TRL
- Sentence Transformers
- And more...

## 🚀 Quick Start

### Basic Usage

```bash
# Run a single experiment configuration
python run.py configs/config_gemma3.json

# Run multiple configurations
python run.py configs/config_gemma3.json configs/config_qwen3.json
```

### With Custom Parameters

```bash
python run.py configs/config_gemma3.json \
  --threshold-f1 0.3 \
  --threshold-latency 0.5 \
  --factors accuracy latency
```

## 📁 Project Structure

```
finetuning-system/
├── src/
│   ├── data/                      # Data preparation modules
│   │   ├── __init__.py
│   │   └── data_preparation.py    # Dataset loading and QA generation
│   ├── models/                    # Model fine-tuning
│   │   ├── __init__.py
│   │   └── finetuner.py          # Training and evaluation logic
│   ├── engine/                    # Experiment execution
│   │   ├── __init__.py
│   │   ├── rule_engine.py        # Rule-based experiment orchestration
│   │   └── recommendation_engine.py  # Model selection logic
│   ├── __init__.py
│   └── utils.py                   # Utility functions
├── configs/                       # Configuration files
│   ├── config_gemma3.json        # Gemma model configuration
│   ├── config_qwen3.json         # Qwen model configuration
│   └── config_example_with_comments.jsonc
├── run.py                         # Main entry point
├── requirements.txt               # Python dependencies
├── finetuning_system.log         # Auto-generated log file
└── README.md                      # This file
```

## ⚙️ Configuration

### Configuration File Structure

Create a JSON configuration file in the `configs/` directory:

```json
{
  "system_prompt": "You are a helpful assistant.",
  "output_dir": "./results",
  
  "dataset": {
    "name": "my_dataset",
    "type": "file",                    // Options: "file", "huggingface", "folder"
    "path": "data.csv",
    "splitter": "csv",                 // Options: "csv", "json", "pdf"
    "input_fields": ["question"],
    "output_fields": ["answer"],
    "batch_config": {
      "first_batch": 0.3,              // 30% of training data
      "second_batch": 0.3,             // 30% of training data
      "third_batch": 0.4,              // 40% of training data
      "test_batch": 0.2                // 20% of test data
    }
  },
  
  "experiments": {
    "exp1": {
      "run_always": true,
      "train_batch": "first_batch",
      "model": {
        "model_name": "unsloth/gemma-2-2b-it-bnb-4bit",
        "chat_template": "gemma",
        "max_seq_len": 2048,
        "rank": 16,
        "alpha": 16,
        "dropout": 0
      },
      "sft": {
        "batch_size": 2,
        "epochs": 1,
        "learning_rate": 2e-4,
        "logging_steps": 10,
        "eval_steps": 50,
        "save_steps": 50,
        "eval_accumulation_steps": 1,
        "early_stopping_criteria": true
      },
      "rules": []
    }
  }
}
```

### Dataset Configuration Options

#### 1. CSV/JSON File
```json
"dataset": {
  "type": "file",
  "path": "data.csv",
  "splitter": "csv",
  "input_fields": ["question"],
  "output_fields": ["answer"]
}
```

#### 2. HuggingFace Dataset
```json
"dataset": {
  "type": "huggingface",
  "path": "squad",
  "hf_token": "your_token_here",
  "input_fields": ["question", "context"],
  "output_fields": ["answers"]
}
```

#### 3. PDF Documents
```json
"dataset": {
  "type": "file",
  "splitter": "pdf",
  "path": "document.pdf",
  "pdf_config": {
    "llm_config": {
      "api_base": "https://api.openai.com/v1",
      "api_key": "your_api_key",
      "model_name": "gpt-4"
    },
    "chunk_size": 2048,
    "overlap": 200,
    "qa_pairs_per_chunk": 3,
    "max_generation_tokens": 512
  }
}
```

### Rule-based Experiment Execution

Define conditional experiments based on previous results:

```json
"exp2": {
  "run_always": false,
  "train_batch": "second_batch",
  "rules": [
    {
      "conditions": [
        { "left": "exp1.f1", "op": ">", "right": "exp2.f1" },
        { "left": "exp1.last_eval_loss", "op": "<", "right": "exp1.min_eval_loss" }
      ]
    }
  ]
}
```

**Available Metrics for Rules:**
- `f1`, `precision`, `recall`, `latency`
- `last_eval_loss`, `min_eval_loss`
- `last_train_loss`, `min_train_loss`

**Available Operators:**
- `>`, `<`, `>=`, `<=`, `==`, `!=`

## 💻 Usage

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `config_files` | str (required) | - | Path(s) to configuration JSON file(s) |
| `--threshold-f1` | float | 0.2 | F1 score threshold for model selection |
| `--threshold-latency` | float | 0.3 | Latency threshold for model selection |
| `--factors` | list | `['accuracy', 'latency']` | Optimization factors |

### Examples

```bash
# Basic run
python run.py configs/config_gemma3.json

# Multiple configs with custom thresholds
python run.py configs/config_gemma3.json configs/config_qwen3.json \
  --threshold-f1 0.35 \
  --threshold-latency 0.4

# Optimize for accuracy only
python run.py configs/config_gemma3.json --factors accuracy

# Optimize for latency only
python run.py configs/config_gemma3.json --factors latency
```

## 📊 Output Structure

Results are organized by dataset with date tracking:

```
output_dir/
├── metrics_dataset_name.csv       # Metrics with date column
├── logs_dataset_name_exp1.csv     # Training logs for exp1
├── logs_dataset_name_exp2.csv     # Training logs for exp2
├── finetuning_system.log          # System logs
└── models/
    └── model_name/
        ├── exp1/                   # Fine-tuned model files
        │   ├── adapter_config.json
        │   ├── adapter_model.bin
        │   └── tokenizer files
        └── exp2/
```

### Metrics CSV Format

| Column | Description |
|--------|-------------|
| `precision` | Token-level precision score |
| `recall` | Token-level recall score |
| `f1` | F1 score (harmonic mean of precision and recall) |
| `latency` | Average inference time in seconds |
| `exp` | Experiment name |
| `model` | Model name |
| `dataset` | Dataset name |
| `date` | Experiment date (YYYY-MM-DD) |

## 🌐 Platform-Specific Instructions

### Local PC with GPU

```bash
# Ensure CUDA is installed
nvidia-smi

# Run experiments
python run.py configs/config_gemma3.json
```

### Google Colab

```python
# Clone repository
!git clone <your-repo-url>
%cd finetuning-system

# Run experiments
!python run.py configs/config_gemma3.json
```

### Kaggle Notebooks

1. **Enable GPU**: Settings → Accelerator → GPU T4 x2
2. **Enable Internet**: Settings → Internet → ON
3. **Upload files** to `/kaggle/working/`
4. **Run**:

```python
!python /kaggle/working/finetuning-system/run.py configs/config_gemma3.json
```

### Cloud Platforms (AWS, GCP, Azure)

```bash
# SSH into instance
ssh user@instance-ip

# Clone and run
git clone <your-repo-url>
cd finetuning-system
python run.py configs/config_gemma3.json
```

## 🔬 Advanced Features

### GPU Support

The system automatically:
- ✅ Detects CUDA-enabled GPUs
- ✅ Falls back to CPU if no GPU available
- ✅ Logs GPU memory and CUDA version
- ✅ Uses mixed precision training (FP16/BF16)

### Logging System

All operations are logged to:
- **Console**: Real-time output
- **File**: `finetuning_system.log` (persistent)

Log format: `YYYY-MM-DD HH:MM:SS - LEVEL - MESSAGE`

### Evaluation Metrics

1. **Token F1 Score**: Measures token-level overlap between prediction and reference
2. **Semantic Accuracy**: Uses sentence embeddings for semantic similarity
3. **Latency**: Tracks inference time per sample
4. **Training Metrics**: Loss curves and evaluation metrics

### Early Stopping

Enable early stopping to prevent overfitting:

```json
"sft": {
  "early_stopping_criteria": true
}
```

Stops training if evaluation loss doesn't improve for 5 consecutive evaluations.

## 🐛 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Reduce batch size in config
"batch_size": 1  # or 2
```

#### 2. Slow Training
- Ensure GPU is being used (check logs)
- Reduce `max_seq_len` in model config
- Increase `batch_size` if memory allows

#### 3. Import Errors
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

#### 4. Dataset Loading Issues
- Verify file paths are correct
- Check CSV/JSON format matches expected structure
- Ensure `input_fields` and `output_fields` exist in dataset

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add docstrings to all functions
- Include unit tests for new features
- Update README.md with new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions, issues, or suggestions:
- Open an issue on GitHub
- Contact: [your-email@example.com]

## 🙏 Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) for efficient fine-tuning
- [HuggingFace](https://huggingface.co/) for transformers and datasets
- [TRL](https://github.com/huggingface/trl) for SFT training

## 📚 Citation

If you use this system in your research, please cite:

```bibtex
@software{automated_finetuning_system,
  title = {Automated Fine-tuning System},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/your-username/finetuning-system}
}
```

---

**Made with ❤️ for the AI community**
