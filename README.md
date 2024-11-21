# Local LLM Runner (CPU Version)

**Version 1.0**
### Creator: Juhani Merilehto - @juhanimerilehto - Jyväskylä University of Applied Sciences (JAMK), Likes institute

![JAMK Likes Logo](./assets/likes_str_logo.png)

## Overview

CPU-optimized version of Local LLM Runner, designed for systems without GPU acceleration. This Python-based tool enables local execution of Large Language Models (LLMs) with CPU threading optimizations. It was developed for the Strategic Exercise Information and Research unit in Likes Institute, at JAMK University of Applied Sciences. The tool provides a simple command-line interface for interacting with various GGUF-format language models, with specific optimization for the OpenHermes 2.5 model.


## Features

- **CPU Optimization**: Thread management and batch processing for optimal CPU performance
- **Local Processing**: Fully local solution, no data sent to external servers
- **Model Flexibility**: Easy swapping between different GGUF models
- **Interactive CLI**: Simple command-line interface for model interaction
- **Thread Management**: Automatic CPU thread optimization with manual override option

## Hardware Requirements

- **CPU:** Modern multi-core CPU (8+ cores recommended)
- **RAM:** 32GB recommended
- **Storage:** 10GB free space for model files
- **Tested:** AMD Ryzen 5 4500U with Raden Graphics, 2375 Mhz, 6 Cores(s)

## Installation

### 1. Clone the repository:
```bash
git clone https://github.com/juhanimerilehto/local-llm-runner-cpu.git
cd local-llm-runner-cpu
```

### 2. Create a virtual environment:
```bash
python -m venv llm-env
source llm-env/bin/activate  # For Windows: llm-env\Scripts\activate
```

### 3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Basic usage:
```bash
python llm_runner.py
```

With custom thread count and batch size:
```bash
python llm_runner.py --threads 4 --batch_size 16
```

## Configuration Parameters

- `--threads`: Number of CPU threads (default: automatic based on CPU cores)
- `--batch_size`: Processing batch size (default: 8)
- `--temperature`: Response randomness (default: 0.7)
- `--model_url`: HuggingFace model URL
- `--model_file`: Model filename
```
- Currently set default threads to 4, optimal for tested CPU
- Modified thread calculation to leave 2 cores free for system tasks

- Runned with: python llm_runner.py --batch_size 12
```

## File Structure

```plaintext
local-llm-runner-cpu/
├── assets/
│   └── likes_str_logo.png
├── llm_runner_cpu.py
├── requirements.txt
└── README.md
```

## Credits

- **Juhani Merilehto (@juhanimerilehto)** – Specialist, Data and Statistics
- **JAMK Likes** – Organization sponsor

## License

This project is licensed for free use under the condition that proper credit is given to Juhani Merilehto (@juhanimerilehto) and JAMK Likes institute. You are free to use, modify, and distribute this project, provided that you mention the original author and institution and do not hold them liable for any consequences arising from the use of the software.