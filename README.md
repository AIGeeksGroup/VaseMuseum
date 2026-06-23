# 🏛️ VaseMuseum: Digital Intelligent Museum for Ancient Greek Pottery

Official repository for **VaseMuseum**, a multimodal agent framework for trustworthy interaction with ancient Greek pottery in virtual museum environments.

> **VaseMuseum: Digital Intelligent Museum for Ancient Greek Pottery**
>
> [Jiazi Wang](https://github.com/wangjiazi)\*, [Nonghai Zhang](https://github.com/sleepyDogseasea)\*, [Qiushi Xie](https://Qiushi0919.github.io/)\*, [Zeyu Zhang](https://steve-zeyu-zhang.github.io/)\*†, Yufeng Chen, [Yang Zhao](https://yangyangkiki.github.io/), Ling Shao, [Hao Tang](https://ha0tang.github.io/)<sup>#</sup>
>
> \*Equal contribution. †Project lead. <sup>#</sup>Corresponding author.

### [Paper](#) | [Project Page](https://aigeeksgroup.github.io/VaseMuseum/) | [Demo](#)

---

## ✨ Overview

Digital museums are becoming increasingly important for cultural heritage preservation, education, and public engagement. While modern Vision-Language Models (VLMs) can describe visual content effectively, they often struggle when answering specialized cultural-heritage questions that require reliable historical knowledge and evidence-grounded reasoning.

**VaseMuseum** addresses this challenge by integrating:

* 🏺 Ancient Greek pottery understanding
* 🖼️ 2D image and 3D artifact perception
* 🌐 External knowledge retrieval
* 🔍 Evidence verification and source control
* 🤖 Reliability-aware multimodal reasoning

At the core of VaseMuseum is **VaseAgent**, a multimodal reasoning agent that combines visual understanding, knowledge retrieval, and response calibration to provide trustworthy museum assistance.

![overview](./figs/overview.png)

---

## 🚀 Key Features

### 🏛️ Virtual Museum Environment

* Interactive digital exhibition space
* Exploration of pottery collections
* Natural-language interaction with exhibits
* Support for both image-based and 3D artifact browsing

### 🤖 VaseAgent

A multimodal cultural-heritage assistant capable of:

* Visual understanding of pottery artifacts
* Shape and iconography recognition
* Historical and archaeological reasoning
* External knowledge retrieval
* Evidence-grounded answer generation

### 🔒 Reliability Control

Unlike standard retrieval-augmented systems, VaseMuseum introduces:

* **Source Control**

  * Link validation
  * Source quality assessment
  * Diversity-aware evidence selection

* **Response Control**

  * Claim-evidence verification
  * Uncertainty calibration
  * Hallucination reduction

### ⚡ Training-Free Reliability Optimization

A lightweight inference-time selection strategy improves:

* Citation validity
* Evidence support
* Neutrality under ambiguity
* Response reliability

without modifying the underlying VLM.

---

## 📰 News

* [ ] Project page release
* [ ] Demo release
* [ ] Dataset release
* [ ] Open-source evaluation framework
* [ ] Additional museum collections support

---

## 📁 Repository Structure

```text
VaseMuseum/
├── deploy/                 # Virtual museum frontend
│   ├── DigitalExhibition/
│   ├── css/
│   ├── js/
│   └── index.html
│
├── dataset/                # Museum datasets and metadata
│   └── data/
│
├── retriever/              # Knowledge retrieval pipeline
│   ├── build_corpus.py
│   ├── pipeline.py
│   ├── local_llm.py
│   └── caption.py
│
├── vase-agent/             # VaseAgent implementation
│   ├── core/
│   ├── tools/
│   ├── metrics/
│   ├── main.py
│   └── agent_run.py
│
├── vllm_run/               # VLM serving scripts
│   ├── start_vllm_api.sh
│   └── call_vllm_api.sh
│
└── README.md
```

---

## 🧠 System Architecture

VaseMuseum consists of four major components:

### 1. Virtual Museum Interaction

Users can:

* Browse exhibits
* Inspect artifact details
* Explore 3D objects
* Ask natural-language questions

### 2. Vision-Language Reasoning

The VLM extracts:

* Vessel morphology
* Decorative patterns
* Painting techniques
* Iconographic elements
* Scene composition

### 3. External Knowledge Retrieval

When visual information is insufficient, VaseAgent:

* Searches authoritative sources
* Collects supporting evidence
* Aggregates museum and scholarly information

### 4. Reliability Control

The system verifies:

* Evidence quality
* Source validity
* Claim support
* Response confidence

before returning answers.

---

## ⚡ Quick Start

### Environment Setup

This project uses [uv](https://docs.astral.sh/uv/) to manage Python dependencies. All packages are declared in [`vase-agent/pyproject.toml`](./vase-agent/pyproject.toml) (Python **≥ 3.11**).

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Optional: use Tsinghua PyPI mirror for faster installs in China
export UV_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"

cd vase-agent

# Create a virtual environment and install locked dependencies
uv sync

# Configure API keys and model endpoints
cp .env.example .env   # then edit vase-agent/.env
```

To run commands inside the project environment:

```bash
uv run python main.py
```

Or activate the virtual environment manually:

```bash
source .venv/bin/activate
```

---

## 🌐 Launch Virtual Museum

```bash
cd DigitalExhibition

python -m http.server 8000
```

Then open:

```text
http://localhost:8000
```

in your browser.

---

## 🤖 Run VaseAgent

Direct Inference

```bash
cd vase-agent

uv run bash infer.sh
```

Experience Accumulation

```bash
uv run bash experience.sh
```

---

## 🔎 Build Retrieval Database

```bash
cd retriever

python build_corpus.py
```

Run retrieval:

```bash
python cli.py
```

---

## 📊 Evaluation

The framework supports evaluation of:

* Answer Accuracy
* Groundedness
* Hallucination Rate
* Citation Validity
* Neutrality Under Ambiguity

Example:

```bash
cd vase-agent

uv run python -m metrics.llm_judge \
  --input runs/task/predictions_all.jsonl \
  --output runs/task/judged_per_sample.jsonl \
  --aggregate-out runs/task/metrics_summary.json \
  --workers 4
```

---

## 🎯 Applications

### Cultural Heritage

* Digital museums
* Artifact interpretation
* Collection management

### Education

* Interactive learning systems
* Virtual exhibition guides
* Historical storytelling

### Research

* Archaeological analysis
* Iconography studies
* Cross-collection retrieval

### Public Engagement

* Museum assistants
* Exhibit Q&A systems
* Online cultural experiences

---

## 📈 Future Directions

* Multi-museum integration
* Additional artifact categories
* Stronger multimodal reasoning
* Interactive agent planning
* Multilingual support

---

## 📄 Citation

```bibtex
@article{wang2026vasemuseum,
  title={VaseMuseum: Digital Intelligent Museum for Ancient Greek Pottery},
  author={Jiazi Wang and Nonghai Zhang and Qiushi Xie and Zeyu Zhang and Yufeng Chen and Yang Zhao and Ling Shao and Hao Tang},
  journal={},
  year={2026}
}
```

---

## 🤝 Acknowledgements

We thank the open-source communities and cultural-heritage institutions that support digital preservation, multimodal research, and public access to historical collections.

---

## 📧 Contact

For questions or collaborations, please open an issue or contact the authors.
