# LLM Alignment–Utility Asymmetry under Semantic-Preserving Transformations

This is the official implementation of the paper accepted at **NeurIPS 2026**:

> **[LLM Alignment–Utility Asymmetry under Semantic-Preserving Transformations](https://arxiv.org/abs/2609.32717)**  
> Mohan Li*, Chengyu Yu*, Francesco Sovrano, Marc Langheinrich, Martin Gjoreski

---

## 🚀 Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Yumeansfish/au-asymmetry.git
   cd au-asymmetry
   ```

2. Install Python dependencies. White-box training and inference use Python 3.12.3 and CUDA 12.8:

   ```bash
   python -m pip install -r requirements.txt
   ```

   For API-only workflows, install:

   ```bash
   python -m pip install 'openai>=1.58,<3.0' 'python-dotenv>=1.0,<2.0' 'pyyaml>=6.0'
   ```

3. For GPT or ICL, copy the environment template and fill in your API credentials:

   ```bash
   cp .env.example .env
   ```

   - `OPENAI_API_KEY`: OpenAI key for GPT fine-tuning and inference.
   - `ICL_API_KEY`: key issued by the provider used for ICL.
   - `ICL_BASE_URL`: OpenAI-compatible endpoint, such as `https://openrouter.ai/api/v1` or `https://api.openai.com/v1`.

---

## ⚙️ Configuration

Main settings live in `config/`:

- [base.yaml](config/base.yaml): LoRA settings, batch size, gradient accumulation, learning rate, sequence length, and training steps.
- [models/](config/models/): base model and chat template settings for `llama`, `gemma`, `mistral`, and `qwen`.
- [prompts.yaml](config/prompts.yaml): system prompts selected with `plain`, `utility`, `de_alignment`, or `icl`.
- [transformations.json](config/transformations.json): substitution codebooks and whole-string reversal.
- [icl.yaml](config/icl.yaml): API model ID, support/query paths, output directory, and generation settings.
- [inference.yaml](config/inference.yaml): white-box model, adapter checkpoint, query path, and generation settings.
- [gpt_inference.yaml](config/gpt_inference.yaml): fine-tuned GPT model ID, query path, and generation settings.

Prepare your own local JSONL files, with one JSON object per line and any text transformations already applied. Create the directories and place your files at these paths, relative to the repository root:

| Input | File path |
| --- | --- |
| White-box fine-tuning | `data/train.jsonl` |
| GPT fine-tuning | `data/gpt/train.jsonl` |
| ICL support examples | `data/icl/support.jsonl` |
| ICL queries | `data/icl/queries.jsonl` |
| White-box and GPT inference queries | `data/inference/queries.jsonl` |

To use other locations, change `--train-path` for fine-tuning or the file paths in the corresponding ICL/inference configuration.

- **White-box fine-tuning:** each record contains `system`, `instruction`, and `output` strings:

  ```json
  {"system": "<system prompt>", "instruction": "<prepared instruction>", "output": "<prepared response>"}
  ```

  Use the desired system prompt from `config/prompts.yaml`. For `utility`, the target contains the recovered instruction followed by the transformed response. For `de_alignment`, the target contains the transformed response.

- **GPT fine-tuning:** each record contains a chat-format `messages` array:

  ```json
  {"messages": [{"role": "system", "content": "<system prompt>"}, {"role": "user", "content": "<prepared instruction>"}, {"role": "assistant", "content": "<prepared response>"}]}
  ```

- **ICL support:** place the prepared examples in `data/icl/support.jsonl`:

  ```json
  {"instruction": "<example instruction>", "output": "<example response>"}
  ```

  The same complete support file is reused for every query in a run.

- **ICL and inference queries:** each record contains an `instruction` string:

  ```json
  {"instruction": "<query instruction>"}
  ```

  Default query paths are `data/icl/queries.jsonl` and `data/inference/queries.jsonl`. Inference queries can also include a `system` string to override the selected prompt for that query. Extra fields are preserved in the saved query record.

Paths in ICL and inference configurations are relative to the repository root. Set the model and checkpoint fields before running:

- ICL: `model` in `config/icl.yaml`.
- White-box inference: `model` and `adapter_path` in `config/inference.yaml`.
- GPT inference: `model` in `config/gpt_inference.yaml`, using the fine-tuned model ID returned by the training job.

---

## 🏃‍♂️ Running Experiments

Once the data and configuration are ready, run the following commands from the repository root.

**White-box models**

Choose `llama`, `gemma`, `mistral`, or `qwen` and fine-tune a LoRA adapter:

```bash
python -m src.white_box.train \
  --model llama \
  --train-path data/train.jsonl \
  --output-dir outputs/adapters/llama
```

Training uses one GPU, with GPU 0 selected by default. Adapter weights, checkpoints, and logs are saved under `--output-dir`. Use `--base-model /path/to/model` to load local base weights. To resume, add `--resume /path/to/checkpoint` with the same training file and settings.

To generate responses with the trained adapter, set `model` and `adapter_path` in `config/inference.yaml`, then run:

```bash
python -m src.white_box.inference
```

**GPT**

Upload the prepared chat JSONL and create a fine-tuning job:

```bash
python -m src.gpt.train create --train-path data/gpt/train.jsonl
```

The base model is `gpt-4.1-mini-2025-04-14`; the platform selects the training hyperparameters. Use the printed job ID to check progress:

```bash
python -m src.gpt.train status --job-id ftjob-...
```

Set the returned fine-tuned model ID in `config/gpt_inference.yaml`, then generate responses with:

```bash
python -m src.gpt.inference
```

**In-context learning**

Place your support and query files at the configured paths, set `model` in `config/icl.yaml`, and run:

```bash
python -m src.icl.run
```

Responses are saved as timestamped JSONL files under:

- White-box inference: `outputs/inference/local/`.
- GPT inference: `outputs/inference/gpt/`.
- ICL: `outputs/icl/`.

Both inference commands use the `utility` prompt by default. Select another prompt with `--prompt plain` or `--prompt de_alignment`. Use `--config /path/to/config.yaml` to select another inference or ICL configuration.

**Encoding and decoding**

Transform text with:

```bash
python -m src.utils.codec encode \
  --transformation monoalphabetic_substitution --text "hello world"
```

Decode text with:

```bash
python -m src.utils.codec decode \
  --transformation monoalphabetic_substitution --text "xpvve nekvq"
```

Available transformations are `monoalphabetic_substitution`, `polygraphic_substitution`, `variable_length_substitution`, and `flip_coding`. Text is normalized to lowercase. Encoding also supports `--perturbation substitution`, `insertion`, or `deletion` with `--eta` for monoalphabetic substitution. Omit `--text` to read from standard input.

---

## Citation

If you use this code, please cite our paper:

```bibtex
@misc{li2026llmalignmentutilityasymmetry,
      title={LLM Alignment--Utility Asymmetry under Semantic-Preserving Transformations}, 
      author={Mohan Li and Chengyu Yu and Francesco Sovrano and Marc Langheinrich and Martin Gjoreski},
      year={2026},
      eprint={2609.32717},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2609.32717}, 
}
```
