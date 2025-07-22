# *World Modeling Makes a Better Planner:* Dual Preference Optimization for Embodied Task Planning


  [**📖 arXiv**](https://arxiv.org/abs/2503.10480) | [**🤗 Paper**](https://huggingface.co/papers/2503.10480) | [**🤗 Dataset**](https://huggingface.co/datasets/sinwang/D2PO) | [**GitHub**](https://github.com/sinwang20/D2PO) | [**📣 Twitter/X**](https://x.com/wang_siyin/status/1900427845652160659) 

This repository contains the code and data for our paper: [World Modeling Makes a Better Planner: Dual Preference Optimization for Embodied Task Planning](https://arxiv.org/abs/2503.10480).


Recent advances in large vision-language models (LVLMs) have shown promise for embodied task planning, yet they struggle with fundamental challenges like dependency constraints and efficiency. Existing approaches either solely optimize action selection or leverage world models during inference, overlooking the benefits of learning to model the world as a way to enhance planning capabilities. We propose **Dual Preference Optimization (D²PO)**, a new learning framework that jointly optimizes state prediction and action selection through preference learning, enabling LVLMs to understand environment dynamics for better planning. To automatically collect trajectories and stepwise preference data without human annotation, we introduce a tree search mechanism for extensive exploration via trial-and-error. Extensive experiments on VoTa-Bench demonstrate that our D^2PO-based method significantly outperforms existing methods and GPT-4o when applied to Qwen2-VL (7B), LLaVA-1.6 (7B), and LLaMA-3.2 (11B), achieving superior task success rates with more efficient execution paths.

---

# 🎉 News
[2025-05-16] Our paper is accepted by ACL 2025 (main)! 
[2025-03-26] Our paper is accepted by [ICLR 2025 Workshop on World Models](https://sites.google.com/view/worldmodel-iclr2025/)!

# 🤗 D2PO Dataset

The **D2PO** dataset contains various data splits for alignment training, including supervised fine-tuning and direct preference optimization.

| Split Name   | Description                               | Size       | 
|--------------|-------------------------------------------|--------------|
| [🤗 SFT_Policy](https://huggingface.co/datasets/sinwang/D2PO/blob/main/sft-data-600-images+.json) | SFT data for action selection             | 4.5k | 
| [🤗 DPO_Policy](https://huggingface.co/datasets/sinwang/D2PO/blob/main/dpo-data-600-images+.json) | DPO data for action selection  | 15k        |
| [🤗 DPO_World](https://huggingface.co/datasets/sinwang/D2PO/blob/main/dpo-data-600-world+.json)  | DPO data for state prediction     | 8.7k        | 



## 🚀 Install

1. Clone the whole repo.
    ```bash
    $ git clone {repo_url}
    ```

1. Setup a virtual environment.
    ```bash
    $ conda create -n vota python=3.8
    $ conda activate vota
    ```

1. Install PyTorch (2.0.0) first (see https://pytorch.org/get-started/locally/).
    ```bash
    # exemplary install command for PyTorch 2.0.0 with CUDA 11.7
    $ pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 --index-url https://download.pytorch.org/whl/cu117
    ```

1. Install python packages in `requirements.txt`.
    ```bash
    $ pip install -r requirements.txt
    ```

---

## 📊 Benchmarking on VoTA-Bench

### 📦 Download ALFRED dataset.
```bash
$ cd alfred/data
$ sh download_data.sh json
```


### 🖥️ Running on Headless Server

If running the ALFRED experiments on a headless server, start the X display. Below script uses 1 for the X_DISPLAY id, but you can use different ids such as 0.

```bash
$ sudo python3 alfred/scripts/startx.py 1
```

Alternatively, you can use Xvfb:

```bash
$ Xvfb :1
```


### 🤖 Model Server

Both [vllm](https://github.com/vllm-project/vllm) and [sglang](https://github.com/sgl-project/sglang) are supported as model servers.

**Example: Start a vllm server for `Qwen2-VL-7B-Instruct`**

```bash
python -m vllm.entrypoints.openai.api_server --served-model-name Qwen2-VL-7B-Instruct --model Qwen/Qwen2-VL-7B-Instruct --port 30000
```

### 📝 Running Evaluation
```bash
$ python src/evaluate2.py --config-name=config_alfred
```

We use [Hydra](https://hydra.cc/) for configuration management. You can override settings in ./conf/config_alfred.yaml or via the command line.

Notes:
- `model_name` and `base_url` must match your chosen model server.
- `api_key` is required for OpenAI models like GPT-4o.
- `icl`: (True/False) enable or disable example usage. 
- `sft`: (True/False) set to True for SFT-style prompts.
- `eval_set`: choose 'valid_seen' or 'valid_unseen'.
- `eval_start_index` & `eval_end_index`: control the evaluation data range.



## 🌲 Data Exploration

1. First, set the `api_key` and `base_url` in `./src/task_planner.py` (lines 17–19). You can specify different models for different modules as needed.

2. Run the `scripts/run_{task_type}.sh` script to generate data in parallel using multiple GPUs. This script launches multiple processes to execute src/evaluate3.py, which collects data through a tree search mechanism.
You can control task parallelism and index assignment within the shell script using the following parameters:

  - `BASE_START_INDEX=`: starting index
  - `NODE_INCREMENT=50`: increment per node
  - `INCREMENT=10`: number of tasks per process
  - `NUM_TASKS=5`: number of parallel processes to launch

3. Process the generated data as required.



## 📝 TODO

- [x] Open source evaluation data and scripts (See section: 📊 Benchmarking on VoTA-Bench)
- [x] Release data collection scripts and training data

## 👋 Citation


**BibTeX:**

```bibtex
@article{wang2025world,
  title={World Modeling Makes a Better Planner: Dual Preference Optimization for Embodied Task Planning},
  author={Siyin Wang and Zhaoye Fei and Qinyuan Cheng and Shiduo Zhang and Panpan Cai and Jinlan Fu and Xipeng Qiu},
  journal={arXiv preprint arXiv:2503.10480},
  year={2025}
}
```
