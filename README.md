

<h1 align="center">Structure Diffusion Quantization</h1>

<p align="center">
    [<a href="">paper</a>] &bull;
    [<a href="https://www.mtandhj.com/slides/sdq/">slide</a>] &bull;
    [<a href="https://github.com/MTandHJ/RecBoard">baselines</a>]
</p>


This repository contains the implementation of "**Structure Diffusion Quantization: Learning Stable Coarse-to-Fine Semantic IDs for Generative Recommendation**".

## ⚙️ Requirements

```bash
conda create -n sdq python=3.10
conda activate sdq
python -m pip install --upgrade pip
python -m pip install -e .
freerec setup
```

The commands below assume that processed data are available under `../data/Processed/<dataset>/`. Each dataset directory must contain `item.txt` with the `TITLE`, `CATEGORIES`, and `BRAND` fields. Before quantization, generate the textual item features expected by the default configurations:

```bash
python encode_textual_features.py \
    --root ../data \
    --dataset Amazon2014Beauty_550_LOU \
    --model sentence-t5-xl \
    --model-dir ./models \
    --device cuda
```

This command writes `sentence-t5-xl_title_categories_brand.pkl` to the dataset directory. Set `--model-dir ""` to load `sentence-t5-xl` directly from Hugging Face instead of a local model directory. Select a PyTorch build compatible with the target CUDA runtime when preparing a GPU environment.

## 🚀 Usage

### Tokenizer

Run `SDQ-VAE` with the provided Beauty configuration:

```bash
python train_sdq_vae.py \
    --config configs/sdq/Amazon2014Beauty_550_LOU.yaml
```

Run the non-trainable `SDQ-KMeans` variant:

```bash
python train_sdq_kmeans.py \
    --dataset Amazon2014Beauty_550_LOU
```

### T5

T5 training requires the `sid_vocab.json` exported by an SDQ-VAE or SDQ-KMeans run. Specify its path through `--sid-vocab-file`:

```bash
python train_t5.py \
    --config configs/t5/Amazon2014Beauty_550_LOU.yaml \
    --sid-vocab-file logs/SDQ/Amazon2014Beauty_550_LOU/<run-id>/sid_vocab.json
```

Replace `<run-id>` with the identifier of the selected quantization run. For SDQ-KMeans, use the corresponding vocabulary under `logs/SDQ-KMeans/` instead.

Replace the configuration path with `Amazon2014Toys_550_LOU.yaml` or `Amazon2014Sports_550_LOU.yaml` to run the corresponding dataset. Experiment artifacts, including the generated `sid_vocab.json`, are written under `logs/`.


## Citation

```
@inproceedings{xu2027sdq,
  title={Structure diffusion quantization: Learning stable coarse-to-fine semantic ids for generative recommendation},
  author={Xu, Cong and Liao, Zihan and Wei, Lingxiao and Wang, Jun and Zhang, Wei},
  booktitle={ACM Conference on Knowledge Discovery and Data Mining (KDD)},
  year={2027}
}
```
