<h1 align="center">Forget Sharpness: Perturbed Forgetting of Model Biases Within SAM Dynamics</h1>
<h3 align="center">Ankit Vani, Frederick Tung, Gabriel L. Oliveira, Hossein Sharifi-Noghabi</h3>

### [[Paper]](https://openreview.net/pdf?id=cU20finY8V)

**Abstract**: Despite attaining high empirical generalization, the sharpness of models trained with sharpness-aware minimization (SAM) do not always correlate with generalization error. Instead of viewing SAM as minimizing sharpness to improve generalization, our paper considers a new perspective based on SAM's training dynamics. We propose that perturbations in SAM perform *perturbed forgetting*, where they discard undesirable model biases to exhibit learning signals that generalize better. We relate our notion of forgetting to the information bottleneck principle, use it to explain observations like the better generalization of smaller perturbation batches, and show that perturbed forgetting can exhibit a stronger correlation with generalization than flatness. While standard SAM targets model biases exposed by the steepest ascent directions, we propose a new perturbation that targets biases exposed through the model's outputs. Our output bias forgetting perturbations outperform standard SAM, GSAM, and ASAM on ImageNet, robustness benchmarks, and transfer to CIFAR-{10,100}, while sometimes converging to sharper regions. Our results suggest that the benefits of SAM can be explained by alternative mechanistic principles that do not require flatness of the loss surface.

##

### This code is based on [PyTorch Image Models (timm) v0.9.13dev0](https://github.com/huggingface/pytorch-image-models/tree/f2fdd97e9f859285363c05988820c9350b737e59).

Scripts for training and evaluating the models reported in our paper are provided in `scripts/`. Particularly, `scripts/run_experiments.sh` includes commands to train the models on ImageNet and fine-tune them on CIFAR-10/100.

An implementation of [Sharpness-Aware Minimization (SAM)](https://arxiv.org/abs/2010.01412), [Surrogate Gap Guided SAM (GSAM)](https://arxiv.org/abs/2203.08065), and [Adaptive SAM (ASAM)](https://arxiv.org/abs/2102.11600) using PyTorch's `foreach` API and other optimizations is provided in `perturbed_forgetting/optim/sam.py`.

The output-bias forgetting (OBF) perturbation function is provided in `perturbed_forgetting/loss/output_bias_forget.py`.

**Other tips:**

- To avoid a memory leak with [TensorFlow Datasets (TFDS)](https://www.tensorflow.org/datasets/overview), you may need [`gperftools`](https://github.com/gperftools/gperftools), which provides `libtcmalloc.so.4`. Replace the path to this file in the placeholders in the included scripts, otherise remove the `LD_PRELOAD=...` line.
- To log metrics using [Weights and Biases (wandb)](https://docs.wandb.ai/), pass `--log-wandb` to `train.py`.

##

### Citation
```bibtex
@inproceedings{vani2024forget,
  title={Forget Sharpness: Perturbed Forgetting of Model Biases Within {SAM} Dynamics},
  author={Ankit Vani and Frederick Tung and Gabriel L. Oliveira and Hossein Sharifi-Noghabi},
  booktitle={Forty-first International Conference on Machine Learning},
  year={2024},
  url={https://openreview.net/forum?id=cU20finY8V}
}
```
