# Task Drift

### Adversarial Attacks and Defences for Task Drift Detection in LLMs

This repository contains the code and experiments from my MSc thesis, where I **studied adversarial robustness of task drift detection in Large Language Models (LLMs).**

Task drift detection, introduced by Microsoft, uses lightweight logistic regression probes on hidden activations of LLMs to detect **stealthy prompt injection attacks**. My work extends this line of research by:

### Adversarial Attacks

* Implemented and adapted the **Greedy Coordinate Gradient (GCG)** algorithm for LLM safety research.

* Extended the attack from a single classifier to **multiple probes attached to different layers** of the model.

* Designed a **layer-wise gradient accumulation scheme** that combines upstream/backprop gradients with classifier-specific losses.

* Successfully discovered **universal adversarial suffixes** that generalize across prompts.

* Attacks achieved near-perfect success on Microsoft-trained probes for **Phi-3 (3.8B)** and **LLaMA-3 (8B)**.

### Defences

* Explored **adversarial training** using PGD — found it ineffective and highly unstable against suffix attacks.

* Proposed a new defence strategy:

  1. Generate a diverse pool of adversarial suffixes.
  2. Randomly append one to prompts during training.
  3. Retrain classifiers on the resulting poisoned activations.

* This approach preserved baseline accuracy while achieving **80–99% robustness under strict evaluation** and **96–100% robustness under majority-vote metrics**.

* Direct suffix optimisation against these defended classifiers proved **extremely difficult** — demonstrating strong resilience.

### Key Results

* **Attacks**: Discovered suffixes that fool all probes on multiple LLMs.

* **Defences**: Achieved state-of-the-art robustness while maintaining accuracy on clean inputs.

* **Takeaway**: Randomized adversarial suffix training is an effective and lightweight defence against prompt injection attacks.

### Technical Details

* **Models**: [Phi-3 Mini (3.8B)](https://huggingface.co/microsoft/phi-3-mini-3.8b), [LLaMA-3 (8B)](https://huggingface.co/meta-llama/Meta-Llama-3-8B)

* **Classifiers**: Logistic regression probes trained on hidden activations.

* **Libraries**: PyTorch, HuggingFace Transformers.

* **Techniques**: Gradient-based adversarial optimization, adversarial training, model probing.