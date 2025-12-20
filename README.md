# Spam vs Ham Classification with Bayes Inverse + SmolLM2

This project explores **spam vs. ham email classification** using a **Bayes inverse classifier** on top of **SmolLM2-135M**. I evaluate three approaches—**zero-shot**, **naive prompting**, and **full fine-tuning**—and improve generalization using **label-aware data synthesis**, **weight decay**, and **gradient clipping**.

> Course: CPEN 455 (UBC)  
> Author: Anirudh Devanand

---

## Overview

The Bayesian inverse classification method leverages the generative capabilities of LLMs to perform classification tasks. By modeling the joint distribution of inputs and labels, we can compute the posterior probabilities of labels given the inputs, allowing for effective classification.

$$
\begin{aligned}
P_\theta(Y_\text{label}|X_{\leq i}) &= \frac{P_{\theta}(X_{\leq i}|Y_\text{label})P_{\theta}(Y_\text{label})}{P_{\theta}(X_{\leq i})}\\
&= \frac{P_{\theta}(X_{\leq i}|Y_\text{label})P_{\theta}(Y_\text{label})}{\displaystyle\sum_{Y'}P_{\theta}(X_{\leq i}|Y')P_{\theta}(Y')}\\
&= \frac{P_{\theta}(X_{\leq i},Y_\text{label})}{\displaystyle\sum_{Y'}P_{\theta}(X_{\leq i},Y')}
\end{aligned}
$$

Where:
- $X_{\leq i}$: Input sequence up to position $i$, representing the email content.
- $Y_\text{label}$: The label indicating whether the email is spam or not spam.
- $\theta$: Parameters of the pre-trained language model.
- $P_\theta(Y_\text{label}|X_{\leq i})$: Posterior probability of label $Y_\text{label}$ given input $X_{\leq i}$
- $P_\theta(X_{\leq i}|Y_\text{label})$: Likelihood of input $X_{\leq i}$ given label $Y_\text{label}$
- $P_\theta(X_{\leq i},Y_\text{label})$: Joint probability of input $X_{\leq i}$ and label $Y_\text{label}$
- $P_\theta(Y_\text{label})$: Prior probability of label $Y_\text{label}$



---

## Methods Evaluated

### 1) Zero-shot
Apply Bayes inverse directly with the pre-trained model (no task-specific training).

### 2) Naive prompting
Prepend a short instruction to each email but still use pre-trained weights.

### 3) Full fine-tuning (main method)
Supervised fine-tuning on the labeled train/validation subset, then recompute Bayes inverse probabilities.
All follow-up experiments (data synthesis, regularization, ensembling) build on top of this pipeline.

---

## Results (Train/Validation Split)

| Method | Accuracy |
|------|----------|
| Zero-shot | 37.50% |
| Naive prompting | 53.125% |
| Full fine-tune | 78.472% |

**Takeaway:** Without task-specific training, performance is near random for this binary classification problem; fine-tuning is the main driver of improvement.

## Improvements

### Label-aware data synthesis (augmentation)

I implemented a data augmentation script `generate_synthetic_emails.py` that expands the labeled dataset:
- Input: `train_val_subset.csv`
- Output: `train_val_augmented.csv` (original + synthetic rows)

**Spam augmentations**
- Phrase injection (e.g., “Limited time offer…”, “Earn extra money…”, “Your account will be closed soon”)
- Random uppercase
- Punctuation noise

**Ham augmentations**
- Follow-up markers (e.g., “(follow up)”)
- Polite closings (e.g., “Please let me know if you have any questions.”)

Hyperparameters:
- `--spam_factor` and `--ham_factor` control how many synthetic variants are created per original email.


### Regularization: weight decay + gradient clipping (most effective)

Initial fine-tuning showed strong signs of overfitting: high train/val accuracy but lower accuracy on the hidden Kaggle test set.

Fixes:
- Add small **weight decay** to AdamW
- Apply **gradient clipping** (global grad norm)
- Increase learning rate (doubled from default)

**Outcome:** This regularization step gave the best generalization out of all methods tried.


### Ensembling (helpful but capped)

I trained **5 fine-tuned models** with different seeds and ensembled by averaging spam/ham probabilities.
This reduced variance, but the best Kaggle submission still plateaued at ~**75%** accuracy and did not close the gap as effectively as the single regularized model.

---

## Repository Pointers (as referenced in the report)

- `examples/bayes_inverse.py` — Bayes inverse classifier + fine-tuning pipeline
- `generate_synthetic_emails.py` — label-aware augmentation
- `train_val_subset.csv` — labeled subset input for training
- `train_val_augmented.csv` — synthesized output dataset
- `examples/chatbot_example.py` — interactive generation example (KV cache benefits)

---

## How to Run (Typical Flow)

1. Python Environment
   - [Install UV](https://docs.astral.sh/uv/getting-started/installation/#installing-uv).
   - Install Python dependencies with the following command from the project root:
    ```bash
    uv sync
    ```
2. Git clone autograder to this directory.
    ```bash
    git clone git@github.com:DSL-Lab/CPEN455-Project-2025W1-Autograder.git autograder
    ```
    If you are using windows laptop, you should first run `git clone git@github.com:DSL-Lab/CPEN455-Project-2025W1-Autograder.git` and then rename it to `autograder`.
    - Confirm the released datasets exist under `autograder/cpen455_released_datasets/`

3. Optional Generate synthetic data

    ```bash
    python generate_synthetic_emails.py \
    --input train_val_subset.csv \
    --output train_val_augmented.csv \
    --spam_factor <N> \
    --ham_factor <M>
    ```

4. train the model:
    ```
    # Run bayes inverse full finetune example
    bash examples/bayes_inverse_full_finetune.sh
    ```

5. Validate outputs:
    - Generate probabilities on test set with the following command (writes CSVs into `bayes_inverse_probs/`):
    ```bash
    uv run -m examples.save_prob_example
    ```
    - Validate everything end-to-end by executing `bash autograder/auto_grader.sh`; this is the same entry point used during grading.
