# Testing the Gradient-Noise Model of Large-Batch Training

This repository implements a minimal empirical test of the local batch-size model proposed in McCandlish, Kaplan, Amodei et al., *An Empirical Model of Large-Batch Training* (2018).

The paper models a minibatch gradient as a noisy estimate of the full training gradient and predicts how the learning rate that gives the best expected one-step training-loss change should depend on batch size. The central prediction is

$$
\epsilon_{\text{opt}}(B)
=
\frac{\epsilon_{\max}}
{1 + B_{\text{noise}} / B}.
$$

where

$$
\epsilon_{\max}
=
\frac{\lVert G \rVert^2}{G^T H G}
$$

is the optimal local step size for the exact full-data gradient, and

$$
B_{\text{noise}}
=
\frac{\operatorname{tr}(H\Sigma)}{G^T H G}
$$

is the gradient-noise scale.

Here:

- $G = \nabla_\theta L(\theta)$ is the full-data gradient.
- $H = \nabla_\theta^2 L(\theta)$ is the Hessian of the full-data loss.
- $\Sigma = \operatorname{Cov}(g_i)$ is the covariance of per-example gradients.
- $B$ is minibatch size.

The interpretation of $B_{\text{noise}}$ is that it marks the transition between two regimes:

$$
B \ll B_{\text{noise}}
\quad\Rightarrow\quad
\epsilon_{\text{opt}}(B) \propto B,
$$

so increasing batch size substantially improves the best achievable one-step progress, while

$$
B \gg B_{\text{noise}}
\quad\Rightarrow\quad
\epsilon_{\text{opt}}(B) \approx \epsilon_{\max},
$$

so further batch increases give diminishing returns.

This repository first tests this local prediction on MNIST. At fixed model checkpoints, it measures the empirical learning rate that gives the maximum expected one-step loss reduction after one SGD-style update, equivalently the minimum standard loss change $\Delta L = L_{\text{after}} - L_{\text{before}}$, for different batch sizes, and compares the measured curve with the theoretical formula above.

## Current project stage

The current repository state contains a completed first MNIST experiment and report notebook:

- `mnist_optimal_batch_lr.py` runs the local one-step experiment with PyTorch and torchvision.
- `mnist_optimal_batch_lr_report.ipynb` explains the objective, formulas, procedure, retained plots, fitted parameters, and current limitations.
- `outputs/mnist_optimal_batch_lr_dense/` contains the current dense run results: raw CSV data, aggregate CSVs, fitted parameters, and report plots.

Current dense-run configuration:

- Checkpoints: steps `100` and `500`.
- Batch sizes: `4, 8, 16, 32, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048`.
- Sampled gradient minibatches per condition: `K = 10`.
- Evaluation subset: `2,000` fixed MNIST training examples.
- Epsilon grid: dense around the observed optimum and broad enough to show loss increase for too-large steps.

Current fitted values, using the quadratic-refined $\epsilon_{\text{opt}}$ estimates:

| checkpoint | $\epsilon_{\max}$ | $B_{\text{noise}}$ |
|---:|---:|---:|
| 100 | 0.148 | 49.8 |
| 500 | 0.187 | 151 |

The main qualitative result is that normalized $\epsilon_{\text{opt}} / \epsilon_{\max}$ follows the predicted saturating curve versus $B / B_{\text{noise}}$. The raw grid optima are also saved, but the normalized theory plot uses a local quadratic refinement because the raw grid argmin is visibly discretized in the high-batch regime.

## Local experimental question

At fixed parameters $\theta$, for batch size $B$, define a minibatch gradient

$$
g_{B,k}
=
\nabla_\theta L_{b_k}(\theta)
=
\frac{1}{B}
\sum_{(x_i,y_i)\in b_k}
\nabla_\theta \ell(f_\theta(x_i), y_i),
$$

where $b_k$ is the $k$-th randomly sampled minibatch and $\ell$ is per-example cross-entropy loss.

For a candidate learning rate $\epsilon$, form the hypothetical update

$$
\theta'_{B,k,\epsilon}
=
\theta - \epsilon g_{B,k}.
$$

The one-step loss change is measured on a fixed large evaluation subset:

$$
\Delta L_{B,k}(\epsilon)
=
L_{\text{eval}}(\theta'_{B,k,\epsilon})
-
L_{\text{eval}}(\theta).
$$

A negative value means that the update reduced evaluation loss.

For each batch size, average over many independently sampled minibatches:

$$
\overline{\Delta L}_{B}(\epsilon)
=
\frac{1}{K}
\sum_{k=1}^{K}
\Delta L_{B,k}(\epsilon).
$$

The empirical optimal learning rate is then

$$
\epsilon_{\text{opt}}^{\text{meas}}(B)
=
\arg\min_{\epsilon}
\overline{\Delta L}_{B}(\epsilon).
$$

The experiment tests whether

$$
\epsilon_{\text{opt}}^{\text{meas}}(B)
\approx
\frac{\epsilon_{\max}}
{1 + B_{\text{noise}} / B}.
$$

## MNIST setup

- Dataset: MNIST training set.
- Model: MLP:

$$
784 \rightarrow 256 \rightarrow \text{ReLU} \rightarrow 10.
$$

- Loss: mean cross-entropy over examples in a batch.
- Optimizer used to create checkpoints: plain SGD, no momentum, no weight decay, no learning-rate schedule.
- Baseline training batch size: $64$.
- Checkpoints in the dense report run: steps $100$ and $500$.
- Evaluation set in the dense report run: a fixed held-out subset of $2{,}000$ MNIST training examples.
- Number of sampled minibatches per condition in the dense report run: $K = 10$.

The evaluation set is kept fixed during a local experiment. The sampled minibatches used to construct gradients are drawn separately from the remaining training data.

## Protocol

For each saved checkpoint $\theta_t$:

1. Freeze model parameters at $\theta_t$.

2. Choose batch sizes:

$$
B \in \{4, 8, 16, 32, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048\}.
$$

3. Choose a learning-rate grid that is broad enough to show instability at large steps and dense near the observed optimum:

$$
\epsilon \in
\{10^{-4}, 3 \cdot 10^{-4}, 10^{-3}, \ldots, 0.04, 0.05, \ldots, 0.4, 0.5, 0.7, 1.0\}.
$$

4. For each batch size $B$:

   1. Sample $K$ independent minibatches $b_1, \ldots, b_K$.
   2. Compute $g_{B,k}$ for each minibatch at the same frozen $\theta_t$.
   3. For every candidate $\epsilon$, form:

      $$
      \theta'_{B,k,\epsilon}
      =
      \theta_t - \epsilon g_{B,k}.
      $$

4. Evaluate the loss change:

      $$
      \Delta L_{B,k}(\epsilon)
      =
      L_{\text{eval}}(\theta'_{B,k,\epsilon})
      -
      L_{\text{eval}}(\theta_t).
      $$

   5. Average $\Delta L_{B,k}(\epsilon)$ over $k$.

5. For each $B$, find:

$$
\epsilon_{\text{opt}}^{\text{meas}}(B)
=
\arg\min_{\epsilon}
\overline{\Delta L}_{B}(\epsilon).
$$

6. Fit the measured values to:

$$
\epsilon_{\text{opt}}(B)
=
\frac{\epsilon_{\max}}
{1 + B_{\text{noise}} / B}.
$$

## Outputs

For each checkpoint, the experiment produces:

1. A plot for finding the maximum one-step loss reduction, shown as the expected one-step loss change

$$
\overline{\Delta L}_{B}(\epsilon),
$$

against $\epsilon$ for each batch size. The optimum is the minimum of this curve because negative $\Delta L$ means loss reduction.

2. A plot of

$$
\epsilon_{\text{opt}}^{\text{meas}}(B)
$$

against $B$.

3. A fitted estimate of

$$
\epsilon_{\max}
\quad\text{and}\quad
B_{\text{noise}}.
$$

4. A comparison between the measured learning-rate curve and the theoretical prediction.

The experiment is local: it does not yet claim that this one-step-optimal rule is globally optimal for full neural-network training. A separate full-training experiment is needed to test whether the locally measured noise scale predicts the time--compute tradeoff over complete training runs.

## Running the MNIST experiment

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run the experiment:

```bash
python3 mnist_optimal_batch_lr.py
```

All experiment knobs are defined at the top of `mnist_optimal_batch_lr.py`: seed, device, checkpoint steps, batch sizes, number of sampled minibatches `K`, epsilon grid, and evaluation subset size. The default settings match the protocol above and can be expensive on CPU because every hypothetical update is evaluated on the fixed evaluation subset. For a smoke test, temporarily reduce `CHECKPOINT_STEPS`, `BATCH_SIZES`, `K`, and `EVAL_SUBSET_SIZE`.

Outputs are written under `outputs/mnist_optimal_batch_lr_dense/`:

- `config.json`: full run configuration plus resolved device.
- `raw_results.csv`: one row per checkpoint, batch size, epsilon, and sampled minibatch.
- `aggregate_results.csv`: mean and standard error of both `delta_loss = L_before - L_after` and `loss_change = L_after - L_before`.
- `epsilon_opt_results.csv`: grid optimum and quadratic local refinement for each checkpoint and batch size.
- `fit_results.csv`: fitted `epsilon_max` and `B_noise` using the quadratic-refined optima.
- `*.png`: retained report plots, including the maximum-loss-reduction curves, normalized epsilon theory comparison, secondary local-quadratic check, and supplemental checkpoint context plots.
