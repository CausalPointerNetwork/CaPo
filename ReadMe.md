# 🧬 Causal Structure Learning via Pointer Networks & DAGMA

An end-to-end continuous optimization framework for Causal Discovery (DAG learning) across benchmark Bayesian networks. This repository combines **Pointer Networks** for order searching and **DAGMA** (DAGs via M-matrices) with dynamic reward engineering and Optuna hyperparameter optimization.

---

## 📌 Key Features

* **Order Search via Policy Gradient:** Leverages Pointer Networks to output a causal node topological ordering using Reinforcement Learning (REINFORCE).
* **DAGMA-based Graph Learning:** Optimizes continuous adjacency weight matrices subject to acyclicity constraints using log-det M-matrix formulation.
* **Automated Tuning:** Integrated Optuna hyperparameter search across custom loss weights ($w_{\text{nll}}$, $w_{\text{dag}}$, $w_{\text{bic}}$, $L_1$ regularization, and Learning Rate).
* **Markov Boundary Constraints:** Computes fast observational Markov Boundaries to prune search space effectively.
* **Comprehensive Benchmarking:** Pre-configured evaluation pipeline for canonical datasets (`Asia`, `Sachs`, `Alarm`, `Child`, `Insurance`, `Hepar`, etc.) with SHD and F1-score logging.

---

## 🛠️ Requirements & Installation

Make sure you have Python 3.8+ installed along with GPU support for PyTorch:

```bash
pip install torch numpy pandas scikit-learn optuna
