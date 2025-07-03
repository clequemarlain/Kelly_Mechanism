# CoGNETs Project – Resource Allocation Algorithms

This repository contains prototype implementations of algorithms developed and tested in the context of the **CoGNETs Project** (Collaborative Games for Networked Edge Systems). The goal is to explore learning dynamics for **strategic resource allocation** in distributed edge computing environments, modeled as games.

---

## 📌 Project Context

The CoGNETs Project investigates how **distributed learning and game-theoretic approaches** can optimize pricing and resource allocation in edge computing networks. This repository specifically includes:

- Explorations of **learning dynamics** and **equilibrium computation** in resource allocation games.
- Experimental code for evaluating the performance and convergence of different learning methods.

---

## ⚙️ Algorithms Implemented

### Best Response Dynamics
A classic learning dynamic where each agent iteratively updates their resource allocation by solving their individual optimization problem, assuming the strategies of other agents are fixed.

> ✔️ Suitable for:
> - Benchmarking equilibrium computation.
> - Studying convergence and oscillations in resource allocation games.

---

### Dual Averaging
A distributed optimization method where agents update their strategies based on **gradient feedback** and averaging mechanisms. Adapted here to the context of **budget-constrained allocation games**.

> ✔️ Suitable for:
> - Studying no-regret learning dynamics.
> - Scenarios with noisy or partial feedback.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Recommended packages:
  - `numpy`
  - `matplotlib`
  - `scipy`

Install them using:

```bash
pip install -r requirements.txt
