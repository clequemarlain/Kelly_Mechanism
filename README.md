<p align="center">
  <img src="cognetslogo1" alt="CoGNETs Logo" width="400"/>
</p>




<div style="text-align:center;">
    <img src="https://github.com/clequemarlain/Projets_de_Master/assets/79755084/e2e4cab3-2b09-4050-8076-5fc955835ca8" alt="Image 1" width="40%">
    <img src="https://github.com/clequemarlain/Projets_de_Master/assets/79755084/0b545ddb-585c-4041-a438-63536d05780b" alt="Image 2" width="40%">
</div>
# CoGNETs: Resource Allocation Algorithms for Edge Computing

This repository contains learning algorithms and simulation scripts developed as part of the **CoGNETs Project (Continuums of Game NETs)**.  
The goal is to design and evaluate distributed learning algorithms for **strategic resource allocation** in edge computing networks modeled as games.

---

## 🔍 Project Scope

Edge computing environments involve multiple agents competing for limited computational resources.  
The CoGNETs project investigates **learning dynamics and game-theoretic mechanisms** for:

- Distributed resource allocation
- Pricing and bidding in edge systems
- Equilibrium computation and convergence analysis

---

## ⚙️ Algorithms Included

### 🔹 Best Response Dynamics
Agents iteratively compute their optimal resource allocation given the strategies of others.  
Helps study:
- Nash equilibrium convergence
- Oscillations in strategic environments

### 🔹 Dual Averaging
A no-regret learning approach where agents use gradient feedback to update their allocations.  
Useful for:
- Convergence in noisy settings
- Studying learning rates and stability

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Recommended libraries:
  - `numpy`
  - `scipy`
  - `matplotlib`

Install all dependencies using:

```bash
pip install -r requirements.txt
