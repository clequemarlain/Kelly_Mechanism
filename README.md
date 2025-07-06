<div align="center" style="display: flex; justify-content: center; align-items: center; gap: 20px;">
    <img src="cognetslogo1.png" alt="CoGNETs Logo" style="width: 30%;">
    <img src="avignon_university_logo.png" alt="Avignon University Logo" style="width: 30%;">
</div>

# CoGNETs Game Catalogue – Kelly Mechanism Simulations

Welcome to the repository for **Kelly Mechanism Simulations** developed within the **CoGNETs Project (Continuums of Game NETs)**.  
This repository provides simulation scripts and learning algorithms for resource allocation games, as detailed in the **CoGNETs Game Catalogue (Section 5.1: Kelly Mechanism)**.

---

## 📚 Reference: CoGNETs Game Catalogue

This repository implements algorithms described in the CoGNETs Game Catalogue:

- **5.1.3 Learning via Dual Averaging**
- **5.1.4 Learning via Best Response Dynamics**

**Features:**
- Complete **problem formulation** for proportional resource allocation games
- Algorithms for **computing Nash equilibria**
- Distributed learning algorithms:
  - **Dual Averaging (DAQ)**
  - **Best Response Dynamics (SBRD)**
  - **Exponential Learning (XL)**

---

📄 **Related Publication:**  
_"Learning to Bid in Proportional Allocation Auctions with Budget Constraints."_

👉 [Access the report on HAL](https://hal.archives-ouvertes.fr/hal-XXXXXXX) <br>
*(replace with your actual link)*

---

## 🔍 Problem Scope

The **Kelly Mechanism** addresses resource allocation in settings such as edge networks, where multiple agents (players) compete for a shared resource by bidding.  
Each agent seeks to maximize a utility function under budget and resource constraints.

This project focuses on the study and simulation of **distributed learning dynamics** in such strategic, multi-agent environments.

---

## ⚙️ Algorithms Implemented

- **Synchronous Best Response Dynamics (SBRD):**  
  Iterative computation of each player's best response.  
  _Focus:_ Equilibrium search, convergence/divergence analysis

- **Dual Averaging (DAQ):**  
  No-regret, gradient-based learning.  
  _Focus:_ Robustness and stability under noisy feedback

- **Exponential Learning (XL):**  
  Acceleration of Dual Averaging dynamics for improved convergence speed

---

## 🗂️ Project Structure

```
├── build_game.py        # Defines the Kelly game, alpha-fair utility, DA, SBRD, XL, and learning process
├── config_table.py      # Simulation configuration parameters
├── main.py              # Entry point: runs experiments and outputs results
├── utils.py             # Helper functions (plotting, logging, etc.)
└── README.md            # This file
```

---

## 🚀 Getting Started

1. **Clone the repository:**
    ```bash
    git clone https://github.com/clequemarlain/Kelly_Mechanism.git
    cd Kelly_Mechanism
    ```
2. **Install requirements:**  
   *(Add requirements.txt if needed)*

3. **Run experiments:**  
   Edit `config_table.py` to set up your scenario, then execute:
    ```bash
    python main.py
    ```

---

## 🤝 Acknowledgements

This research was conducted as part of the CoGNETs project (WP3 2024–2025), with the support of Avignon University.

---
