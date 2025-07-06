<div align="center">
    <img src="cognetslogo1.png" alt="CoGNETs Logo" width="60%">
</div>

# CoGNETs Game Catalogue – Kelly Mechanism Simulations

This repository contains simulation scripts and learning algorithms developed for the **CoGNETs Project (Continuums of Game NETs)**.  
It focuses on resource allocation mechanisms modelled as games and implements several algorithms described in the **CoGNETs Game Catalogue (Section 5.1: Kelly Mechanism)**.

---

## 📚 Reference: CoGNETs Game Catalogue

This repository corresponds to the **Kelly Mechanism** described in the CoGNETs Game Catalogue:

5.1.3 Learning via Dual Averaging ..................................

5.1.4 Learning via Best Response Dynamics ..........................


It implements:
- The **problem formulation** and resource allocation game
- Algorithms for **computing Nash equilibria**
- Learning algorithms:
  - **Dual Averaging (DAQ)**
  - **Best Response Dynamics (SBRD)**
  - **Exponential Learning (XL)**

---

📄 **Related Publication:**  
_"Learning to Bid in Proportional Allocation Auctions with Budget Constraints."_  
 
👉 [Access the report on HAL](https://hal.archives-ouvertes.fr/hal-XXXXXXX) (replace with your actual link)

---


## 🔍 Problem Scope

The Kelly Mechanism models the resource allocation problem, which can be applied in edge networks where multiple agents (players) bid for a shared resource.  
Each agent aims to maximize a utility function subject to budget and resource constraints.

This project investigates **distributed learning dynamics** for such strategic settings.

---

## ⚙️ Algorithms Implemented

### 🔹 Synchronous Best Response Dynamics (SBRD)
Iterative computation of each player's best response.  
Focus:
- Equilibrium search
- Convergence/divergence analysis

### 🔹 Dual Averaging (DAQ)
No-regret gradient-based learning.  
Focus:
- Stability in noisy.

### 🔹 Exponential Learning (XL)
Acceleration of Dual Average learning dynamics to improve convergence speed.

---

## 🛠️ Project Structure

├── build_game.py # Defines the Kelly game,  alpha-fair utility function, DA, SBRD, XL, and learning process

├── config_table.py # Simulation configuration parameters

├── main.py # Main script to run experiments and print results

├── utils.py # Helper functions (plots, logging, etc.)

└── README.md # This file


🤝 Acknowledgements

This research was conducted as part of the CoGNETs project (WP3 2024 - 2025)
