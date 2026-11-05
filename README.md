# Crime Prediction System — GothamWatch: Pre-Emptive Tactical Containment

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/priyadarshan-l/gothamwatch-crime-prediction/blob/main/GothamWatch_Crime_Prediction_Colab.ipynb)

### Team Members
- **Priyadarshan** (`RA2311003011719`)
- **Sagnik Sannigrahi** (`RA2311003011709`)
- **Kshetij Abheru** (`RA2311003011484`)

### Overview
An autonomous crime-intelligence and pre-emptive tactical containment system trained and evaluated on **25,000 real City of Chicago Crime Dataset (`ijzp-q8t2`)** records mapped across an 8-sector spatial city graph:
1. **Pillar 1 — Semantic Web Knowledge Graph (`rdflib` RDF/OWL + SPARQL)**: 434 RDF triples linking underworld syndicates, operatives, vehicles, weapons, M.O. signatures, high-value vaults, critical infrastructure, and real Chicago crime incidents with automated OWL/Rule-based decoy inference.
2. **Pillar 2 — Graph Deep Learning (`PyTorch SpatioTemporalGAT`)**: Multi-Head Graph Attention Network + Temporal GRU achieving **0.8716 Threat ROC-AUC** and **96.77% Decoy Detection Accuracy** on the test split.
3. **Pillar 3 — Parallel Multi-Agent Adversarial War Room (`asyncio`)**: 6 autonomous agents executing in parallel (`Oracle` -> `[Red-Cell Mastermind || SPARQL Detective || Fleet Monitor]` -> `Blue-Cell Commander` -> `Sentinel Safety Validator`) to seal bridge choke points (`R-1 Gotham Bridge`) before getaway.
