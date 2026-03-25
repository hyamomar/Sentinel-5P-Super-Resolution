```markdown
# Self-Supervised Super-Resolution for Sentinel-5P

This module extends the supervised S5-DSCR framework to a **self-supervised setting**, enabling super-resolution without requiring high-resolution ground truth.

---

## Motivation

In real-world satellite imaging:
- High-resolution ground truth is often unavailable  
- Supervised approaches rely on simulated degradation  

This motivates a **self-supervised alternative**

---

## Key Idea

Instead of training with LR–HR pairs:

- Input: HR image  
- Model predicts: Super-resolved HR (SHR)  
- Learning is based on **scale consistency**

---

## Methodology

The framework includes:

- Scale-equivariant learning  
- Multi-scale consistency  
- Shared architecture with supervised model (ResDSC)  

---

## Training Strategy

- No explicit LR–HR pairs  
- Implicit supervision via scaling  
- Consistency constraints across resolutions  

---

## Evaluation Strategy

Unlike supervised methods:

- No synthetic LR generation  
- HR is directly used as input  
- Model predicts a refined HR (SHR)

Evaluation is performed between:
- Original HR
- Predicted SHR
