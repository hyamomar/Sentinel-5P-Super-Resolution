<h1 align="center">
  Sentinel-5P Super-Resolution
</h1>

<h3 align="center">
  Supervised and Self-Supervised Deep Learning for Hyperspectral Image Super-Resolution
</h3>

<p align="center">
  Hyam Omar Ali, Antoine Crosnier, Romain Abraham, Baptiste Combelles, Fabrice Jégou, Bruno Galerne
</p>

<p align="center">
  Université d’Orléans · Université de Tours · CNRS · University of Khartoum · ENS Lyon · LPC2E
</p>

---

This repository presents deep learning approaches for enhancing the spatial resolution of Sentinel-5P (S5P) hyperspectral satellite data.  
It includes both a **supervised framework** and a **self-supervised extension** designed for real-world deployment where high-resolution ground truth is unavailable.

---

## Supervised Super-Resolution

<h4 align="center">
  <a href="https://arxiv.org/abs/2501.17210">
    Depth Separable Architecture for Sentinel-5P Super-Resolution
  </a>
</h4>

<p align="center"><strong>Accepted at IGARSS 2025</strong></p>

<p align="center">
  <img src="images/SR_results.png" width="750"/>
</p>

<p align="center">
  <em>Super-resolution results for selected spectral bands using the S5-DSCR model.</em>
</p>

---

## Self-Supervised Super-Resolution

<h4 align="center">
  <a href="#">
    Self-Supervised Super-Resolution for Sentinel-5P Hyperspectral Images
  </a>
</h4>

<p align="center"><strong>Available on arXiv</strong></p>

<p align="center">
  <img src="images/GT_SHR.png" width="900"/>
</p>

<p align="center">
  <em>Qualitative results demonstrating enhanced spatial detail and consistency across spectral bands without using HR ground truth.</em>
</p>

---

## Overview

- **Supervised SR:** learns from synthetic LR–HR pairs using physics-based degradation  
- **Self-Supervised SR:** learns directly from real observations without HR ground truth  
- **Goal:** improve spatial resolution while preserving spectral fidelity in S5P hyperspectral data  

---

## Explore the Repository

- 🔹 Supervised implementation → `supervised/`  
- 🔹 Self-supervised framework → `self_supervised/`  

---

## Motivation

Sentinel-5P provides hyperspectral observations critical for atmospheric and environmental monitoring, but its spatial resolution limits fine-scale analysis.

This work addresses this limitation by:
- enhancing spatial detail  
- preserving spectral consistency  
- enabling deployment in real-world scenarios where high-resolution data do not exist  

---

## Citation

If you use this repository, please cite:

```bibtex
@inproceedings{ali2025depth,
  title={Depth Separable Architecture for Sentinel-5P Super-Resolution},
  author={Ali, Hyam Omar and Abraham, Romain and Galerne, Bruno},
  booktitle={IGARSS 2025},
  year={2025}
}
