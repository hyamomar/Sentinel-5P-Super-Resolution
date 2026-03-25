<h1 align="center">
  <a href="https://arxiv.org/abs/2501.17210">
    Depth Separable Architecture for Sentinel-5P Super-Resolution
  </a>
</h1>


<h3 align="center">Hyam Omar Ali¹ ², Romain Abraham¹, Bruno Galerne¹ ³</h3>

<p align="center">¹ Institut Denis Poisson, Université d’Orléans, Université de Tours, CNRS, France</p>  
<p align="center">² Faculty of Mathematical Sciences, University of Khartoum, Sudan</p>  
<p align="center">³ Institut Universitaire de France (IUF)</p>

<p align="center"><strong>Pulished at IGARSS 2025</strong></p>


---

This repository contains the implementation of the S5-DSCR model, a deep learning approach designed to enhance the spatial resolution of Sentinel-5 Precursor (S5P) hyperspectral data.

The S5P satellite provides extensive hyperspectral observations across eight spectral bands, each containing approximately 500 channels. The proposed model leverages Depthwise Separable Convolution (DSC) to efficiently exploit spatial and spectral correlations while maintaining a lightweight architecture.

---

<p align="center">
  <img src="../images/SR_results.png" width="750"/>
</p>

<p align="center"><em>Super-resolution results for bands 3, 5 and 7. Each image is displayed using the first three PCA components.</em></p>

---

## Objectives

- Improve the spatial resolution of Sentinel-5P data  
- Exploit spatial and spectral correlations in hyperspectral imagery  
- Train models independently for each spectral band  

---

## Data Preparation 

### Dataset 

Sentinel-5P Level-1B radiance data were used as the primary data source. These data are freely accessible through the Copernicus Data Space Ecosystem.

The dataset consists of 15 orbits acquired on January 4, 2023 and September 7, 2023. Each orbit includes eight spectral bands, each with approximately 500 channels.

Due to variations in acquisition geometry, image dimensions vary across bands. To ensure consistency, all images were cropped into fixed-size patches:
- 512 × 256 (most bands)  
- 512 × 215 (SWIR bands)  

---

<p align="center">
  <img src="../images/Image_split.png" width="750"/>
</p>

<p align="center"><em>Full radiance image and corresponding cropped patches.</em></p>

---

### Degradation Model

Low-resolution images are generated using a physics-based degradation model that simulates the S5P acquisition process, including spatial blurring. A scaling factor of ×4 is applied.

---

## Methodology

The S5-DSCR model employs Depthwise Separable Convolution layers combined with residual connections to efficiently capture spectral dependencies across channels while enhancing spatial resolution.

---

<p align="center">
  <img src="../images/architecture.png" width="750"/>
</p>

<p align="center"><em>Architecture of S5-DSCR and its lightweight variant.</em></p>

---

<p align="center">
  <img src="../images/DSC.png" width="500"/>
</p>

<p align="center"><em>Depthwise Separable Convolution module.</em></p>

---

## Citation

```bibtex
@inproceedings{ali2025depth,
  title={Depth Separable Architecture for Sentinel-5P Super-Resolution},
  author={Ali, Hyam Omar and Abraham, Romain and Galerne, Bruno},
  booktitle={IGARSS 2025},
  year={2025}
}
