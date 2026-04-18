
---

<h1 align="center">
  <a href="https://arxiv.org/abs/2501.17210">
    Self-Supervised Super-Resolution for Sentinel-5P Hyperspectral Images
  </a>
</h1>


<h3 align="center">
Hyam Omar Ali¹ ², Antoine Crosnier¹ ³, Romain Abraham¹, Baptiste Combelles⁴, Fabrice Jégou⁴, Bruno Galerne¹ ⁵
</h3>

<p align="center">¹ Université d’Orléans, Université de Tours, CNRS, Institut Denis Poisson (IDP), UMR 7013, France</p>  
<p align="center">² Faculty of Mathematical Sciences, University of Khartoum, Sudan</p>  
<p align="center">³ ENS Lyon, France</p>  
<p align="center">⁴ Laboratory of Physics and Chemistry of the Environment and Space (LPC2E), CNRS UMR 7328, University of Orléans, France</p>  
<p align="center">⁵ Institut Universitaire de France (IUF), France</p>


---

Sentinel-5P (S5P) plays a critical role in atmospheric and environmental monitoring. However, its limited spatial resolution restricts its use for fine-scale analysis of localised emission sources.

Existing super-resolution approaches for S5P rely on supervised learning with synthetic degradation. Since true high-resolution data do not exist, these approaches remain limited in real-world applications.

This work introduces a self-supervised hyperspectral super-resolution framework specifically designed for Sentinel-5P, enabling training without high-resolution ground truth.

---

## Key Contributions

- We introduce novel U-Net–based architectures tailored for Sentinel-5P hyperspectral data, incorporating Depthwise Separable Convolution (DSC) to improve parameter efficiency while preserving spectral fidelity.  

- We adopt a self-supervised super-resolution framework based on SURE and Equivariant Imaging, and propose a **sensor-aware noise model** derived from Sentinel-5P Signal-to-Noise Ratio (SNR) metadata, enabling the loss function to adapt to band-specific noise characteristics.  

- We demonstrate that the proposed framework achieves performance comparable to supervised learning while enabling super-resolution directly from real observations at their native resolution. The method consistently outperforms bicubic interpolation and is further validated using coincident EMIT hyperspectral data.  

---

## Methodology

We propose a self-supervised hyperspectral super-resolution framework for Sentinel-5P that learns spatial enhancement directly from degraded observations, without requiring high-resolution ground truth.

The approach relies on a physics-based degradation model describing the S5P acquisition process, combined with sensor-aware noise estimation derived from Signal-to-Noise Ratio (SNR) metadata. This enables realistic modelling of measurement noise across spectral bands.

Training is performed using a composite self-supervised loss that combines:

- **Stein’s Unbiased Risk Estimator (SURE):** enforces consistency with the observed measurements  
- **Equivariant Imaging constraint:** ensures stability of the reconstruction under spatial scaling  

Together, these components allow the model to learn meaningful spatial structures directly from real observations.

---

### Network Architecture

We introduce U-Net–based architectures tailored for hyperspectral data, incorporating Depthwise Separable Convolutions (DSC) for efficient feature extraction and improved spectral fidelity.

<p align="center">
  <img src="../images/global.png" width="750"/>
</p>

<p align="center">
  <img src="../images/unet_v2.png" width="750"/>
</p>

The model follows a residual learning strategy, refining bicubic upsampled inputs to recover high-frequency spatial details.

---

## Results

The proposed approach achieves:
- Performance comparable to the counterparts of the supervised methods 
- Enhanced spatial detail and sharper boundaries compared to bicubic interpolation  
- Consistent reconstruction across spectral bands  
- Physically meaningful super-resolution validated using independent EMIT observations 


### Qualitative Evaluation (GT–SHR Setting)

<p align="center">
  <img src="../images/GT_SHR.png" width="900"/>
</p>

<p align="center">
<em>Qualitative comparison across multiple spectral bands. The proposed method produces sharper spatial details and consistent reconstruction across bands compared to bicubic interpolation.</em>
</p>

---

### Cross-Sensor Validation (EMIT vs S5P)

<p align="center">
  <img src="../images/EMIT.png" width="700"/>
</p>

<p align="center">
<em>Comparison with EMIT hyperspectral data demonstrates that reconstructed structures are physically meaningful rather than hallucinated.</em>
</p>
---

## Citation

(To be updated with arXiv reference)
