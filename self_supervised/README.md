```markdown
<h1 align="center">
  Self-Supervised Super-Resolution for Sentinel-5P Hyperspectral Images
</h1>

<h3 align="center">Hyam Omar Ali, Romain Abraham, Bruno Galerne</h3>

<p align="center"><strong>Available on arXiv</strong></p>

---

Sentinel-5P (S5P) plays a critical role in atmospheric and environmental monitoring. However, its limited spatial resolution restricts its use for fine-scale analysis of localised emission sources.

Existing super-resolution approaches for S5P rely on supervised learning with synthetic degradation. Since true high-resolution data do not exist, these approaches remain limited in real-world applications.

This work introduces a **self-supervised hyperspectral super-resolution framework** specifically designed for Sentinel-5P, enabling training without high-resolution ground truth.

---

## Abstract

Sentinel-5P (S5P) mission plays a critical role in atmospheric and environmental monitoring; however, its spatial resolution often limits its utility for fine-scale analysis of localised emission sources. Existing super-resolution (SR) approaches for S5P rely on supervised learning. Because true high-resolution (HR) data do not exist for S5P, these methods depend on synthetic low-resolution (LR) data, limiting their applicability to real observations.

In this study, we propose a self-supervised hyperspectral SR framework tailored specifically for S5P that enables training without HR ground-truth. Our proposed framework employs a composite self-supervised loss that combines Stein’s Unbiased Risk Estimator with an Equivariant Imaging constraint, explicitly incorporating the S5P degradation operator and band-specific noise statistics derived from sensor Signal-to-Noise Ratio metadata.

In addition, we introduce Depthwise Separable Convolution U-Net architectures designed to maximise efficiency and spectral fidelity for hyperspectral data. The framework is evaluated in two settings: (i) LR-HR, where self-supervised learning is directly compared to supervised methods, and (ii) GT-SHR, where HR is unavailable and evaluation relies on non-reference metrics and qualitative analysis.

Quantitative results demonstrate that self-supervised models achieve performance comparable to supervised counterparts while maintaining strong consistency across spectral bands. Qualitative results show enhanced spatial detail and sharper structures compared to bicubic interpolation. Additional validation using coincident EMIT hyperspectral data confirms that the reconstructed outputs recover meaningful spatial structures rather than hallucinated features.

---

## Key Contributions

- Self-supervised SR framework for Sentinel-5P  
- Composite loss combining SURE and equivariant imaging  
- Integration of sensor noise statistics (SNR-based)  
- Efficient DSC-based U-Net architectures  
- Evaluation without ground-truth HR  

---

## Results

The proposed approach achieves:
- Performance comparable to supervised methods  
- Improved spatial sharpness over bicubic interpolation  
- Physically meaningful reconstruction validated with EMIT data  

---

## Citation

(To be updated with arXiv reference)
