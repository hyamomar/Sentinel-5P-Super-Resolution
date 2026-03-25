
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
