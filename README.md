# MappingBBHParameters
This repository accompanies the paper ["Mapping Parameter Correlations in Binary Black Hole Waveforms"](https://arxiv.org/abs/2502.17402). 


## **Contents**
### **Neural networks** trained for mismatch predictions at $30M_\odot, 90M_\odot, 270M_\odot$.
  - Networks are trained within NRSur7dq4 region of validity:
    - **Mass ratio constraint**: $q \leq 6$ (equivalently $\eta > 6/49 \approx 0.12$).
    - **Spin constraints**: $\chi_i < 1$, $\theta_i \in [0, \pi]$, $\phi_i \in [0, 2\pi]$.
  - The <small>NRSur7dq4<small> surrogate model is used for training the **90M⊙** and **270M⊙** networks.
  - The phenomenological <small>IMRPhenomXPHM<small> waveform model is used for training the **30M⊙** lowmass network.
 
For details regarding the **neural network architecture**, training methodology, and implementation, please refer to:  
[Ferguson 2022 (arXiv:2209.15144)](https://arxiv.org/abs/2209.15144).

The trained models are stored in the following folders:
- **`mismatch_allmodes_lowmass/`** → 30M⊙ network  
- **`mismatch_allmodes_90/`** → 90M⊙ network  
- **`mismatch_allmodes_270/`** → 270M⊙ network  
    
### **Jupyter notebooks**  
  - `torun.ipynb` Running the mapping algorithm described in the paper.  
  - `Fig7.ipynb` Generating Fig. 7 of the paper, which illustrates **2D and projected 3D** paths of parameter mappings for the **$90M_\odot$ network** (reads in saved paths generated by the mapper).
  
### **Scripts and Configuration**
- **`algo.py`**  
  Contains core functions for running the mapping algorithm and generating and computing mismatches within the parameter space.
- **`plotting.mplstyle`**  
  A Matplotlib style configuration file for figure formatting.

### **Data Files**
- **`mapped_paths_2D.csv`**  and  **`mapped_paths_3D.csv`**  
  Precomputed **2D**  and **3D** degeneracy paths generated by the mapper for the **90M⊙** system, used for running `Fig7.ipynb`

## **Citation**
If you use this repository, please cite:
*"Mapping Parameter Correlations in Spinning Binary Black Hole Mergers"*  
arXiv preprint [arXiv:2502.17402](https://arxiv.org/abs/2502.17402), 2025.

```bibtex
@article{Kang:2025nio,
    author = "Kang, Karen and Miller, Simona J. and Chatziioannou, Katerina and Ferguson, Deborah",
    title = "{Mapping Parameter Correlations in Spinning Binary Black Hole Mergers}",
    eprint = "2502.17402",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    month = "2",
    year = "2025"
}
