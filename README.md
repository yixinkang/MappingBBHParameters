# MappingBBHParameters
This repository accompanies the paper ["Mapping Parameter Correlations in Binary Black Hole Waveforms"](https://arxiv.org/abs/2502.17402). 


## **Contents**
### **Neural Networks for Mismatch Predictions**  
Trained neural networks for predicting waveform mismatches at **$30M_\odot, 90M_\odot, 270M_\odot$**.
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

#### Using the Trained Neural Network
The **input** to the network consists of a feature array of shape $(N, 14)$, where each row corresponds to a pair of binary black hole systems[$\boldsymbol{\lambda}_1,\boldsymbol{\lambda}_2$. 
Each **binary system** is characterized by:
$$\boldsymbol{\lambda}_i = \left\{\eta_i, \vec{\chi}_{1i}, \vec{\chi}_{2i}\right\}$$
where:
- $\eta_i$ is the **symmetric mass ratio**,
- $\vec{\chi}_{1i}$ and $\vec{\chi}_{2i}$ are the **dimensionless spin vectors** of the two black holes.
The output is a 1D array of length $N$, containing the predicted mismatch for each binary pair.

If using the trained mismatch prediction network on its own, follow these steps:
##### 1. Importing the network model
```python
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
mass = '90'  # Change to desired total mass ('lowmass', '90', or '270')
MODEL_PATH = f"mismatch_allmodes_{mass}"
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
```
##### 2. Running mismatch prediction
```python
# Define binary systems
lambda_0 = [0.25, 0, 0, 0, 0, 0, 0]  # Equal mass, nonspinning
lambda_1   = [0.1875, 0, 0, 0, 0, 0, 0]  # q=3 (eta=0.1875), nonspinning
lambda_2   = [0.25, 0, 0, 0.6, 0, 0, 0.6]  # Equal mass, spinning
lambda_3   = [0.1875, 0, 0, 0.6, 0, 0, 0.6]  # q=3, spinning
# Construct feature rows
features = np.array([
    lambda_0 + lambda_1,  # Equal mass vs. q=3 (nonspinning)
    lambda_0 + lambda_2,  # Equal mass nonspinning vs. spinning
    lambda_1 + lambda_3   # q=3, nonspinning vs. spinning
])
mismatch_prediction = model.predict(features).flatten()
print(f"Predicted mismatches: {mismatch_prediction}")
```
### **Jupyter notebooks**  
  - `torun.ipynb` Running the mapping algorithm described in the paper.  
  - `Fig7.ipynb` Generating Fig. 7 of the paper, which illustrates **2D and projected 3D** paths of parameter mappings for the **$90M_\odot$ network** (reads in saved paths generated by the mapper).
  
### **Scripts and Configuration**
- **`algo.py`**  
  Contains core functions for running the mapping algorithm and generating and computing mismatches within the parameter space.
- **`plotting.mplstyle`**  
  A Matplotlib style configuration file for figure formatting.
- **`requirements.txt`**
  Lists dependencies for required Python packages. To install dependencies, run:  
  ```bash
  pip install -r requirements.txt

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
