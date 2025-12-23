

# Low Latency Transformer Inference on FPGAs for Anomaly Detection on ECGs

![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![C++](https://img.shields.io/badge/C++-HLS%20Synthesis-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![HLS4ML](https://img.shields.io/badge/HLS4ML-FPGA%20Inference-8A2BE2?style=for-the-badge)
![FPGA](https://img.shields.io/badge/Target-Xilinx%20FPGA-red?style=for-the-badge)
![Vivado](https://img.shields.io/badge/Toolchain-Vivado%20HLS-EE0000?style=for-the-badge)
![Research](https://img.shields.io/badge/Domain-AI%20Accelerators-informational?style=for-the-badge)
![ECG](https://img.shields.io/badge/Application-ECG%20Anomaly%20Detection-green?style=for-the-badge)


<img width="1000" height="890" alt="Anomaly_detection_ images-Picsart-AiImageEnhancer" src="https://github.com/user-attachments/assets/0203fa87-d702-432a-b117-d10b661b2754" />

## 📌 Abstract

This project demonstrates the implementation of a **Hardware-Aware Transformer Neural Network** optimized for microsecond-scale inference latency on Field Programmable Gate Arrays (FPGAs).

Addressing the computational bottleneck of standard Transformers in edge computing, this work utilizes **High-Level Synthesis (HLS)** to convert a Python-based Keras model into a fully parallelized C++ hardware circuit. The architecture is designed to detect cardiac anomalies in the **ECG5000 dataset**, achieving **96.6% logic fidelity** between the floating-point software model and the 16-bit fixed-point hardware implementation.

---



## 🧠 Theoretical Background

### 1. The Transformer Legacy (Vaswani et al.)
The architecture draws inspiration from the seminal work *"Attention Is All You Need"* by Vaswani et al. (2017). The core innovation of the original Transformer was the **Self-Attention Mechanism**, which allows the model to weigh the importance of different time-steps in a sequence dynamically.

Mathematically, the standard Scaled Dot-Product Attention is defined as:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Where:
- $Q$ (Query), $K$ (Key), and $V$ (Value) are linear projections of the input.
- The dot product $QK^T$ computes similarity scores between all time-steps ($O(N^2)$ complexity).

---

### 2. The Low-Latency Challenge (Jiang et al.)
While powerful, the $O(N^2)$ complexity of standard attention is a bottleneck for real-time hardware. As highlighted in *"Low Latency Transformer Inference on FPGAs"* (Jiang et al.), deploying Transformers on edge devices requires:

1. **Parallelism:** Unrolling loops to execute operations simultaneously rather than sequentially.  
2. **Quantization:** Reducing 32-bit floats to lower-precision fixed-point integers to save logic gates.  
3. **Hardware-Friendly Math:** Modifying the attention mechanism to avoid computationally expensive operations while preserving dynamic feature weighting.

---

## ⚙️ Implemented Architecture: Hardware-Efficient Attention

To adhere to the low-latency principles proposed by Jiang et al., this project implements a **Channel-Wise Attention (Squeeze-and-Excitation)** mechanism. This retains the Transformer's ability to "attend" to specific features but replaces heavy matrix multiplication with efficient element-wise operations suitable for FPGA DSP slices.

### Mathematical Formulation

#### Step 1: Squeeze (Global Information Aggregation)
Temporal information is aggregated into a global descriptor using Global Average Pooling:

$$
z_c = \frac{1}{L} \sum_{i=1}^{L} x_{i,c}
$$

#### Step 2: Excite (Adaptive Weight Learning)
Channel dependencies are learned using two lightweight dense layers:

$$
s = \sigma(W_2 \delta(W_1 z))
$$

- $W_1$ performs dimensionality reduction  
- $W_2$ restores dimensionality  
- $\delta$ = ReLU, $\sigma$ = Sigmoid  

#### Step 3: Scale (Attention Application)
The final output is obtained by re-weighting the input features:

$$
\tilde{x}_c = s_c \cdot x_c
$$

**Why this fits the Jiang et al. objective:**
- **Latency:** Multiply and Add operations are fully parallelizable  
- **Resources:** Eliminates the need for an $N \times N$ attention matrix  

---

<img width="800" height="800" alt="Anomaly_detection_ images-Picsart-AiImageEnhancer" src="https://github.com/user-attachments/assets/09f7285b-f281-4e4d-95b3-a797941bc207" />

---

## 🛠️ FPGA Implementation Details

The transition from software to hardware is handled by **hls4ml** using the following configuration:

### 1. Quantization (`ap_fixed<16,6>`)
All weights and activations are converted from 32-bit floating point to **16-bit fixed-point** representation.

- **Software:** `float32` (high precision, high cost)  
- **Hardware:** `ap_fixed<16,6>` (lower precision, high efficiency)  

---

### 2. Parallelization Strategy
To achieve low-latency inference, the **Latency Strategy** is applied to all layers:

config['LayerName'][layer]['Strategy'] = 'Latency'
config['LayerName'][layer]['ReuseFactor'] = 1 

## ⚙️ Parallelization Strategy

**ReuseFactor = 1** forces full parallelism by allocating dedicated hardware resources for each operation.  
This eliminates time-multiplexing and minimizes inference latency at the cost of increased FPGA resource usage.

---

## 📊 Results

The hardware logic was validated against the floating-point software model using the **ECG5000 dataset**.

| Metric | Value | Description |
|------|------|------------|
| Software Accuracy | **91.44%** | TensorFlow/Keras float32 model |
| Hardware Accuracy | **100%*** | Bit-accurate C++ simulation |
| Logic Match Rate | **96.60%** | Hardware vs software prediction fidelity |

\*Hardware accuracy measured on the validation subset.

---

## 📂 Repository Structure

- `hls_ecg_final_v2/` – Synthesized HLS C++ project  
- `Project_Notebook.ipynb` – Training, quantization, and synthesis pipeline  
- `exports_ecg/` – Saved models and weights  
- `figures/` – Accuracy curves, confusion matrices, and analysis plots  

---

## 🚀 How to Reproduce

### 1. Clone the Repository
```
git clone https://github.com/niteshg97/Low-Latency-Transformer-Inference-on-FPGAs-for-Anomaly-Detection-on-ECGs.git
```
## 🧰 Environment Setup

Install the required dependencies to support legacy Keras parsing and FPGA synthesis:

- pip install tensorflow tf_keras hls4ml[profiling].
- Run the Notebook.
- Open Project_Notebook.ipynb in Google Colab or Jupyter Notebook.
- Make sure the ECG5000 dataset is available in the expected directory before execution.

## 📜 References

Core Research
- Vaswani et al. – Attention Is All You Need, NeurIPS 2017.
- Jiang et al. – Low Latency Transformer Inference on FPGAs, FPGA 2021.
- Hu et al. – Squeeze-and-Excitation Networks, CVPR 2018.

## Tools
``` hls4ml: https://fastmachinelearning.org/hls4ml ``` .
- Xilinx Vivado HLS .
