# Low-Latency-Transformer-Inference-on-FPGAs-for-Anomaly-Detection-on-ECGs
This project successfully implements a Hardware-Aware Neural Network for detecting cardiac anomalies in ECG signals, optimized for Low Latency Inference on FPGAs (Field Programmable Gate Arrays).

Hardware-Friendly Architecture: Implemented a Channel Attention (Squeeze-and-Excitation) mechanism instead of standard Multi-Head Attention. This novel approach avoids expensive matrix multiplications ($O(N^2)$), replacing them with FPGA-efficient element-wise operations (Multiply, Add) while maintaining high classification accuracy.

Low Latency Optimization: Leveraged High-Level Synthesis (HLS) to fully unroll neural network loops (Strategy='Latency', ReuseFactor=1). This forces the FPGA to execute operations in massive parallelism rather than sequentially, achieving microsecond-scale inference potential.

Precision Quantization: Successfully compressed the 32-bit floating-point model into 16-bit Fixed Point (ap_fixed<16,6>) logic. This reduced memory usage by 50% without significant loss in accuracy, making the model deployable on edge devices (e.g., Xilinx PYNQ-Z2).

Validated the hardware logic using C-Simulation, achieving a 96.6% Logic Match between the Python software model and the synthesized C++ hardware core.
