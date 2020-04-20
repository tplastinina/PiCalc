# PiCalc

# How to execute

Before start the app please follow this [guide](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)

---
Note: you need to have compatible GPU, [list](https://developer.nvidia.com/cuda-gpus) of allowed GPUs
---

# Results

The results were tested on GeForce 840M.

| Length of random vectors   | CPU, ms       | CPU, result | GPU, ms  | CPU, result |  
| -------------              |:-------------:| -----:      | -----:   | -----:      |
|  1048576                   | 0.001000      | 3.139828    | 0.000824 | 3.139828    |  
| 16777216                   | 0.170000      | 3.141407    | 0.010149 | 3.141407    |
| 33554432                   | 0.30000       | 3.141263    | 0.019768 | 3.141263    |
