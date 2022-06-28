# NoisyDIRECT
NoisyDIRECT is a simulation-based robust optimization algorithm that extends the original DIRECT algorithm for noisy objective fucntions. It can automatically identify the level of simulation stochasticity for each decision vector within the search space and optimally allocate the computational resource so as to converge to the opti-mal solution in a computationally efficient and reliable manner.

The algorithm is coded as a package in the noisy_direct.py file, while the demo.py file provides some demos on test functions.

The following are example results obtained from solving the Goldstein-Price function with and without noise.

![](demo20%figures/Goldstein-Price-function.png)
