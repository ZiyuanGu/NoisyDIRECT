# NoisyDIRECT
NoisyDIRECT is a simulation-based robust optimization algorithm that extends the original DIRECT algorithm for noisy objective fucntions. It can automatically identify the level of simulation stochasticity for each decision vector within the search space and optimally allocate the computational resource so as to converge to the opti-mal solution in a computationally efficient and reliable manner.

The algorithm is coded as a package in the noisy_direct.py file while the demo.py file provides some demos on test functions.

The following are example results obtained from solving the Goldstein-Price function with and without noise.

<p float="left">
  <img src="demo figures/Goldstein-Price-function.png" height="300">
  <img src="demo figures/Goldstein-Price-function-DIRECT.png" height="300">
  <img src="demo figures/Goldstein-Price-function-NoisyDIRECT.png" height="300">
</p>

# Some references
Gu, Z., Li, Y., Saberi, M., Rashidi, T.H., Liu, Z., 2023. Macroscopic parking dynamics and equitable pricing: Integrating trip-based modeling with simulation-based robust optimization. Transp. Res. Part B 173, 354-381.
Jones, D.R., Perttunen, C.D., Stuckman, B.E., 1993. Lipschitzian optimization without the Lipschitz constant. J. Optim. Theory Appl. 79(1), 157-181.  
Deng, G., Ferris, M.C., 2007. Extension of the DIRECT optimization algorithm for noisy functions, 2007 Winter Simulation Conference. IEEE, Washington, DC, pp. 497-504.  
Deng, G., Ferris, M.C., 2006. Adaptation of the UOBYQA Algorithm for Noisy Functions, Proceedings of the 2006 Winter Simulation Conference. IEEE, Monterey, CA, pp. 312-319.
