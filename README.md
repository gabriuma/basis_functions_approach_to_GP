# Practical Hilbert space approximate Gaussian processes for probabilistic programming
Approximate eigendecomposition of the covariance function of a Gaussian process. The method is based on interpreting the covariance function as the kernel of a pseudo-differential operator and approximating it using Hilbert space methods. This results in a reduced-rank approximation for the covariance function (Solin and Särkkä, 2020).
# Contents in the repository
The repo contains three folders 'Paper' 'multi_dimensional' and 'unidimensional'. 
'multi_dimensional' and 'uni:dimensional' folders containts the Stan codes of some models, such as regular Gaussian process model, Hilber space approximate GP model and spline-based model. There are unidimensional and multidimensional versions of these code, and they are particularized to different datasets.
'Paper' folder contains the main mamuscript work for this project, with supplementary material with more case studies. Stan codes for the case studies are also provided.
