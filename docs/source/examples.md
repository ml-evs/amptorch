# Examples
## Gaussian Multiple (GMP) Descriptors
In addition to conventional Atom-centered Symmetry Functions as fingerprinting scheme, AmpTorch also support GMP descriptors that uses multipole expansions to describe the reconstructed electronic density around every central atom and its neighbors to encode local environments. Because the formulation of symmetry functions does not take into element types into account, the interactions among different elements are divided into different columns as input. As a result, the number of feature dimensions undesirably increases with the number of elements present. A major advantage of GMPs is that the input dimensions remain constant regardless of the number of chemical elements, and therefore can be adopted for complex datasets. For more technical details and theorical backgrounds, please refer to *Lei, X., & Medford, A. J. (2021). A Universal Framework for Featurization of Atomistic Systems. http://arxiv.org/abs/2102.02390*

For an example script of using GMP, please refer to:
```
examples/GMP/GMP_example.py
```
