# LRTCTR

MATLAB algorithms for low-rank tensor completion in tensor ring (TR)
decomposition.

The package includes the Riemannian preconditioned algorithms and
quotient-manifold algorithms for general and uniform TR decompositions.

## Quick Start

```matlab
install
rng(1)
n = [20 20 20]; d = 3; r = [2 2 2 2];
Xtrue = TR_randn(n, d, r);
Omega = makeOmegaSet_mod(n, 800);
PA = TR_sample(Xtrue, Omega);
X0 = TR_randn(n, d, r);
opts.maxiter = 10;
opts.verbosity = 1;
[X, duration, errorOmega] = TR_RGDQ(X0, PA, Omega, opts);
```

## Installation

Run `install.m` from MATLAB. The script:

1. adds the LRTCTR directories to the MATLAB path;
2. loads the bundled Manopt 7.1.0 package;
3. compiles the MEX routines using the C compiler configured in MATLAB.

## Algorithms

Riemannian preconditioned TR completion algorithms [1]:

- `TR_RGD_exact`: Riemannian preconditioned gradient descent method with exact line search
- `TR_RGD_Armijo`: Riemannian preconditioned gradient descent method with Armijo backtracking line search
- `TR_RGD_RBB2`: Riemannian preconditioned gradient descent method with Riemannian Barzilai--Borwein stepsize
- `TR_RCG_HS`: Riemannian conjugate gradient method with Hestenes--Stiefel rule

Riemannian Gauss--Newton under a preconditioned metric [2]:

- `TR_RGN`: Riemannian Gauss--Newton method

Quotient-manifold algorithms [3]:

- `TR_RGDQ`: TR-RGD(Q), Riemannian gradient descent method on the quotient manifold
- `TR_RCGQ`: TR-RCG(Q), Riemannian conjugate gradient method on the quotient manifold
- `uTR_RGDQ`: uTR-RGD(Q), Riemannian gradient descent method on the quotient manifold for uniform TR decomposition
- `uTR_RCGQ`: uTR-RCG(Q), Riemannian conjugate gradient method on the quotient manifold for uniform TR decomposition

| Algorithms | Tensor order | TR format |
| --- | --- | --- |
| `TR_RGD_exact`, `TR_RGD_Armijo`, `TR_RGD_RBB2`, `TR_RCG_HS` | 3 or 4 | General TR |
| `TR_RGN` | 3 | General TR |
| `TR_RGDQ`, `TR_RCGQ` | General | General TR |
| `uTR_RGDQ`, `uTR_RCGQ` | General (`d >= 3`) | Uniform TR with a shared core |


## Examples

- `Test_Synthetic_Noiseless.m`: preconditioned algorithms with noiseless data
- `Test_Synthetic_Noisy.m`: preconditioned algorithms with noisy data
- `Test_Synthetic_Quotient.m`: tests for TR-RGD(Q) and TR-RCG(Q), third-order
- `Test_Synthetic_General_Quotient.m`: fourth- and fifth-order TR-RGD(Q)
  and TR-RCG(Q), including nonuniform dimensions and ranks
- `Test_Synthetic_Uniform_Quotient.m`: uTR-RGD(Q) and uTR-RCG(Q)

Run `install.m` once, then run an example from the `examples` directory.

## Paper Experiments

Scripts for reproducing the numerical experiments in the TR quotient paper
are provided in [`TR_Quotient_Exps`](TR_Quotient_Exps). See the README in
that directory for experiment descriptions, runtime notes, and data setup.

## References

1. Riemannian preconditioned algorithms for tensor completion via tensor
   ring decomposition, Bin Gao, Renfeng Peng, and Ya-xiang Yuan,
   *Computational Optimization and Applications*, 88(2):443--468, 2024.
   [https://doi.org/10.1007/s10589-024-00559-7](https://doi.org/10.1007/s10589-024-00559-7)
2. Optimization on Product Manifolds under a Preconditioned Metric,
   Bin Gao, Renfeng Peng, and Ya-xiang Yuan,
   *SIAM Journal on Matrix Analysis and Applications*,
   46(3):1816--1845, 2025.
   [https://doi.org/10.1137/24M1643773](https://doi.org/10.1137/24M1643773)
3. Quotient geometry of tensor ring decomposition,
   Bin Gao, Renfeng Peng, and Ya-xiang Yuan,
   arXiv preprint arXiv:2601.21874, 2026.
   [https://arxiv.org/abs/2601.21874](https://arxiv.org/abs/2601.21874)

## Authors

- [Bin Gao](https://gaobin.cc)
- [Renfeng Peng](https://jimmypeng1998.github.io)
- [Ya-xiang Yuan](https://lsec.cc.ac.cn/~yyx/index.html)

LRTCTR is distributed under the GNU General Public License version 3 or
later. The bundled Manopt package retains its own license files in
`manopt/`.
