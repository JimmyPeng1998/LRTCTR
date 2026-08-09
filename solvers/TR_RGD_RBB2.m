function [Xnew, duration, errorOmega, errorGamma] = TR_RGD_RBB2(X, PA, Omega, varargin)
%TR_RGD_RBB2 Preconditioned TR-RGD with the RBB2 stepsize.
%
% Simplified interface:
%   [Xnew, duration, errorOmega] = ...
%       TR_RGD_RBB2(X, PA, Omega, opts)
%
% Simplified interface with test statistics:
%   [Xnew, duration, errorOmega, errorGamma] = ...
%       TR_RGD_RBB2(X, PA, Omega, PAGamma, Gamma, opts)
%
% Legacy interface:
%   [Xnew, duration, errorOmega, errorGamma] = ...
%       TR_RGD_RBB2(X, PA, Omega, SizeOmega, PAGamma, Gamma, ...
%       SizeGamma, p, opts)
%
% Input:
%   X: initial TR tensor (struct)
%   PA: observed tensor entries on Omega (SizeOmega-by-1 vector)
%   Omega: training multi-indices (SizeOmega-by-d matrix)
%   PAGamma: optional tensor entries on Gamma (SizeGamma-by-1 vector)
%   Gamma: optional test multi-indices (SizeGamma-by-d matrix)
%   opts: user-defined options (struct)
%       - maxiter: maximum number of iterations (100)
%       - maxTime: maximum running time in seconds (1000)
%       - gradtol: tolerance for the Riemannian gradient norm (1e-8)
%       - train_tol: tolerance for the relative training error (0)
%       - tol: tolerance for the relative change in the objective (1e-6)
%       - lambda: core regularization parameter (1e-10)
%   SizeOmega: number of training entries in the legacy interface (scalar)
%   SizeGamma: number of test entries in the legacy interface (scalar)
%   p: sampling ratio SizeOmega/prod(X.n) in the legacy interface (scalar)
%
% Output:
%   Xnew: recovered TR tensor (struct)
%   duration: cumulative running time at each iteration (vector)
%   errorOmega: relative training error at each iteration (vector)
%   errorGamma: relative test error at each iteration (vector); empty when
%      PAGamma and Gamma are not supplied
%
% In the simplified interfaces, SizeOmega, SizeGamma and p are computed
% automatically. PAGamma and Gamma are used
% only to record test statistics and do not participate in the tensor update
% or stopping criteria. The legacy interface retains its original opts.err
% stopping behavior.
%
% Reference: Riemannian preconditioned algorithms for tensor completion via
%    tensor ring decomposition,
%    Bin Gao, Renfeng Peng, and Ya-xiang Yuan,
%    Computational Optimization and Applications, 88(2):443--468, 2024.
%    https://doi.org/10.1007/s10589-024-00559-7
%
% Original author: Renfeng Peng, Jul. 05, 2023.
% Last modified: Renfeng Peng, Aug. 08, 2026.

[Xnew, duration, errorOmega, errorGamma] = TR_compat_dispatch( ...
    @TR_RGD_RBB2_legacy, X, PA, Omega, varargin{:});
end
