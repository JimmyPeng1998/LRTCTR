function [Xnew, duration, errorOmega, errorGamma] = TR_RGN(X, PA, Omega, varargin)
%TR_RGN Riemannian Gauss--Newton method for TR completion.
%
% Simplified interface:
%   [Xnew, duration, errorOmega] = ...
%       TR_RGN(X, PA, Omega, opts)
%
% Simplified interface with test statistics:
%   [Xnew, duration, errorOmega, errorGamma] = ...
%       TR_RGN(X, PA, Omega, PAGamma, Gamma, opts)
%
% Legacy interface:
%   [Xnew, duration, errorOmega, errorGamma] = ...
%       TR_RGN(X, PA, Omega, SizeOmega, PAGamma, Gamma, ...
%       SizeGamma, p, opts)
%
% Input:
%   X: initial third-order TR tensor (struct)
%   PA: observed tensor entries on Omega (SizeOmega-by-1 vector)
%   Omega: training multi-indices (SizeOmega-by-3 matrix)
%   PAGamma: optional tensor entries on Gamma (SizeGamma-by-1 vector)
%   Gamma: optional test multi-indices (SizeGamma-by-3 matrix)
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
% TR_RGN currently supports third-order tensors only. In the simplified
% interfaces, SizeOmega, SizeGamma and p are computed automatically.
% In the simplified interfaces, PAGamma and Gamma are used only to record
% test statistics and do not participate in the tensor update or stopping
% criteria. The legacy interface retains its original opts.err stopping
% behavior.
%
% Reference: Optimization on Product Manifolds under a Preconditioned Metric,
%    Bin Gao, Renfeng Peng, and Ya-xiang Yuan,
%    SIAM Journal on Matrix Analysis and Applications, 46(3):1816--1845, 2025.
%    https://doi.org/10.1137/24M1643773
%
% Original author: Renfeng Peng, Jul. 18, 2024.
% Last modified: Renfeng Peng, Aug. 08, 2026.

[Xnew, duration, errorOmega, errorGamma] = TR_compat_dispatch( ...
    @TR_RGN_legacy, X, PA, Omega, varargin{:});
end
