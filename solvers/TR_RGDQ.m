function [Xnew, duration, errorOmega, errorGamma, info] = ...
    TR_RGDQ(X, PA, Omega, varargin)
%TR_RGDQ Riemannian gradient descent on the TR quotient manifold.
%
% Basic interface:
%   [Xnew, duration, errorOmega] = TR_RGDQ(X, PA, Omega, opts)
%
% Interface with test statistics:
%   [Xnew, duration, errorOmega, errorGamma, info] = ...
%       TR_RGDQ(X, PA, Omega, PAGamma, Gamma, opts)
%
% Input:
%   X: initial order-d TR tensor with fields n, r and core (struct)
%   PA: observed entries on Omega (SizeOmega-by-1 vector)
%   Omega: training multi-indices (SizeOmega-by-d matrix)
%   PAGamma: optional observed entries on Gamma (SizeGamma-by-1 vector)
%   Gamma: optional test multi-indices (SizeGamma-by-d matrix)
%   opts: optional solver options (struct)
%       - maxiter: maximum number of iterations (100)
%       - maxTime: maximum running time in seconds (1000)
%       - gradtol: gradient-norm tolerance (1e-8)
%       - train_tol: relative training-error tolerance (0)
%       - tol: relative objective-change tolerance (0)
%       - verbosity: Manopt display level (2)
%       - minstepsize: minimum accepted stepsize (eps)
%
% Output:
%   Xnew: recovered TR tensor with fields n, r and core (struct)
%   duration: cumulative running time at each iteration (vector)
%   errorOmega: relative training error at each iteration (vector)
%   errorGamma: relative test error at each iteration (vector); empty when
%      PAGamma and Gamma are not supplied
%   info: Manopt iteration information (struct array)
%
% Arbitrary tensor order, mode sizes and cyclic TR ranks are supported.
% Test data are used only for statistics. Manopt is required.
%
% Reference: Quotient geometry of tensor ring decomposition,
%    Bin Gao, Renfeng Peng, and Ya-xiang Yuan,
%    arXiv preprint arXiv:2601.21874, 2026.
%    https://arxiv.org/abs/2601.21874
%
% Original author: Renfeng Peng, Aug. 05, 2026.
% Last modified: Renfeng Peng, Aug. 08, 2026.

[Xnew, duration, errorOmega, errorGamma, info] = TR_quotient_completion( ...
    'RGD', X, PA, Omega, varargin{:});
end
