function [d, n, r] = TR_validate_tensor(X, requireInjective)
%TR_VALIDATE_TENSOR Validate the public LRTCTR core-cell representation.
%
% Input:
%   X: TR tensor with fields d, n, r and core (struct)
%   requireInjective: whether to enforce n(k) >= r(k)r(k+1) (logical)
%
% Output:
%   d: validated tensor order (scalar)
%   n: validated mode sizes (1-by-d double vector)
%   r: validated cyclic ranks (1-by-(d+1) double vector)
%
% References:
%   [1] Riemannian preconditioned algorithms for tensor completion via
%       tensor ring decomposition,
%       Bin Gao, Renfeng Peng, and Ya-xiang Yuan,
%       Computational Optimization and Applications, 88(2):443--468, 2024.
%       https://doi.org/10.1007/s10589-024-00559-7
%   [2] Optimization on Product Manifolds under a Preconditioned Metric,
%       Bin Gao, Renfeng Peng, and Ya-xiang Yuan,
%       SIAM Journal on Matrix Analysis and Applications,
%       46(3):1816--1845, 2025.
%       https://doi.org/10.1137/24M1643773
%   [3] Quotient geometry of tensor ring decomposition,
%       Bin Gao, Renfeng Peng, and Ya-xiang Yuan,
%       arXiv preprint arXiv:2601.21874, 2026.
%       https://arxiv.org/abs/2601.21874
%
% Original author: Renfeng Peng, Aug. 05, 2026.
%


if nargin < 2
    requireInjective = false;
end
if ~isstruct(X) || ~isscalar(X) || ...
        ~all(isfield(X, {'d', 'n', 'r', 'core'}))
    error('LRTCTR:TensorFormat', ...
        'X must be a scalar structure with fields d, n, r and core.');
end
[d, n, r] = TR_validate_dimensions(X.n, X.r, requireInjective);
if ~isnumeric(X.d) || ~isscalar(X.d) || ~isreal(X.d) || ...
        ~isfinite(X.d) || X.d ~= d
    error('LRTCTR:TensorFormat', 'X.d must equal the number of mode sizes.');
end
if ~iscell(X.core) || numel(X.core) ~= d
    error('LRTCTR:TensorFormat', 'X.core must contain exactly d TR cores.');
end
for k = 1:d
    core = X.core{k};
    if ~isa(core, 'double') || ~isreal(core) || ...
            size(core, 1) ~= r(k) || size(core, 2) ~= r(k+1) || ...
            size(core, 3) ~= n(k) || numel(core) ~= r(k)*r(k+1)*n(k)
        error('LRTCTR:CoreSize', ...
            'Core %d must be a real double array of size %d-by-%d-by-%d.', ...
            k, r(k), r(k+1), n(k));
    end
end
end
