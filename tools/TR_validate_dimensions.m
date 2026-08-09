function [d, n, r] = TR_validate_dimensions(n, r, requireInjective)
%TR_VALIDATE_DIMENSIONS Validate mode sizes and cyclic TR ranks.
%
% Input:
%   n: tensor mode sizes (vector)
%   r: TR ranks with d or d+1 entries (vector)
%   requireInjective: whether to enforce n(k) >= r(k)r(k+1) (logical)
%
% Output:
%   d: tensor order (scalar)
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


if nargin < 3
    requireInjective = false;
end
if ~isnumeric(n) || ~isreal(n) || ~isvector(n) || numel(n) < 2 || ...
        any(~isfinite(n)) || any(n < 1) || any(n ~= floor(n))
    error('LRTCTR:ModeSizes', ...
        'n must contain at least two positive integer mode sizes.');
end
n = double(n(:)');
d = numel(n);

if ~isnumeric(r) || ~isreal(r) || ~isvector(r) || ...
        any(~isfinite(r)) || any(r < 1) || any(r ~= floor(r))
    error('LRTCTR:Ranks', 'TR ranks must be positive integers.');
end
r = double(r(:)');
if numel(r) == d
    r(d+1) = r(1);
elseif numel(r) ~= d+1 || r(d+1) ~= r(1)
    error('LRTCTR:Ranks', ...
        'r must have d entries, or d+1 entries with r(d+1)=r(1).');
end

if requireInjective && any(n < r(1:d).*r(2:d+1))
    error('LRTCTR:InjectivityDimensions', ...
        ['Injective TR cores require n(k) >= r(k)*r(k+1) ', ...
         'for every k.']);
end
end
