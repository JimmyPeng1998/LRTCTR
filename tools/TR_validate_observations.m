function values = TR_validate_observations(indices, values, n, ...
    allowEmpty, label)
%TR_VALIDATE_OBSERVATIONS Validate sampled indices and matching values.
%
% Input:
%   indices: sampled multi-indices (m-by-d matrix)
%   values: sampled values (m-by-1 vector), or empty when permitted
%   n: tensor mode sizes (1-by-d vector)
%   allowEmpty: whether empty indices or values are permitted (logical)
%   label: label used in validation error messages (character vector)
%
% Output:
%   values: validated sampled values as a column vector
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


if nargin < 4
    allowEmpty = false;
end
if nargin < 5
    label = 'Observation';
end
d = numel(n);
if ~isnumeric(indices) || ~isreal(indices) || ~ismatrix(indices) || ...
        size(indices, 2) ~= d || any(~isfinite(indices(:))) || ...
        any(indices(:) < 1) || any(indices(:) ~= floor(indices(:)))
    error('LRTCTR:Indices', ...
        '%s indices must be a real integer matrix with d columns.', label);
end
if ~allowEmpty && isempty(indices)
    error('LRTCTR:EmptyObservations', '%s set must not be empty.', label);
end
for k = 1:d
    if any(indices(:, k) > n(k))
        error('LRTCTR:Indices', ...
            '%s indices contain an entry outside mode %d.', label, k);
    end
end

if isempty(values) && allowEmpty
    values = zeros(0, 1);
    return
end
if ~isnumeric(values) || ~isreal(values) || ~isvector(values) || ...
        numel(values) ~= size(indices, 1) || any(~isfinite(values(:)))
    error('LRTCTR:ObservedValues', ...
        '%s values must be a finite real vector matching the index rows.', label);
end
values = double(values(:));
end
