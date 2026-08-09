function relativeError = TR_uniform_relativeError(point, Omega, PA, d, n, r)
%TR_UNIFORM_RELATIVE_ERROR Relative error for a uniform TR tensor.
%
% Input:
%   point: uniform quotient-manifold point with shared core U (struct)
%   Omega: evaluated multi-indices (m-by-d matrix)
%   PA: reference tensor entries on Omega (m-by-1 vector)
%   d: tensor order (integer greater than or equal to 3)
%   n: common mode size (scalar)
%   r: common TR rank (scalar)
%
% Output:
%   relativeError: norm of the residual divided by norm(PA) (scalar)
%
% Reference: Quotient geometry of tensor ring decomposition,
%    Bin Gao, Renfeng Peng, and Ya-xiang Yuan,
%    arXiv preprint arXiv:2601.21874, 2026.
%    https://arxiv.org/abs/2601.21874
%
% Original author: Renfeng Peng, Aug. 05, 2026.
%


X.d = d;
X.n = n;
X.r = r;
X.U = point.U;
Px = uTR_sample(X, Omega);

PA = PA(:);
if ~isa(PA, 'double') || ~isreal(PA) || ...
        any(~isfinite(PA)) || numel(PA) ~= size(Omega, 1)
    error('TR_uniform_relativeError:Values', ...
        'PA must be a finite real double vector matching Omega.');
end
denominator = norm(PA);
if denominator == 0
    error('TR_uniform_relativeError:ZeroReference', ...
        'The relative error is undefined because the reference norm is zero.');
end
relativeError = norm(Px - PA)/denominator;
end
