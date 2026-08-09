function [cost, eucGrad] = TR_uniform_costgrad(point, Omega, PA, p, d, n, r)
%TR_UNIFORM_COSTGRAD Completion cost and Euclidean gradient for uniform TR.
%
% Input:
%   point: uniform quotient-manifold point with shared core U (struct)
%   Omega: observed multi-indices (m-by-d matrix)
%   PA: observed tensor entries on Omega (m-by-1 vector)
%   p: sampling ratio used to scale the objective (scalar)
%   d: tensor order (integer greater than or equal to 3)
%   n: common mode size (scalar)
%   r: common TR rank (scalar)
%
% Output:
%   cost: sampled least-squares objective value (scalar)
%   eucGrad: Euclidean gradient with respect to the shared core (struct)
%
% The shared core occurs at all d positions. Consequently, the Euclidean
% gradient is the sum of the position-wise gradients returned by the
% general-order TR MEX routine.
%
% Reference: Quotient geometry of tensor ring decomposition,
%    Bin Gao, Renfeng Peng, and Ya-xiang Yuan,
%    arXiv preprint arXiv:2601.21874, 2026.
%    https://arxiv.org/abs/2601.21874
%
% Original author: Renfeng Peng, Aug. 05, 2026.
%


if ~isscalar(d) || ~isreal(d) || ~isfinite(d) || d < 3 || d ~= floor(d)
    error('TR_uniform_costgrad:Order', ...
        'd must be an integer greater than or equal to three.');
end
if ~isscalar(n) || ~isreal(n) || ~isfinite(n) || n < 1 || n ~= floor(n)
    error('TR_uniform_costgrad:ModeSize', ...
        'n must be a positive integer scalar.');
end
if ~isscalar(r) || ~isreal(r) || ~isfinite(r) || r < 1 || r ~= floor(r)
    error('TR_uniform_costgrad:Rank', ...
        'r must be a positive integer scalar.');
end
if ~isstruct(point) || ~isscalar(point) || ~isfield(point, 'U') || ...
        ~isa(point.U, 'double') || ~isreal(point.U) || ...
        size(point.U, 1) ~= r || size(point.U, 2) ~= n || ...
        size(point.U, 3) ~= r || numel(point.U) ~= r*n*r
    error('TR_uniform_costgrad:Point', ...
        'point.U must be a real double array of size r-by-n-by-r.');
end
if ~isnumeric(Omega) || ~isreal(Omega) || ~ismatrix(Omega) || ...
        size(Omega, 2) ~= d || isempty(Omega) || ...
        any(~isfinite(Omega(:))) || any(Omega(:) < 1) || ...
        any(Omega(:) ~= floor(Omega(:))) || any(Omega(:) > n)
    error('TR_uniform_costgrad:Indices', ...
        'Omega must contain valid tensor indices in an m-by-d matrix.');
end
PA = PA(:);
if ~isa(PA, 'double') || ~isreal(PA) || ...
        any(~isfinite(PA)) || numel(PA) ~= size(Omega, 1)
    error('TR_uniform_costgrad:Values', ...
        'PA must be a finite real double vector matching Omega.');
end
if ~isscalar(p) || ~isreal(p) || ~isfinite(p) || p <= 0
    error('TR_uniform_costgrad:SamplingRatio', ...
        'p must be a positive finite scalar.');
end

SizeOmega = size(Omega, 1);
indices = reshape(uint32(Omega'), [], 1);
core = reshape(permute(point.U, [2 1 3]), [n, r*r])';
dims = uint32(n*ones(1, d));
ranks = uint32(r*ones(1, d+1));

if d == 3
    G = cell(3, 1);
    [Px, G{1}, G{2}, G{3}] = ComputeGradsAndPx_mex( ...
        3, dims, ranks, core(:), core(:), core(:), ...
        uint32(SizeOmega), indices, p, PA);
    eucGrad.U = permute(reshape(G{1}, [n, r, r]), [2 1 3]) + ...
              permute(reshape(G{2}, [n, r, r]), [2 1 3]) + ...
              permute(reshape(G{3}, [n, r, r]), [2 1 3]);
else
    if exist('ComputeGradsAndPxGeneral_mex', 'file') ~= 3
        error('TR_uniform_costgrad:GeneralMexNotFound', ...
            ['ComputeGradsAndPxGeneral_mex is required when d > 3. ', ...
             'Run install.m to compile the general-order MEX routine.']);
    end
    coreVecs = repmat({core(:)}, d, 1);
    mexOutputs = cell(d+1, 1);
    [mexOutputs{:}] = ComputeGradsAndPxGeneral_mex( ...
        uint32(d), dims, ranks, coreVecs{:}, ...
        uint32(SizeOmega), indices, p, PA);
    Px = mexOutputs{1};
    eucGrad.U = zeros(r, n, r);
    for k = 1:d
        eucGrad.U = eucGrad.U + permute(mexOutputs{k+1}, [1 3 2]);
    end
end

residual = Px - PA;
cost = 0.5*(residual'*residual)/p;
end
