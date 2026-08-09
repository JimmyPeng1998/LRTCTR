function values = uTR_sample(X, Omega)
%UTR_SAMPLE Evaluate a uniform TR tensor only at the indices in Omega.
%
% Input:
%   X: order-d uniform TR tensor with shared core U of size r-by-n-by-r
%      (struct)
%   Omega: queried multi-indices (m-by-d matrix)
%
% Output:
%   values: values of X at Omega (m-by-1 vector)
%
% X.U is the shared r-by-n-by-r core. Omega is an m-by-d index matrix. The
% same core is contracted at all d positions without forming the full tensor.
%
% Reference: Quotient geometry of tensor ring decomposition,
%    Bin Gao, Renfeng Peng, and Ya-xiang Yuan,
%    arXiv preprint arXiv:2601.21874, 2026.
%    https://arxiv.org/abs/2601.21874
%
% Original author: Renfeng Peng, Aug. 05, 2026.
%


requiredFields = {'d', 'n', 'r', 'U'};
if ~isstruct(X) || ~isscalar(X) || ~all(isfield(X, requiredFields))
    error('uTR_sample:TensorFormat', ...
        'X must contain the fields d, n, r and U.');
end

d = double(X.d);
n = double(X.n);
r = double(X.r);
if ~isscalar(d) || ~isreal(d) || ~isfinite(d) || d < 3 || d ~= floor(d)
    error('uTR_sample:Order', ...
        'X.d must be an integer greater than or equal to three.');
end
if ~isscalar(n) || ~isreal(n) || ~isfinite(n) || n < 1 || n ~= floor(n)
    error('uTR_sample:ModeSize', 'X.n must be a positive integer scalar.');
end
if ~isscalar(r) || ~isreal(r) || ~isfinite(r) || r < 1 || r ~= floor(r)
    error('uTR_sample:Rank', 'X.r must be a positive integer scalar.');
end
if ~isnumeric(Omega) || ~isreal(Omega) || ~ismatrix(Omega) || ...
        size(Omega, 2) ~= d
    error('uTR_sample:IndexSize', 'Omega must have d columns.');
end
if any(~isfinite(Omega(:))) || any(Omega(:) < 1) || ...
        any(Omega(:) ~= floor(Omega(:))) || ...
        any(Omega(:) > n)
    error('uTR_sample:Indices', ...
        'Omega must contain valid positive integer indices.');
end
if ~isa(X.U, 'double') || ~isreal(X.U) || ...
        size(X.U, 1) ~= r || size(X.U, 2) ~= n || ...
        size(X.U, 3) ~= r || numel(X.U) ~= r*n*r
    error('uTR_sample:CoreSize', 'X.U must have size r-by-n-by-r.');
end

core = reshape(permute(X.U, [2 1 3]), [n, r*r])';
SizeOmega = size(Omega, 1);
indices = reshape(uint32(Omega'), [], 1);
dims = uint32(n*ones(1, d));
ranks = uint32(r*ones(1, d+1));

if d == 3
    values = ComputePx_mex(3, dims, ranks, ...
        core(:), core(:), core(:), uint32(SizeOmega), indices);
else
    if exist('ComputePxGeneral_mex', 'file') ~= 3
        error('uTR_sample:GeneralMexNotFound', ...
            ['ComputePxGeneral_mex is required when d > 3. ', ...
             'Run install.m to compile the general-order MEX routine.']);
    end
    coreVecs = repmat({core(:)}, d, 1);
    values = ComputePxGeneral_mex(uint32(d), dims, ranks, ...
        coreVecs{:}, uint32(SizeOmega), indices);
end
end
