function values = TR_sample(X, Omega)
%TR_SAMPLE Evaluate a TR tensor only at the indices in Omega.
%
% Input:
%   X: TR tensor with core{k} of size r(k)-by-r(k+1)-by-n(k) (struct)
%   Omega: queried multi-indices (m-by-d matrix)
%
% Output:
%   values: values of X at Omega (m-by-1 vector)
%
% Omega is an m-by-d matrix whose rows are tensor indices. This function
% does not construct the full tensor.
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


[d, n, r] = TR_validate_tensor(X, false);
TR_validate_observations(Omega, [], n, true, 'Sampling');

SizeOmega = size(Omega, 1);
indices = reshape(uint32(Omega'), [], 1);
switch d
    case 3
        values = ComputePx_mex(3, uint32(n), uint32(r), ...
            X.core{1}(:), X.core{2}(:), X.core{3}(:), ...
            uint32(SizeOmega), indices);
    case 4
        values = ComputePx_mex(4, uint32(n), uint32(r), ...
            X.core{1}(:), X.core{2}(:), X.core{3}(:), X.core{4}(:), ...
            uint32(SizeOmega), indices);
    otherwise
        if exist('ComputePxGeneral_mex', 'file') == 3
            coreVecs = cell(d, 1);
            for k = 1:d
                coreVecs{k} = X.core{k}(:);
            end
            values = ComputePxGeneral_mex(uint32(d), uint32(n), uint32(r), ...
                coreVecs{:}, uint32(SizeOmega), indices);
        else
            values = zeros(SizeOmega, 1);
            for ind = 1:SizeOmega
                product = eye(r(1));
                for k = 1:d
                    product = product*X.core{k}(:, :, Omega(ind, k));
                end
                values(ind) = trace(product);
            end
        end
end
end
