function [cost, eucGrad, values] = TR_costgrad(X, Omega, PA)
%TR_COSTGRAD Sampled least-squares cost and gradient for a general TR.
%
% Input:
%   X: TR tensor with core{k} of size r(k)-by-r(k+1)-by-n(k) (struct)
%   Omega: observed multi-indices (m-by-d matrix)
%   PA: observed tensor entries on Omega (m-by-1 vector)
%
% Output:
%   cost: sampled least-squares objective value (scalar)
%   eucGrad: Euclidean gradient in the same TR representation as X (struct)
%   values: values of X at Omega (m-by-1 vector)
%
% The objective is
%   0.5/p * norm(TR_sample(X, Omega) - PA)^2,
% where p = size(Omega, 1)/prod(X.n). Both the cost and the gradients with
% respect to all TR cores are computed without forming the full tensor.
%
% Reference: Quotient geometry of tensor ring decomposition,
%    Bin Gao, Renfeng Peng, and Ya-xiang Yuan,
%    arXiv preprint arXiv:2601.21874, 2026.
%    https://arxiv.org/abs/2601.21874
%
% Original author: Renfeng Peng, Aug. 05, 2026.
%


[d, n, r] = TR_validate_tensor(X, false);
SizeOmega = size(Omega, 1);
PA = TR_validate_observations(Omega, PA, n, false, 'Training');
eucGrad.d = d;
eucGrad.n = n;
eucGrad.r = r;
eucGrad.core = cell(d, 1);
for k = 1:d
    eucGrad.core{k} = zeros(size(X.core{k}));
end

p = SizeOmega/prod(n);

% Use the general-order MEX backend when it is available. Its gradient
% outputs use exactly the same core layout as X, so no order-dependent
% permutations are needed.
if exist('ComputeGradsAndPxGeneral_mex', 'file') == 3
    coreVecs = cell(d, 1);
    mexOutputs = cell(d+1, 1);
    for k = 1:d
        coreVecs{k} = X.core{k}(:);
    end
    indices = reshape(uint32(Omega'), [], 1);
    [mexOutputs{:}] = ComputeGradsAndPxGeneral_mex( ...
        uint32(d), uint32(n), uint32(r), coreVecs{:}, ...
        uint32(SizeOmega), indices, p, PA);
    values = mexOutputs{1};
    for k = 1:d
        eucGrad.core{k} = mexOutputs{k+1};
    end
    residual = values-PA;
    cost = 0.5*(residual'*residual)/p;
    return
end

values = zeros(SizeOmega, 1);
prefix = cell(d+1, 1);
suffix = cell(d+1, 1);
slices = cell(d, 1);

for ind = 1:SizeOmega
    for k = 1:d
        slices{k} = X.core{k}(:, :, Omega(ind, k));
    end

    prefix{1} = eye(r(1));
    for k = 1:d
        prefix{k+1} = prefix{k}*slices{k};
    end
    values(ind) = trace(prefix{d+1});

    suffix{d+1} = eye(r(1));
    for k = d:-1:1
        suffix{k} = slices{k}*suffix{k+1};
    end

    residualScale = (values(ind)-PA(ind))/p;
    for k = 1:d
        environment = suffix{k+1}*prefix{k};
        index = Omega(ind, k);
        eucGrad.core{k}(:, :, index) = ...
            eucGrad.core{k}(:, :, index) + residualScale*environment';
    end
end

residual = values-PA;
cost = 0.5*(residual'*residual)/p;
end
