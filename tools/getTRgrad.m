function grad=getTRgrad(X,k,d,i)
%GETTRGRAD Contract all sampled TR core slices except the k-th core.
%
% Input:
%   X: TR tensor (struct)
%   k: differentiated core index (scalar)
%   d: tensor order (scalar)
%   i: tensor multi-index (1-by-d vector)
%
% Output:
%   grad: contracted product of all core slices except core k (matrix)
%
% Reference: Riemannian preconditioned algorithms for tensor completion via
%    tensor ring decomposition,
%    Bin Gao, Renfeng Peng, and Ya-xiang Yuan,
%    Computational Optimization and Applications, 88(2):443--468, 2024.
%    https://doi.org/10.1007/s10589-024-00559-7
%
% Original author: Renfeng Peng, Dec. 01, 2023.
% Last modified: Renfeng Peng, Aug. 05, 2026.
%

index=mod(k,d)+1;
grad=X.core{index}(:,:,i(index));
for l=k+1:k+d-2
    grad=grad*X.core{mod(l,d)+1}(:,:,i(mod(l,d)+1));
end
