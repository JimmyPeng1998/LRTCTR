function fullX=getFullTR(X,dim,r,d)
%GETFULLTR Convert a TR representation to a full tensor.
%
% Input:
%   X: TR tensor (struct)
%   dim: tensor mode sizes (1-by-d vector)
%   r: cyclic TR ranks (1-by-(d+1) vector)
%   d: tensor order (scalar)
%
% Output:
%   fullX: full tensor represented by X (numeric array)
%
% Reference: Riemannian preconditioned algorithms for tensor completion via
%    tensor ring decomposition,
%    Bin Gao, Renfeng Peng, and Ya-xiang Yuan,
%    Computational Optimization and Applications, 88(2):443--468, 2024.
%    https://doi.org/10.1007/s10589-024-00559-7
%
% Original author: Renfeng Peng, Jul. 05, 2023.
% Last modified: Renfeng Peng, Aug. 05, 2026.
%

H=ComputeUneqk(X,dim,prod(dim),r,d);
U=reshape(X.core{1},[r(1)*r(1+1) dim(1)])';
fullX=reshape(U*H{1}',dim);
end
