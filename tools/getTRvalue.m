function x=getTRvalue(X,i,d)
%GETTRVALUE Evaluate one entry of a TR tensor.
%
% Input:
%   X: TR tensor (struct)
%   i: tensor multi-index (1-by-d vector)
%   d: tensor order (scalar)
%
% Output:
%   x: tensor entry X(i(1),...,i(d)) (scalar)
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

x=X.core{1}(:,:,i(1));
for k=2:d-1
    x=x*X.core{k}(:,:,i(k));
end
y=X.core{d}(:,:,i(d))';
x=x(:)'*y(:);
