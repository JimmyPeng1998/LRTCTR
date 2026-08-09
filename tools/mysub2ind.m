function ind=mysub2ind(sub,dims)
%MYSUB2IND Convert a multi-index to a linear index.
%
% Input:
%   sub: one-based multi-index (row vector)
%   dims: array dimensions (vector)
%
% Output:
%   ind: corresponding one-based linear index (scalar)
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

c=cumprod(dims(1:end-1));
ind=(sub-1)*[1 c]'+1;
