function sub=myind2sub(ind,dims)
%MYIND2SUB Convert a linear index to a multi-index.
%
% Input:
%   ind: one-based linear index (scalar)
%   dims: array dimensions (vector)
%
% Output:
%   sub: corresponding one-based multi-index (row vector)
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

d=max(size(dims));
c=cumprod(dims(1:end-1));
sub=zeros(1,d);

for i=d:-1:1
    if i==1
        sub(1)=ind;
    else
        sub(i)=floor((ind-1)/c(i-1));
        ind=ind-sub(i)*c(i-1);
        sub(i)=sub(i)+1;
    end
end
