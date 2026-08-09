function X=TR_randn(dim,d,r,varargin)
%TR_RANDN Generate a random TR tensor with standard normal core entries.
%
% Input:
%   dim: tensor mode sizes (1-by-d vector)
%   d: tensor order (scalar)
%   r: TR ranks with d or d+1 entries (vector)
%   varargin: optionally {'full', true} to return the full tensor
%
% Output:
%   X: random TR tensor (struct), or its full numeric array when requested
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

r(d+1)=r(1);



X.r=r;
X.d=d;
X.n=dim;

for i=1:d
    X.core{i}=randn(X.r(i),X.r(i+1),dim(i)); 
end

if nargin>3 && strcmp(varargin{1},"full")==1 && varargin{2}==1
    H=ComputeUneqk(X,dim,prod(dim),r,d);
    U=reshape(X.core{1},[r(1)*r(1+1) dim(1)])';
    X=reshape(U*H{1}',dim);
end
