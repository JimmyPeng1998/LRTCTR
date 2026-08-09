function H=ComputeUneqkTUneqk(X,k)
%COMPUTEUNEQKTUNEQK Form the Gram matrix of a mode-k TR subchain unfolding.
%
% Input:
%   X: TR tensor (struct)
%   k: selected core index (scalar)
%
% Output:
%   H: Gram matrix of the mode-k subchain unfolding (matrix)
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

d=X.d;
r=X.r;
n=X.n;
map=[1:d 1:d];

ind=map(k+d-1);
W=reshape(permute(X.core{ind},[2 1 3]),[r(ind)*r(ind+1) n(ind)])';
H=W'*W;
I=speye(r(k));
for t=2:d-1
    ind=map(k+d-t);
%     temp=sparse(r(k)*r(ind),r(k)*r(ind));
    temp=zeros(r(k)*r(ind));
    for i=1:n(ind)
        temp=temp+kron(X.core{ind}(:,:,i),I)*H*kron(X.core{ind}(:,:,i)',I);
    end
%     H=full(temp);
    H=temp;
end
