function error=getFullError(X,A)
%GETFULLERROR Compute the squared full-tensor error for a fourth-order TR.
%
% Input:
%   X: fourth-order TR tensor (struct)
%   A: reference full tensor (numeric array)
%
% Output:
%   error: squared Frobenius error between X and A (scalar)
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
n=X.n;
error=0;
for i=1:n(1)
    for j=1:n(2)
        for k=1:n(3)
            for l=1:n(4)
                error=error+(getTRvalue(X,[i,j,k,l],d)-A(i,j,k,l))^2;
            end
        end
    end
end
