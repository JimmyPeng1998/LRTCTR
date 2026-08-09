function value = TR_norm(X)
%TR_NORM Frobenius norm of a TR tensor without forming the full tensor.
%
% Input:
%   X: TR tensor with core{k} of size r(k)-by-r(k+1)-by-n(k) (struct)
%
% Output:
%   value: Frobenius norm of the represented full tensor (scalar)
%
% Reference: Riemannian preconditioned algorithms for tensor completion via
%    tensor ring decomposition,
%    Bin Gao, Renfeng Peng, and Ya-xiang Yuan,
%    Computational Optimization and Applications, 88(2):443--468, 2024.
%    https://doi.org/10.1007/s10589-024-00559-7
%
% Original author: Renfeng Peng, Aug. 05, 2026.
%


[d, n, r] = TR_validate_tensor(X, false);
transferProduct = eye(r(1)^2);

for k = 1:d
    transfer = zeros(r(k)^2, r(k+1)^2);
    for index = 1:n(k)
        coreSlice = X.core{k}(:, :, index);
        transfer = transfer + kron(coreSlice, coreSlice);
    end
    transferProduct = transferProduct*transfer;
end

normSquared = trace(transferProduct);
roundoffTol = 100*eps(max(1, norm(transferProduct, 'fro')));
if normSquared < -roundoffTol
    error('TR_norm:NegativeSquaredNorm', ...
        'The contracted squared norm is unexpectedly negative.');
end
value = sqrt(max(0, normSquared));
end
