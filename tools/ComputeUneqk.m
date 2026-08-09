function H=ComputeUneqk(X,n,prodn,r,d)
%COMPUTEUNEQK Form the subchain unfolding associated with every TR core.
%
% Input:
%   X: TR tensor (struct)
%   n: tensor mode sizes (1-by-d vector)
%   prodn: product of all mode sizes (scalar)
%   r: cyclic TR ranks (1-by-(d+1) vector)
%   d: tensor order (scalar)
%
% Output:
%   H: mode-k subchain unfoldings (d-by-1 cell array)
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

H=cell(d,1);
for k=1:d
    H{k}=zeros(prodn/n(k),r(k)*r(k+1));
    
    
    dims=[n(1:k-1) n(k+1:d)];
    % Forming U^(\neq k)_(2) Currently seems too ineffective ...
    for ind=1:(prodn/n(k))
        subs=myind2sub(ind,dims);
        subs=[subs(1:k-1) 1 subs(k:d-1)];
        temp=getTRgrad(X,k,d,subs)';
        %         H{k}(ind,:)=reshape(temp',[1 r(k)*r(k+1)]);
        H{k}(ind,:)=temp(:)';
    end
end
end
