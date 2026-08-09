function X=Unfolding2Slices(x,r,d,dim,SizeOmega,Omega)
%UNFOLDING2SLICES Convert unfolded cores to TR slices and evaluate samples.
%
% Input:
%   x: unfolded TR cores (d-by-1 cell array)
%   r: cyclic TR ranks (1-by-(d+1) vector)
%   d: tensor order (scalar)
%   dim: tensor mode sizes (1-by-d vector)
%   SizeOmega: number of sampled entries (scalar)
%   Omega: sampled multi-indices (SizeOmega-by-d matrix)
%
% Output:
%   X: TR tensor with reshaped cores and sampled values (struct)
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

X.r=r;
X.d=d;
X.n=dim;
X.Px=zeros(SizeOmega,1);
temp_Omega=Omega';
for k=1:d
    X.core{k}=reshape(x{k}',[r(k),r(k+1),dim(k)]);
end
% for ind=1:SizeOmega
%     i=Omega(ind,:);
%     X.Px(ind)=getTRvalue(X,i,d);
% end
if d==3
    X.Px=ComputePx_mex(3,uint32(dim),uint32(r),X.core{1}(:),X.core{2}(:),X.core{3}(:),uint32(SizeOmega),uint32(temp_Omega(:)));
elseif d==4
    X.Px=ComputePx_mex(4,uint32(dim),uint32(r),X.core{1}(:),X.core{2}(:),X.core{3}(:),X.core{4}(:),uint32(SizeOmega),uint32(temp_Omega(:)));
end
end
