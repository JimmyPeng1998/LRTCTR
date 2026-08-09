%GENERATINGTR Generate a random TR tensor and its full representation.
%
% Input:
%   None. Parameters are configured in this script.
%
% Output:
%   X: randomly generated TR tensor (struct)
%   fullX: full tensor represented by X (numeric array)
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

clear
clc




% Default Settings
n=100;
d=3;
dim=ones(1,d)*n;
r=6*ones(1,d);
r(d+1)=r(1);



X.r=r;
X.d=d;
X.n=dim;

for i=1:d
    X.core{i}=rand(X.r(i),X.r(i+1),dim(i)); 
end

H=ComputeUneqk(X,dim,prod(dim),r,d);
U=reshape(X.core{1},[r(1)*r(1+1) n(1)])';
fullX=reshape(U*H{1}',dim);
% ComputeTuckerRanks(fullX,2,"Painting",1);
% save("Exp1_Synthetic_3_"+num2str(n)+".mat",'fullX')
% save("Exp1_Synthetic_3_"+num2str(n)+"_normal.mat",'fullX')
