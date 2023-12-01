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