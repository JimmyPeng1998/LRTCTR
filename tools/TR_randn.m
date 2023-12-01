function X=TR_randn(dim,d,r,varargin)
% Generating a TR tensor, with N(0,1) entries in each core. 
% X=TR_randn(dim,d,r,varargin)
% 'full': return a full tensor, otherwise a TR tensor.
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