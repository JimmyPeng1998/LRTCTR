function fullX=getFullTR(X,dim,r,d)
H=ComputeUneqk(X,dim,prod(dim),r,d);
U=reshape(X.core{1},[r(1)*r(1+1) dim(1)])';
fullX=reshape(U*H{1}',dim);
end