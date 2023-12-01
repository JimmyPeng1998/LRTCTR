function H=ComputeUneqkTUneqk(X,k)
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