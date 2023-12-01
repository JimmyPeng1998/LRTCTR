function x=getTRvalue(X,i,d)
x=X.core{1}(:,:,i(1));
for k=2:d-1
    x=x*X.core{k}(:,:,i(k));
end
y=X.core{d}(:,:,i(d))';
x=x(:)'*y(:);