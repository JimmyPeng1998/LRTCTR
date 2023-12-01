function grad=getTRgrad(X,k,d,i)
index=mod(k,d)+1;
grad=X.core{index}(:,:,i(index));
for l=k+1:k+d-2
    grad=grad*X.core{mod(l,d)+1}(:,:,i(mod(l,d)+1));
end