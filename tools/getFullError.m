function error=getFullError(X,A)
d=X.d;
n=X.n;
error=0;
for i=1:n(1)
    for j=1:n(2)
        for k=1:n(3)
            for l=1:n(4)
                error=error+(getTRvalue(X,[i,j,k,l],d)-A(i,j,k,l))^2;
            end
        end
    end
end
