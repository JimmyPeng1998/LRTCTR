function H=ComputeUneqk(X,n,prodn,r,d)
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