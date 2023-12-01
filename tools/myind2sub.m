function sub=myind2sub(ind,dims)
% Only deal with 1 Sample!
d=max(size(dims));
c=cumprod(dims(1:end-1));
sub=zeros(1,d);

for i=d:-1:1
    if i==1
        sub(1)=ind;
    else
        sub(i)=floor((ind-1)/c(i-1));
        ind=ind-sub(i)*c(i-1);
        sub(i)=sub(i)+1;
    end
end

