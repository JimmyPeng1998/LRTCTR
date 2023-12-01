function ind=mysub2ind(sub,dims)
% Only deal with 1 Sample!
c=cumprod(dims(1:end-1));
ind=(sub-1)*[1 c]'+1;