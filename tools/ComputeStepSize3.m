function alpha=ComputeStepSize3(X,D,PA,r,d,dim,SizeOmega,Omega,lambda)
% only for d=3!!!!!!
% tic
temp_Omega=Omega';
temp=Unfolding2Slices(D,r,d,dim,SizeOmega,Omega);
vec3=temp.Px;

vec2=zeros(SizeOmega,1);
for k=1:3
    temp1=temp;
    temp1.core{k}=X.core{k};
%     for ind=1:SizeOmega
%         i=Omega(ind,:);
%         vec2(ind)=vec2(ind)+getTRvalue(temp1,i,d);
%     end
    temp1_Px=ComputePx_mex(3,uint32(dim),uint32(r),...
        temp1.core{1}(:),temp1.core{2}(:),temp1.core{3}(:),...
        uint32(SizeOmega),uint32(temp_Omega(:)));
    vec2=vec2+temp1_Px;
end

vec1=zeros(SizeOmega,1);
for k=1:3
    temp1=X;
    temp1.core{k}=temp.core{k};
%     for ind=1:SizeOmega
%         i=Omega(ind,:);
%         vec1(ind)=vec1(ind)+getTRvalue(temp1,i,d);
%     end
    temp1_Px=ComputePx_mex(3,uint32(dim),uint32(r),...
        temp1.core{1}(:),temp1.core{2}(:),temp1.core{3}(:),...
        uint32(SizeOmega),uint32(temp_Omega(:)));
    vec1=vec1+temp1_Px;
end

ans1=0;
ans2=0;
for k=1:3
    for i=1:dim(k)
        temp3=X.core{k}(:,:,i);
        temp4=temp.core{k}(:,:,i);
        ans2=ans2+0.5*lambda*temp4(:)'*temp4(:);
        ans1=ans1+lambda*temp3(:)'*temp4(:);
    end
end



vec0=X.Px-PA;


coefs=zeros(1,7);

% From x^6 to constant
coefs(1)=vec3'*vec3;
coefs(2)=2*vec3'*vec2;
coefs(3)=2*vec3'*vec1+vec2'*vec2;
coefs(4)=2*vec3'*vec0+2*vec2'*vec1;
coefs(5)=2*vec2'*vec0+vec1'*vec1+ans2;
coefs(6)=2*vec1'*vec0+ans1;
% coefs(7)=vec0'*vec0;
p=coefs(1:6).*[6 5 4 3 2 1];

ts = roots(p);
ts = ts(imag(ts)==0);
ts = ts(ts>0);
if isempty(ts)
    ts = 1;
end


dfts = polyval([coefs(1:6) 0], ts);
[~, iarg] = min(dfts);
alpha = ts(iarg);
% toc

% function x=getTRvalue(X,i,d)
% x=X.core{1}(:,:,i(1));
% for k=2:d-1
%     x=x*X.core{k}(:,:,i(k));
% end
% y=X.core{d}(:,:,i(d))';
% x=x(:)'*y(:);




