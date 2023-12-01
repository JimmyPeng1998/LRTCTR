function X=Unfolding2Slices(x,r,d,dim,SizeOmega,Omega)
X.r=r;
X.d=d;
X.n=dim;
X.Px=zeros(SizeOmega,1);
temp_Omega=Omega';
for k=1:d
    X.core{k}=reshape(x{k}',[r(k),r(k+1),dim(k)]);
end
% for ind=1:SizeOmega
%     i=Omega(ind,:);
%     X.Px(ind)=getTRvalue(X,i,d);
% end
if d==3
    X.Px=ComputePx_mex(3,uint32(dim),uint32(r),X.core{1}(:),X.core{2}(:),X.core{3}(:),uint32(SizeOmega),uint32(temp_Omega(:)));
elseif d==4
    X.Px=ComputePx_mex(4,uint32(dim),uint32(r),X.core{1}(:),X.core{2}(:),X.core{3}(:),X.core{4}(:),uint32(SizeOmega),uint32(temp_Omega(:)));
end
end
