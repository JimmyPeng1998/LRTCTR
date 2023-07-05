function X=TR_rand(r,d,dim,Omega,Gamma,SizeOmega,SizeGamma,lambda,PA,PAGamma)
X.r=r;
X.d=d;
X.n=dim;
X.Px=zeros(SizeOmega,1);
X.PGamma=zeros(SizeGamma,1);
X.normofCores=0;
temp_Omega=uint32(Omega');
temp_Gamma=uint32(Gamma');


for i=1:d
    X.core{i}=rand(X.r(i),X.r(i+1),dim(i)); 
    vec=reshape(X.core{i},[r(i)*r(i+1)*dim(i),1]);
    X.normofCores=X.normofCores+lambda*(vec'*vec)/2;
end

if nargin==10
    if d==3
        X.Px=ComputePx_mex(3,uint32(dim),uint32(r),X.core{1}(:),X.core{2}(:),X.core{3}(:),uint32(SizeOmega),uint32(temp_Omega(:)));
    elseif d==4
        X.Px=ComputePx_mex(4,uint32(dim),uint32(r),X.core{1}(:),X.core{2}(:),X.core{3}(:),X.core{4}(:),uint32(SizeOmega),uint32(temp_Omega(:)));
    end
    X.error=(X.Px-PA)'*(X.Px-PA);


    if d==3
        X.PGamma=ComputePx_mex(3,uint32(dim),uint32(r),X.core{1}(:),X.core{2}(:),X.core{3}(:),uint32(SizeGamma),uint32(temp_Gamma(:)));
    elseif d==4
        X.PGamma=ComputePx_mex(4,uint32(dim),uint32(r),X.core{1}(:),X.core{2}(:),X.core{3}(:),X.core{4}(:),uint32(SizeGamma),uint32(temp_Gamma(:)));
    end
    X.errorGamma=(X.PGamma-PAGamma)'*(X.PGamma-PAGamma);
end