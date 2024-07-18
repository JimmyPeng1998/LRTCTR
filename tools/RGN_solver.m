function [Px,G]=RGN_solver(X,PA,SizeOmega,temp_Omega)
r=uint32(X.r);
n=uint32(X.n);
Px=zeros(SizeOmega,1);
G=cell(3,1);

% CoefMat=zeros(SizeOmega,X.elements);

r1r2=r(1)*r(2);
n1r1r2=n(1)*r1r2;
r2r3=r(2)*r(3);
n2r2r3=n(2)*r2r3;
r3r1=r(3)*r(1);
% tic
% for i=1:SizeOmega
%     inds=Omega(i,:);
%     
%     temp1=X.core{1}(:,:,inds(1));
%     temp2=X.core{2}(:,:,inds(2));
%     temp3=X.core{3}(:,:,inds(3));
%     
%     temp=(temp2*temp3)';
%     Px(i)=temp1(:)'*temp(:);
%     StartInd=(inds(1)-1)*r1r2+1;
%     CoefMat(i,StartInd:(StartInd+r1r2-1))=temp(:)';
%     
%     temp=(temp3*temp1)';
%     StartInd=n1r1r2+(inds(2)-1)*r2r3+1;
%     CoefMat(i,StartInd:(StartInd+r2r3-1))=temp(:)';
%     
%     
%     temp=(temp1*temp2)';
%     StartInd=n1r1r2+n2r2r3+(inds(3)-1)*r3r1+1;
%     CoefMat(i,StartInd:(StartInd+r3r1-1))=temp(:)';
%     
% end
% 
% toc
% tic
% [Px,Coefmat_temp]=RGN_matrix_mex(...
%     3,uint32(n),uint32(r),...
%     X.core{1}(:),X.core{2}(:),X.core{3}(:),...
%     uint32(SizeOmega),uint32(temp_Omega(:)));
[Px,CoefMat]=RGN_matrix_mex(...
    3,n,r,...
    X.core{1}(:),X.core{2}(:),X.core{3}(:),...
    uint32(SizeOmega),temp_Omega(:));
% CoefMat=reshape(CoefMat,[SizeOmega,X.elements]);
% toc
b=-(Px-PA);

% CoefMat=sparse(CoefMat);
% Coefmat_temp=sparse(Coefmat_temp);
% G1G2G3=CoefMat\b;
G1G2G3=(CoefMat*CoefMat'+1e-10*speye(X.elements))\(CoefMat*b);
% G1G2G3=lsqr(CoefMat',b,1e-6,r1r2);
% G1G2G3_temp=(Coefmat_temp'*Coefmat_temp+1e-10*speye(X.elements))\(Coefmat_temp'*b);
G{1}=reshape(G1G2G3(1:n1r1r2),[r1r2,n(1)])';
G{2}=reshape(G1G2G3(n1r1r2+1:n1r1r2+n2r2r3),[r2r3,n(2)])';
G{3}=reshape(G1G2G3(n1r1r2+n2r2r3+1:end),[r3r1,n(3)])';
