function [Xnew,duration,error,errorGamma]=TR_RGD_exact(X,PA,Omega,SizeOmega,PAGamma,Gamma,SizeGamma,p,opts)
% function M = [Xnew,duration,error,errorGamma]=TR_RGD_Armijo(X,PA,Omega,SizeOmega,PAGamma,Gamma,SizeGamma,p,opts)
% 
% Riemannian gradient descent algorithm with exact linesearch for tensor
% ring completion. 
%
% Original author: Renfeng Peng, Jul. 02, 2023.

if ~isfield( opts, 'maxiter');  opts.maxiter = 100;     end
if ~isfield( opts, 'tol');      opts.tol = 1e-6;        end
if ~isfield( opts, 'gradtol');      opts.gradtol = 1e-8;        end
if ~isfield( opts, 'delta');  opts.delta = 1e-8;  end
if ~isfield( opts, 'lambda');  opts.lambda = 1e-10;  end
if ~isfield( opts, 'optimizer');  opts.optimizer = 'GD';  end
if ~isfield( opts, 'const');  opts.const = 0.005;  end
if ~isfield( opts, 'maxTime');  opts.maxTime = 1000;  end
temp_Gamma=Gamma';
temp_Omega=Omega';


%% Initialization
% X Iterator
% Grad Euclidean Gradient
% G Riemann Gradient
% D Search Direction
d=X.d;
n=X.n;
r=X.r;
r(d+1)=r(1);
lambda=opts.lambda;
delta=opts.delta;

Xformer=X;
Gformer=cell(d,1);

G=cell(d,1);
Grad=cell(d,1);
D=cell(d,1);
H=cell(d,1);
Hnew=cell(d,1);

Dnew=cell(d,1);
Gradnew=cell(d,1);
Gnew=cell(d,1);
Xnew=X;
duration=zeros(opts.maxiter+1,1);
error=zeros(opts.maxiter+1,1);
error(1)=X.error;
errorGamma=zeros(opts.maxiter+1,1);
errorGamma(1)=X.errorGamma;

for k=1:d
    D{k}=zeros(n(k),r(k)*r(k+1));
    Dnew{k}=zeros(n(k),r(k)*r(k+1));
    Grad{k}=zeros(r(k),r(k+1),n(k));
end

prodn=prod(n);

%% Iteration
for epoch=1:opts.maxiter
    
    %     for k=1:d
    %         %         H{k}=zeros(SizeOmega,r(k)*r(k+1));
    % %         D{k}=zeros(n(k),r(k)*r(k+1));
    % %         Hnew{k}=zeros(prodn/n(k),r(k)*r(k+1));
    %         Gnew{k}=zeros(n(k),r(k)*r(k+1));
    %         %         Grad{k}=zeros(r(k),r(k+1),n(k));
    %     end
    
    
    tic
    if epoch==1
        Xnew=X;
        %         [Xnew.Px,Gnew]=ComputeGradsAndPx(Xnew,Omega,PA,SizeOmega,p);
        if d==3
            Gnew=cell(3,1);
            [Xnew.Px,Gnew{1},Gnew{2},Gnew{3}]=ComputeGradsAndPx_mex(...
                3,uint32(n),uint32(r),...
                Xnew.core{1}(:),Xnew.core{2}(:),Xnew.core{3}(:),...
                uint32(SizeOmega),uint32(temp_Omega(:)),p,PA);
            Gnew{1}=reshape(Gnew{1},[n(1),r(1)*r(2)]);
            Gnew{2}=reshape(Gnew{2},[n(2),r(2)*r(3)]);
            Gnew{3}=reshape(Gnew{3},[n(3),r(3)*r(1)]);
        elseif d==4
            Gnew=cell(4,1);
            [Xnew.Px,Gnew{1},Gnew{2},Gnew{3},Gnew{4}]=ComputeGradsAndPx_mex(...
                4,uint32(n),uint32(r),...
                Xnew.core{1}(:),Xnew.core{2}(:),Xnew.core{3}(:),Xnew.core{4}(:),...
                uint32(SizeOmega),uint32(temp_Omega(:)),p,PA);
            Gnew{1}=reshape(Gnew{1},[n(1),r(1)*r(2)]);
            Gnew{2}=reshape(Gnew{2},[n(2),r(2)*r(3)]);
            Gnew{3}=reshape(Gnew{3},[n(3),r(3)*r(4)]);
            Gnew{4}=reshape(Gnew{4},[n(4),r(4)*r(1)]);
            %             temp=ComputeGradsAndPx_mex(4,uint32(n),uint32(r),Xnew.core{1}(:),Xnew.core{2}(:),Xnew.core{3}(:),Xnew.core{4}(:),uint32(SizeOmega),uint32(temp_Gamma(:)));
        end
        error_now=(PA-Xnew.Px)'*(PA-Xnew.Px);
    else
        t=1;
        %         while t>1e-10
        %             Xnew.normofCores=0;
        for k=1:d
            Xnew.core{k}=X.core{k}+t*reshape(D{k}',[r(k) r(k+1) n(k)]);
        end
        
        %         [Xnew.Px,Gnew]=ComputeGradsAndPx(Xnew,Omega,PA,SizeOmega,p);
        if d==3
            Gnew=cell(3,1);
            [Xnew.Px,Gnew{1},Gnew{2},Gnew{3}]=ComputeGradsAndPx_mex(...
                3,uint32(n),uint32(r),...
                Xnew.core{1}(:),Xnew.core{2}(:),Xnew.core{3}(:),...
                uint32(SizeOmega),uint32(temp_Omega(:)),p,PA);
            Gnew{1}=reshape(Gnew{1},[n(1),r(1)*r(2)]);
            Gnew{2}=reshape(Gnew{2},[n(2),r(2)*r(3)]);
            Gnew{3}=reshape(Gnew{3},[n(3),r(3)*r(1)]);
        elseif d==4
            Gnew=cell(4,1);
            [Xnew.Px,Gnew{1},Gnew{2},Gnew{3},Gnew{4}]=ComputeGradsAndPx_mex(...
                4,uint32(n),uint32(r),...
                Xnew.core{1}(:),Xnew.core{2}(:),Xnew.core{3}(:),Xnew.core{4}(:),...
                uint32(SizeOmega),uint32(temp_Omega(:)),p,PA);
            Gnew{1}=reshape(Gnew{1},[n(1),r(1)*r(2)]);
            Gnew{2}=reshape(Gnew{2},[n(2),r(2)*r(3)]);
            Gnew{3}=reshape(Gnew{3},[n(3),r(3)*r(4)]);
            Gnew{4}=reshape(Gnew{4},[n(4),r(4)*r(1)]);
            %             temp=ComputeGradsAndPx_mex(4,uint32(n),uint32(r),Xnew.core{1}(:),Xnew.core{2}(:),Xnew.core{3}(:),Xnew.core{4}(:),uint32(SizeOmega),uint32(temp_Gamma(:)));
        end
        error_now=(PA-Xnew.Px)'*(PA-Xnew.Px);
        
        %             if 0.5*error(epoch-1)/p-0.5*error_now/p>0.00001*t*GradAndDirec
        %                 t
        %                 break
        %             end
        %
        %             t=t*0.5;
        
        %         end
        %         Uneqk_new=ComputeUneqk(Xnew,n,prodn,r,d);
        
        
        
        if abs(error_now-error(epoch-1))/error(epoch-1)<opts.tol
            duration(epoch)=toc;
            error(epoch)=error_now;
            
            %             Xnew.PGamma=ComputePx(Xnew,SizeGamma,Gamma);
            if d==3
                Xnew.PGamma=ComputePx_mex(3,uint32(n),uint32(r),Xnew.core{1}(:),Xnew.core{2}(:),Xnew.core{3}(:),uint32(SizeGamma),uint32(temp_Gamma(:)));
            elseif d==4
                Xnew.PGamma=ComputePx_mex(4,uint32(n),uint32(r),Xnew.core{1}(:),Xnew.core{2}(:),Xnew.core{3}(:),Xnew.core{4}(:),uint32(SizeGamma),uint32(temp_Gamma(:)));
            end
            errorGamma(epoch)=(Xnew.PGamma-PAGamma)'*(Xnew.PGamma-PAGamma);
            disp("The algorithm stagnates after "+num2str(epoch)+" iterations.")
            break
        end
        
        if epoch==opts.maxiter
            
            duration(epoch)=toc;
            error(epoch)=error_now;
            
            %             Xnew.PGamma=ComputePx(Xnew,SizeGamma,Gamma);
            if d==3
                Xnew.PGamma=ComputePx_mex(3,uint32(n),uint32(r),Xnew.core{1}(:),Xnew.core{2}(:),Xnew.core{3}(:),uint32(SizeGamma),uint32(temp_Gamma(:)));
            elseif d==4
                Xnew.PGamma=ComputePx_mex(4,uint32(n),uint32(r),Xnew.core{1}(:),Xnew.core{2}(:),Xnew.core{3}(:),Xnew.core{4}(:),uint32(SizeGamma),uint32(temp_Gamma(:)));
            end
            errorGamma(epoch)=(Xnew.PGamma-PAGamma)'*(Xnew.PGamma-PAGamma);
            
            break
        end
        
        %         Gnew=ComputeEuclidGrad(Xnew,n,r,d,Omega,SizeOmega,PA,p,Uneqk_new);
    end
    
    %     normofGrad=0;
    % Compute Riemann Gradient of (U(k))_(2)
    for k=1:d
        
        Gnew{k}=Gnew{k}+lambda*reshape(Xnew.core{k},[r(k)*r(k+1) n(k)])';
        
        
        %         normofGrad=normofGrad+norm(Gnew{k},'fro')^2;
        
        %         Hnew{k}=Uneqk_new{k}'*Uneqk_new{k}+delta*eye(r(k)*r(k+1)); % r(k)r(k+1) square matrix (prod(n)*/n(k))^2*
        
        Hnew2=ComputeUneqkTUneqk(Xnew,k);
        Hnew{k}=Hnew2+delta*eye(r(k)*r(k+1));
        Gnew{k}=Gnew{k}/Hnew{k};
    end
    
    %     sqrt(normofGrad)
    
    %     if sqrt(normofGrad)<opts.gradtol
    %         disp("The algorithm terminates after "+num2str(epoch)+" iterations, with KKT violation "+num2str(normofGrad))
    %         break
    %     end
    
    %% Deciding Step Size
    %     if epoch==1 % The first step must be the steepest descent step
    
    %         Gcores=Unfolding2Slices(Gnew,r,d,X.n,SizeOmega,Omega);
    %         alpha=max(nthroot(-(Gcores.Px'*(PA-X.Px))/(Gcores.Px'*Gcores.Px),d),const)
    for k=1:d
        Dnew{k}=-Gnew{k};
    end
    
    if d==3
        alpha=ComputeStepSize3(Xnew,Dnew,PA,r,d,n,SizeOmega,Omega,lambda);
    else
        alpha=ComputeStepSize4(Xnew,Dnew,PA,r,d,n,SizeOmega,Omega,lambda);
    end
    
    for k=1:d
        Dnew{k}=alpha*Dnew{k};
    end
    
    %         GradAndDirec=alpha*RiemannInner(Gnew,Hnew,Gnew,d);
    %     end
    
    
    duration(epoch)=toc;
    error(epoch)=error_now;
    
    
    
    %     Xnew.PGamma=ComputePx(Xnew,SizeGamma,Gamma);
    if d==3
        Xnew.PGamma=ComputePx_mex(3,uint32(n),uint32(r),Xnew.core{1}(:),Xnew.core{2}(:),Xnew.core{3}(:),uint32(SizeGamma),uint32(temp_Gamma(:)));
    elseif d==4
        Xnew.PGamma=ComputePx_mex(4,uint32(n),uint32(r),Xnew.core{1}(:),Xnew.core{2}(:),Xnew.core{3}(:),Xnew.core{4}(:),uint32(SizeGamma),uint32(temp_Gamma(:)));
    end
    errorGamma(epoch)=(Xnew.PGamma-PAGamma)'*(Xnew.PGamma-PAGamma);
    
    
    
    
    %     Xformer=X;
    %     Hformer=H;
    %     Gformer=G;
    %     Uneqkformer=Uneqk;
    %     Dformer=D;
    
    X=Xnew;
    %     H=Hnew;
    G=Gnew;
    %     Uneqk=Uneqk_new;
    D=Dnew;
    
    if sqrt(errorGamma(epoch)/(PAGamma'*PAGamma))<opts.err
        disp("The algorithm converges after "+num2str(epoch)+" iterations.")
        
        break
    end
    
    if sum(duration(1:(epoch)))>opts.maxTime
        disp("The algorithm exceeds the time budget.")
        break
    end
    
    
end


if epoch==opts.maxiter
    disp("The algorithm reaches the maximum iteration.")
end


duration=cumsum(duration(1:(epoch)));
error=sqrt(error(1:(epoch))/(PA'*PA));
errorGamma=sqrt(errorGamma(1:(epoch))/(PAGamma'*PAGamma));
end


function result=RiemannInner(xi,H,eta,d)
result=0;
for k=1:d
    x=xi{k}*H{k};
    y=eta{k};
    result=result+x(:)'*y(:);
end
end
