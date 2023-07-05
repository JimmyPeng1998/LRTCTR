clear
clc
clf

% For reproducible results
rng(16)

% Compared solvers
solvers={'TR-RGD (Armijo)',... % Armijo backtracking linesearch
    'TR-RGD (RBB)',... % RBB2
    'TR-RGD (exact)',... % Exact linesearch
    'TR-RCG (HS+)'};
selectedSolver=[1 2 3 4];

%% Default Settings
% Tensor size and true TR rank
n=100;
d=3;
dim=ones(1,d)*n;
rank=6;
r=rank*ones(1,d);
r(d+1)=r(1);

const=0.4; % Backtracking constant
lambda=1e-18; % Normalized term

% Training and test set
p=0.05;
SizeOmega=floor(n^d*p);
SizeGamma=100;

Omega = makeOmegaSet_mod( dim, SizeOmega);
Gamma = makeOmegaSet_mod( dim, SizeGamma);





% Generating the true tensor A
Atemp=TR_rand(r,d,dim,Omega,Gamma,SizeOmega,SizeGamma,lambda);
A1=getFullTR(Atemp,dim,r,d);
N=randn(dim);





% Noise level
epsilon=1e-6;


A=A1/norm(A1(:))+epsilon*N/norm(N(:));
if d==3
    PA=A(sub2ind(dim,Omega(:,1),Omega(:,2),Omega(:,3)));
else
    PA=A(sub2ind(dim,Omega(:,1),Omega(:,2),Omega(:,3),Omega(:,4)));
end
if d==3
    PAGamma=A(sub2ind(dim,Gamma(:,1),Gamma(:,2),Gamma(:,3)));
else
    PAGamma=A(sub2ind(dim,Gamma(:,1),Gamma(:,2),Gamma(:,3),Gamma(:,4)));
end


% Stats of results
comparedSolvers=4;
Xnew=cell(comparedSolvers,1);
duration=cell(comparedSolvers,1);
error=cell(comparedSolvers,1);
errorGamma=cell(comparedSolvers,1);

% Algorithm options
maxIter=1000;
maxTime=200;


% Initial guess
X=TR_rand(r,d,dim,Omega,Gamma,SizeOmega,SizeGamma,lambda,PA,PAGamma);
Xtemp=getFullTR(X,dim,r,d);
normX=norm(Xtemp(:));
for i=1:d
    X.core{i}=X.core{i}/nthroot(normX,3);
end


% Comparing selected solvers
for i=selectedSolver
    switch i
        case 1 % TR-RGD (Armijo)
            fprintf('Running TR-RGD (Armijo) ... \n');
            opts=struct('maxiter',maxIter,'maxTime',maxTime,...
                'err',1e-12,'tol',1e-8,'gradtol',1e-8,...
                'delta',1e-15,'lambda',lambda,'const',const);
            [Xnew{1},duration{1},error{1},errorGamma{1}]=TR_RGD_Armijo(X,PA,Omega,SizeOmega,PAGamma,Gamma,SizeGamma,p,opts);
            
        case 2 % TR-RGD (RBB2)
            fprintf('Running TR-RGD (RBB2) ... \n');
            opts=struct('maxiter',maxIter,'maxTime',maxTime,...
                'err',1e-12,'tol',1e-8,'gradtol',1e-8,...
                'delta',1e-15,'lambda',lambda);
            [Xnew{2},duration{2},error{2},errorGamma{2}]=TR_RGD_RBB2(X,PA,Omega,SizeOmega,PAGamma,Gamma,SizeGamma,p,opts);
            
        case 3 % TR-RGD (exact)
            fprintf('Running TR-RGD (exact) ... \n');
            opts=struct('maxiter',maxIter,'maxTime',maxTime,...
                'err',1e-12,'tol',1e-8,'gradtol',1e-8,...
                'delta',1e-15,'lambda',lambda);
            [Xnew{3},duration{3},error{3},errorGamma{3}]=TR_RGD_exact(X,PA,Omega,SizeOmega,PAGamma,Gamma,SizeGamma,p,opts);
            
        case 4 % TR-RCG (HS+)
            fprintf('Running TR-RCG (HS+) ... \n');
            opts=struct('maxiter',maxIter,'maxTime',maxTime,...
                'err',1e-12,'tol',1e-8,'gradtol',1e-8,...
                'delta',1e-15,'lambda',lambda,'const',const);
            [Xnew{4},duration{4},error{4},errorGamma{4}]=TR_RCG_HS(X,PA,Omega,SizeOmega,PAGamma,Gamma,SizeGamma,p,opts);
    end
    
end

% Plotting results
lwidth=4;
msize=8;

colors=get(gca,'ColorOrder');
for i=selectedSolver
    semilogy(duration{i},error{i},'LineWidth',lwidth,'Color',colors(i,:),'Marker','^','MarkerSize',msize)
    hold on
end

legend(solvers{selectedSolver})
xlabel('Time (s)')
ylabel('Training error')
set(gca,'FontSize',16)
set(gca,'YTick',10.^(log10(epsilon):1:0))




figure()
for i=selectedSolver
    semilogy(duration{i},errorGamma{i},'LineWidth',lwidth,'Color',colors(i,:),'Marker','^','MarkerSize',msize)
    hold on
end

legend(solvers{selectedSolver})
xlabel('Time (s)')
ylabel('Test error')
set(gca,'FontSize',16)
set(gca,'YTick',10.^(log10(epsilon):1:0))





















