clear
clc
clf

% For reproducible results
rng(16)

% References:
%   [1] Riemannian preconditioned algorithms for tensor completion via
%       tensor ring decomposition,
%       Bin Gao, Renfeng Peng, and Ya-xiang Yuan,
%       Computational Optimization and Applications, 88(2):443--468, 2024.
%       https://doi.org/10.1007/s10589-024-00559-7
%   [2] Optimization on Product Manifolds under a Preconditioned Metric,
%       Bin Gao, Renfeng Peng, and Ya-xiang Yuan,
%       SIAM Journal on Matrix Analysis and Applications,
%       46(3):1816--1845, 2025.
%       https://doi.org/10.1137/24M1643773
%
% Original author: Renfeng Peng, Jul. 05, 2023.
% Last modified: Renfeng Peng, Aug. 05, 2026.

% Compared solvers
solvers={'TR-RGD (Armijo)',... % Armijo backtracking linesearch
    'TR-RGD (RBB)',... % RBB2
    'TR-RGD (exact)',... % Exact linesearch
    'TR-RCG (HS+)',... % RCG with HS+ stepsize
    'TR-(R)GN'}; % Gauss--Newton method
selectedSolver=[1 2 3 4 5];

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
normA=TR_norm(Atemp);
for i=1:d
    Atemp.core{i}=Atemp.core{i}/nthroot(normA,d);
end





% Noise level
epsilon=1e-6;


PA=TR_sample(Atemp,Omega)+epsilon*randn(SizeOmega,1)/sqrt(prod(dim));
PAGamma=TR_sample(Atemp,Gamma)+epsilon*randn(SizeGamma,1)/sqrt(prod(dim));


% Stats of results
comparedSolvers=5;
Xnew=cell(comparedSolvers,1);
duration=cell(comparedSolvers,1);
error=cell(comparedSolvers,1);
errorGamma=cell(comparedSolvers,1);

% Algorithm options
maxIter=1000;
maxTime=200;


% Initial guess
X=TR_rand(r,d,dim,Omega,Gamma,SizeOmega,SizeGamma,lambda,PA,PAGamma);
normX=TR_norm(X);
for i=1:d
    X.core{i}=X.core{i}/nthroot(normX,d);
end


% Comparing selected solvers
for i=selectedSolver
    switch i
        case 1 % TR-RGD (Armijo)
            fprintf('Running TR-RGD (Armijo) ... \n');
            opts=struct('maxiter',maxIter,'maxTime',maxTime,...
                'train_tol',1e-12,'tol',1e-8,'gradtol',1e-8,...
                'delta',1e-15,'lambda',lambda,'const',const);
            [Xnew{1},duration{1},error{1},errorGamma{1}]=TR_RGD_Armijo(X,PA,Omega,PAGamma,Gamma,opts);
            
        case 2 % TR-RGD (RBB2)
            fprintf('Running TR-RGD (RBB2) ... \n');
            opts=struct('maxiter',maxIter,'maxTime',maxTime,...
                'train_tol',1e-12,'tol',1e-8,'gradtol',1e-8,...
                'delta',1e-15,'lambda',lambda);
            [Xnew{2},duration{2},error{2},errorGamma{2}]=TR_RGD_RBB2(X,PA,Omega,PAGamma,Gamma,opts);
            
        case 3 % TR-RGD (exact)
            fprintf('Running TR-RGD (exact) ... \n');
            opts=struct('maxiter',maxIter,'maxTime',maxTime,...
                'train_tol',1e-12,'tol',1e-8,'gradtol',1e-8,...
                'delta',1e-15,'lambda',lambda);
            [Xnew{3},duration{3},error{3},errorGamma{3}]=TR_RGD_exact(X,PA,Omega,PAGamma,Gamma,opts);
            
        case 4 % TR-RCG (HS+)
            fprintf('Running TR-RCG (HS+) ... \n');
            opts=struct('maxiter',maxIter,'maxTime',maxTime,...
                'train_tol',1e-12,'tol',1e-8,'gradtol',1e-8,...
                'delta',1e-15,'lambda',lambda,'const',const);
            [Xnew{4},duration{4},error{4},errorGamma{4}]=TR_RCG_HS(X,PA,Omega,PAGamma,Gamma,opts);
            
        case 5
            fprintf('Running TR-(R)GN ... \n');
            opts=struct('maxiter',maxIter,'maxTime',maxTime,...
                'train_tol',1e-12,'tol',1e-8,'gradtol',1e-8,...
                'delta',1e-15,'lambda',lambda,'const',const);
            [Xnew{5},duration{5},error{5},errorGamma{5}]=TR_RGN(X,PA,Omega,PAGamma,Gamma,opts);
            
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


















