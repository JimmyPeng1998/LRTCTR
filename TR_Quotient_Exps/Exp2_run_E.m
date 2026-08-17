
function [Xnew, duration, errorOmega, errorGamma, info] = ...
        Exp2_run_E( ...
        method, useQuotientGeometry, X, PA, Omega, PAGamma, Gamma, opts)
%EXP2_RUN_E Run RGD or RCG under the Euclidean geometry.

% The objective is gauge invariant, so its Euclidean gradient is already
% horizontal. This reproduces the RGD(Q) implementation used in the
% original experiments without explicitly projecting that gradient.
d = X.d;
n = X.n;
r = X.r;
p = size(Omega, 1)/prod(n);

% Use the TR factory only as a container for the required Manopt operations.
% Replace its geometric operations by their Euclidean counterparts.
M = TensorRingQuotient(n, r);
if ~useQuotientGeometry
    M.dim = @() sum(n.*r(1:d).*r(2:d+1));
    M.proj = @(~, eta) eta;
    M.tangent = M.proj;
    M.transp = @(~, ~, eta) eta;
end

problem.M = M;
problem.costgrad = @costgrad;

% Use the same options and stopping criterion as the quotient algorithms.
options.maxiter = opts.maxiter;
options.maxtime = opts.maxTime;
options.tolgradnorm = opts.gradtol;
options.minstepsize = opts.minstepsize;
options.verbosity = opts.verbosity;
options.stopfun = @stopfun;
options.statsfun = @statsfun;

X0.core = X.core;

% Call the requested Manopt solver.
if strcmpi(method, 'RGD')
    [Xopt, ~, info] = steepestdescent(problem, X0, options);
else
    options.beta_type = 'H-S';
    [Xopt, ~, info] = conjugategradient(problem, X0, options);
end

% Collect the tensor and convergence histories in the package convention.
Xnew.d = d;
Xnew.n = n;
Xnew.r = r;
Xnew.core = Xopt.core;
duration = reshape([info.time], [], 1);
errorOmega = sqrt(max(0, 2*p*reshape([info.cost], [], 1)))/norm(PA);
errorGamma = reshape([info.errorGamma], [], 1);

    function [cost, grad] = costgrad(Xcurrent)
        [cost, grad] = TR_costgrad(makeTR(Xcurrent), Omega, PA);
    end

    function stats = statsfun(~, Xcurrent, stats)
        valuesGamma = TR_sample(makeTR(Xcurrent), Gamma);
        stats.errorGamma = norm(valuesGamma-PAGamma)/norm(PAGamma);
    end

    function stopnow = stopfun(~, ~, infoNow, last)
        errorNow = sqrt(max(0, 2*p*infoNow(last).cost))/norm(PA);
        stopnow = opts.train_tol > 0 && errorNow < opts.train_tol;
        if ~stopnow && opts.tol > 0 && last > 1
            costNow = infoNow(last).cost;
            costOld = infoNow(last-1).cost;
            relativeChange = abs(costNow-costOld)/max(abs(costNow), eps);
            stopnow = relativeChange < opts.tol;
        end
    end

    function tensor = makeTR(Xcurrent)
        tensor.d = d;
        tensor.n = n;
        tensor.r = r;
        tensor.core = Xcurrent.core;
    end
end
