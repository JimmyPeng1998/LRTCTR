function [Xnew, duration, errorGamma, info] = ...
        Exp1_run_TR_E(method, X, PA, Omega, PAGamma, Gamma, opts)
%EXP1_RUN_TR_E Run a Euclidean-metric TR completion method.

% Replace the quotient geometry by the ambient Euclidean geometry while
% keeping the same cost function, initial point, and stopping criterion.
d = X.d; n = X.n; r = X.r; p = size(Omega,1)/prod(n);
M = TensorRingQuotient(n, r);
M.dim = @() sum(n.*r(1:d).*r(2:d+1));
M.proj = @(~, eta) eta;
M.tangent = M.proj;
M.transp = @(~, ~, eta) eta;
problem.M = M;
problem.costgrad = @costgrad;
options.maxiter = opts.maxiter;
options.maxtime = opts.maxTime;
options.tolgradnorm = opts.gradtol;
options.minstepsize = opts.minstepsize;
options.verbosity = opts.verbosity;
options.statsfun = @statsfun;
options.stopfun = @stopfun;
Xstart.core = X.core;

% Call the requested Manopt solver.
if strcmpi(method, 'RGD')
    [Xopt, ~, info] = steepestdescent(problem, Xstart, options);
else
    options.beta_type = 'H-S';
    [Xopt, ~, info] = conjugategradient(problem, Xstart, options);
end
Xnew = X; Xnew.core = Xopt.core;

% Extract the runtime and test-error histories from Manopt.
duration = reshape([info.time], [], 1);
errorGamma = reshape([info.errorGamma], [], 1);
    function [cost, grad] = costgrad(Y)
        Z = X; Z.core = Y.core;
        [cost, grad] = TR_costgrad(Z, Omega, PA);
    end
    function stats = statsfun(~, Y, stats)
        Z = X; Z.core = Y.core;
        stats.errorGamma = norm(TR_sample(Z, Gamma)-PAGamma)/norm(PAGamma);
    end
    function stop = stopfun(~, ~, records, last)
        errorOmega = sqrt(max(0,2*p*records(last).cost))/norm(PA);
        stop = opts.train_tol > 0 && errorOmega < opts.train_tol;
    end
end
