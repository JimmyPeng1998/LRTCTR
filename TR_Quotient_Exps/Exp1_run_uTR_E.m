function [Xnew, duration, errorGamma, info] = ...
        Exp1_run_uTR_E(method, X, PA, Omega, PAGamma, Gamma, opts)
%EXP1_RUN_UTR_E Run a Euclidean-metric uniform TR completion method.

% Use the ambient Euclidean geometry for the shared core and retain the
% same sampled objective and stopping criterion as the quotient method.
d = X.d; n = X.n; r = X.r; p = size(Omega,1)/(n^d);
M = productmanifold(struct('U', euclideanfactory([r n r])));
problem.M = M;
problem.cost = @cost;
problem.egrad = @gradient;
options.maxiter = opts.maxiter;
options.maxtime = opts.maxTime;
options.tolgradnorm = opts.gradtol;
options.minstepsize = opts.minstepsize;
options.verbosity = opts.verbosity;
options.statsfun = @statsfun;
options.stopfun = @stopfun;
Xstart.U = X.U;

% Call the requested Manopt solver.
if strcmpi(method, 'RGD')
    [Xopt, ~, info] = steepestdescent(problem, Xstart, options);
else
    options.beta_type = 'H-S';
    [Xopt, ~, info] = conjugategradient(problem, Xstart, options);
end
Xnew = X; Xnew.U = Xopt.U;

% Extract the runtime and test-error histories from Manopt.
duration = reshape([info.time], [], 1);
errorGamma = reshape([info.errorGamma], [], 1);
    function value = cost(Y)
        residual = sampled(Y, Omega)-PA;
        value = 0.5*(residual'*residual)/p;
    end
    function grad = gradient(Y)
        [~, grad] = TR_uniform_costgrad(Y, Omega, PA, p, d, n, r);
    end
    function stats = statsfun(~, Y, stats)
        stats.errorGamma = norm(sampled(Y,Gamma)-PAGamma)/norm(PAGamma);
    end
    function stop = stopfun(~, ~, records, last)
        errorOmega = sqrt(max(0,2*p*records(last).cost))/norm(PA);
        stop = opts.train_tol > 0 && errorOmega < opts.train_tol;
    end
    function values = sampled(Y, indices)
        Z = X; Z.U = Y.U; values = uTR_sample(Z, indices);
    end
end
