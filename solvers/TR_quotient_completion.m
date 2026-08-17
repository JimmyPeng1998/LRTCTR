function [Xnew, duration, errorOmega, errorGamma, info] = ...
    TR_quotient_completion(method, X, PA, Omega, varargin)
%TR_QUOTIENT_COMPLETION Shared Manopt driver for TR_RGDQ and TR_RCGQ.
%
% Reference: Quotient geometry of tensor ring decomposition,
%    Bin Gao, Renfeng Peng, and Ya-xiang Yuan,
%    arXiv preprint arXiv:2601.21874, 2026.
%    https://arxiv.org/abs/2601.21874
%
% Original author: Renfeng Peng, Aug. 05, 2026.

[PAGamma, Gamma, opts, hasTest] = parseOptionalInputs(varargin{:});

if exist('manopt_version', 'file') ~= 2
    error('TR_quotient_completion:ManoptNotFound', ...
        'Manopt is required. Run install.m or add Manopt to the MATLAB path.');
end
[d, n, r] = TR_validate_tensor(X, true);
SizeOmega = size(Omega, 1);
p = SizeOmega/prod(n);
PA = TR_validate_observations(Omega, PA, n, false, 'Training');
if norm(PA) == 0
    error('TR_quotient_completion:ZeroReference', ...
        'Relative errors require nonzero training reference values.');
end
if hasTest
    PAGamma = TR_validate_observations( ...
        Gamma, PAGamma, n, false, 'Test');
    if norm(PAGamma) == 0
        error('TR_quotient_completion:ZeroTestReference', ...
            'Relative test error requires nonzero PAGamma.');
    end
end

if ~isfield(opts, 'maxiter');    opts.maxiter = 100;       end
if ~isfield(opts, 'maxTime');    opts.maxTime = 1000;      end
if ~isfield(opts, 'gradtol');    opts.gradtol = 1e-8;      end
if ~isfield(opts, 'train_tol');  opts.train_tol = 0;       end
if ~isfield(opts, 'tol');        opts.tol = 0;             end
if ~isfield(opts, 'verbosity');  opts.verbosity = 2;       end
if ~isfield(opts, 'minstepsize'); opts.minstepsize = eps;  end

% The general quotient factory uses the same core-cell representation as
% the rest of LRTCTR, so no order-dependent reshaping is needed here.
M = TensorRingQuotient(n, r);
problem.M = M;
problem.cost = @cost;
problem.grad = @gradient;

options.maxiter = opts.maxiter;
options.maxtime = opts.maxTime;
options.tolgradnorm = opts.gradtol;
options.minstepsize = opts.minstepsize;
options.verbosity = opts.verbosity;
options.stopfun = @stopfun;
if hasTest
    options.statsfun = @statsfun;
end

X0.core = X.core;

switch upper(method)
    case 'RGD'
        [Xopt, ~, info] = steepestdescent(problem, X0, options);
    case 'RCG'
        options.beta_type = 'H-S';
        [Xopt, ~, info] = conjugategradient(problem, X0, options);
    otherwise
        error('TR_quotient_completion:Method', ...
            'Unknown quotient method "%s".', method);
end

Xnew.d = d;
Xnew.n = n;
Xnew.r = r;
Xnew.core = Xopt.core;
Xnew.Px = sampledValues(Xopt, Omega);
Xnew.error = norm(Xnew.Px - PA)^2;

duration = reshape([info.time], [], 1);
errorOmega = sqrt(max(0, 2*p*reshape([info.cost], [], 1)))/norm(PA);
if hasTest
    Xnew.PGamma = sampledValues(Xopt, Gamma);
    Xnew.errorGamma = norm(Xnew.PGamma - PAGamma)^2;
    errorGamma = reshape([info.errorGamma], [], 1);
else
    errorGamma = [];
end

    function costValue = cost(Xcurrent)
        % Evaluate the objective alone during line-search trial steps.
        residual = sampledValues(Xcurrent, Omega)-PA;
        costValue = 0.5*(residual'*residual)/p;
    end

    function grad = gradient(Xcurrent)
        % Evaluate all d core gradients only when Manopt requests them.
        [~, eucGrad] = TR_costgrad(makeTR(Xcurrent), Omega, PA);
        grad = M.proj(Xcurrent, eucGrad);
    end

    function stats = statsfun(~, Xcurrent, stats)
        valuesGamma = sampledValues(Xcurrent, Gamma);
        stats.errorGamma = norm(valuesGamma-PAGamma)/norm(PAGamma);
    end

    function stopnow = stopfun(~, ~, infoNow, last)
        errorOmegaNow = sqrt(max(0, 2*p*infoNow(last).cost))/norm(PA);
        stopnow = opts.train_tol > 0 && errorOmegaNow < opts.train_tol;
        if ~stopnow && opts.tol > 0 && last > 1
            costNow = infoNow(last).cost;
            costOld = infoNow(last-1).cost;
            relativeChange = abs(costNow - costOld) / ...
                max(abs(costNow), eps);
            stopnow = relativeChange < opts.tol;
        end
    end

    function values = sampledValues(Xcurrent, indices)
        values = TR_sample(makeTR(Xcurrent), indices);
    end

    function tensor = makeTR(Xcurrent)
        % Attach immutable metadata expected by the general TR utilities.
        tensor.d = d;
        tensor.n = n;
        tensor.r = r;
        tensor.core = Xcurrent.core;
    end

    function [valuesGamma, indicesGamma, optsParsed, hasTestData] = ...
            parseOptionalInputs(varargin)
        valuesGamma = [];
        indicesGamma = [];
        optsParsed = struct();
        switch numel(varargin)
            case 0
            case 1
                if ~isstruct(varargin{1})
                    error('TR_quotient_completion:OptionalInputs', ...
                        'The fourth input must be an options structure.');
                end
                optsParsed = varargin{1};
            case 2
                valuesGamma = varargin{1};
                indicesGamma = varargin{2};
            case 3
                valuesGamma = varargin{1};
                indicesGamma = varargin{2};
                optsParsed = varargin{3};
                if ~isstruct(optsParsed)
                    error('TR_quotient_completion:Options', ...
                        'The final input must be an options structure.');
                end
            otherwise
                error('TR_quotient_completion:OptionalInputs', ...
                    'Use (X, PA, Omega, opts) or add PAGamma and Gamma.');
        end
        hasTestData = ~isempty(valuesGamma) || ~isempty(indicesGamma);
        if hasTestData && (isempty(valuesGamma) || isempty(indicesGamma))
            error('TR_quotient_completion:TestPair', ...
                'PAGamma and Gamma must either both be supplied or both omitted.');
        end
    end
end
