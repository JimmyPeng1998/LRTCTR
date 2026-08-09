function [Xnew, duration, errorOmega, errorGamma, info] = ...
    uTR_quotient_completion(method, X, PA, Omega, varargin)
%UTR_QUOTIENT_COMPLETION Shared Manopt driver for uniform TR solvers.
%
% Reference: Quotient geometry of tensor ring decomposition,
%    Bin Gao, Renfeng Peng, and Ya-xiang Yuan,
%    arXiv preprint arXiv:2601.21874, 2026.
%    https://arxiv.org/abs/2601.21874
%
% Original author: Renfeng Peng, Aug. 05, 2026.

[PAGamma, Gamma, opts, hasTest] = parseOptionalInputs(varargin{:});

if exist('manopt_version', 'file') ~= 2
    error('uTR_quotient_completion:ManoptNotFound', ...
        'Manopt is required. Run install.m before using this solver.');
end
requiredFields = {'d', 'n', 'r', 'U'};
if ~isstruct(X) || ~isscalar(X) || ~all(isfield(X, requiredFields))
    error('uTR_quotient_completion:InitialPoint', ...
        'X must contain the fields d, n, r and U.');
end

d = double(X.d);
n = double(X.n);
r = double(X.r);
M = UniTensorRingQuotient(d, n, r);
if ~isa(X.U, 'double') || ~isreal(X.U) || ...
        size(X.U, 1) ~= r || size(X.U, 2) ~= n || ...
        size(X.U, 3) ~= r || numel(X.U) ~= r*n*r
    error('uTR_quotient_completion:CoreSize', ...
        'X.U must be a real double array of size r-by-n-by-r.');
end
if ~isnumeric(Omega) || ~isreal(Omega) || ~ismatrix(Omega) || ...
        size(Omega, 2) ~= d || isempty(Omega) || ...
        any(~isfinite(Omega(:))) || any(Omega(:) < 1) || ...
        any(Omega(:) ~= floor(Omega(:))) || any(Omega(:) > n)
    error('uTR_quotient_completion:Indices', ...
        'Omega must contain valid tensor indices in a nonempty m-by-d matrix.');
end
PA = PA(:);
SizeOmega = size(Omega, 1);
p = SizeOmega/(n^d);
if ~isa(PA, 'double') || ~isreal(PA) || ...
        any(~isfinite(PA)) || numel(PA) ~= SizeOmega
    error('uTR_quotient_completion:ObservedValues', ...
        'PA must be a finite real double vector matching Omega.');
end
if norm(PA) == 0
    error('uTR_quotient_completion:ZeroReference', ...
        'Relative errors require nonzero training reference values.');
end
if hasTest
    PAGamma = PAGamma(:);
    if ~isnumeric(Gamma) || ~isreal(Gamma) || ~ismatrix(Gamma) || ...
            size(Gamma, 2) ~= d || isempty(Gamma) || ...
            any(~isfinite(Gamma(:))) || any(Gamma(:) < 1) || ...
            any(Gamma(:) ~= floor(Gamma(:))) || any(Gamma(:) > n) || ...
            ~isa(PAGamma, 'double') || ~isreal(PAGamma) || ...
            any(~isfinite(PAGamma)) || numel(PAGamma) ~= size(Gamma, 1)
        error('uTR_quotient_completion:TestValues', ...
            ['Gamma must contain valid m-by-d indices, and PAGamma must ', ...
             'be a matching finite real double vector.']);
    end
    if norm(PAGamma) == 0
        error('uTR_quotient_completion:ZeroTestReference', ...
            'Relative test error requires nonzero PAGamma.');
    end
end

if ~isfield(opts, 'maxiter');     opts.maxiter = 100;       end
if ~isfield(opts, 'maxTime');     opts.maxTime = 1000;      end
if ~isfield(opts, 'gradtol');     opts.gradtol = 1e-8;      end
if ~isfield(opts, 'train_tol');   opts.train_tol = 0;       end
if ~isfield(opts, 'tol');         opts.tol = 0;             end
if ~isfield(opts, 'verbosity');   opts.verbosity = 2;       end
if ~isfield(opts, 'minstepsize'); opts.minstepsize = eps;   end

problem.M = M;
problem.costgrad = @costgrad;

options.maxiter = opts.maxiter;
options.maxtime = opts.maxTime;
options.tolgradnorm = opts.gradtol;
options.minstepsize = opts.minstepsize;
options.verbosity = opts.verbosity;
options.stopfun = @stopfun;
if hasTest
    options.statsfun = @statsfun;
end

X0.U = X.U;
switch upper(method)
    case 'RGD'
        [Xopt, ~, info] = steepestdescent(problem, X0, options);
    case 'RCG'
        options.beta_type = 'H-S';
        [Xopt, ~, info] = conjugategradient(problem, X0, options);
    otherwise
        error('uTR_quotient_completion:Method', ...
            'Unknown uniform quotient method "%s".', method);
end

Xnew.d = d;
Xnew.n = n;
Xnew.r = r;
Xnew.U = Xopt.U;
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

    function [cost, grad] = costgrad(Xcurrent)
        [cost, eucGrad] = TR_uniform_costgrad( ...
            Xcurrent, Omega, PA, p, d, n, r);
        grad = M.proj(Xcurrent, eucGrad);
    end

    function stats = statsfun(~, Xcurrent, stats)
        stats.errorGamma = TR_uniform_relative_error( ...
            Xcurrent, Gamma, PAGamma, d, n, r);
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
        Xsampled.d = d;
        Xsampled.n = n;
        Xsampled.r = r;
        Xsampled.U = Xcurrent.U;
        values = uTR_sample(Xsampled, indices);
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
                    error('uTR_quotient_completion:OptionalInputs', ...
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
                    error('uTR_quotient_completion:Options', ...
                        'The final input must be an options structure.');
                end
            otherwise
                error('uTR_quotient_completion:OptionalInputs', ...
                    'Use (X, PA, Omega, opts) or add PAGamma and Gamma.');
        end
        hasTestData = ~isempty(valuesGamma) || ~isempty(indicesGamma);
        if hasTestData && (isempty(valuesGamma) || isempty(indicesGamma))
            error('uTR_quotient_completion:TestPair', ...
                'PAGamma and Gamma must either both be supplied or both omitted.');
        end
    end
end
