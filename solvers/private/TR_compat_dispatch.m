function [Xnew, duration, errorOmega, errorGamma] = ...
    TR_compat_dispatch(legacySolver, X, PA, Omega, varargin)
%TR_COMPAT_DISPATCH Support both legacy and simplified solver interfaces.
%
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
% Original author: Renfeng Peng, Aug. 05, 2026.

% Legacy interface after Omega:
%   SizeOmega, PAGamma, Gamma, SizeGamma, p, opts
if numel(varargin) == 6
    [Xnew, duration, errorOmega, errorGamma] = legacySolver( ...
        X, PA, Omega, varargin{:});
    return
end

[PAGamma, Gamma, opts, hasTest] = parseNewInputs(varargin{:});
[~, n] = TR_validate_tensor(X, false);
PA = TR_validate_observations(Omega, PA, n, false, 'Training');
SizeOmega = size(Omega, 1);
p = SizeOmega/prod(n);
X.Px = TR_sample(X, Omega);
X.error = norm(X.Px - PA)^2;

if ~isfield(opts, 'lambda')
    opts.lambda = 1e-10;
end
X.normofCores = 0;
for k = 1:X.d
    X.normofCores = X.normofCores + ...
        0.5*opts.lambda*norm(X.core{k}(:))^2;
end

if hasTest
    PAGamma = TR_validate_observations(Gamma, PAGamma, n, false, 'Test');
    % In the simplified interface, test data are statistics only.
    opts.err = 0;
else
    % The legacy implementation expects a monitoring set. Reuse the
    % training set internally, then remove this proxy statistic on return.
    Gamma = Omega;
    PAGamma = PA;
    if isfield(opts, 'train_tol')
        opts.err = opts.train_tol;
    else
        opts.err = 0;
    end
end

SizeGamma = size(Gamma, 1);
X.PGamma = TR_sample(X, Gamma);
X.errorGamma = norm(X.PGamma - PAGamma)^2;

[Xnew, duration, errorOmega, errorGamma] = legacySolver( ...
    X, PA, Omega, SizeOmega, PAGamma, Gamma, SizeGamma, p, opts);

if ~hasTest
    errorGamma = [];
    fieldsToRemove = intersect({'PGamma', 'errorGamma'}, fieldnames(Xnew));
    if ~isempty(fieldsToRemove)
        Xnew = rmfield(Xnew, fieldsToRemove);
    end
end
end

function [PAGamma, Gamma, opts, hasTest] = parseNewInputs(varargin)
PAGamma = [];
Gamma = [];
opts = struct();
switch numel(varargin)
    case 0
    case 1
        if ~isstruct(varargin{1})
            error('TR_compat_dispatch:Options', ...
                'The fourth input must be an options structure.');
        end
        opts = varargin{1};
    case 2
        PAGamma = varargin{1};
        Gamma = varargin{2};
    case 3
        PAGamma = varargin{1};
        Gamma = varargin{2};
        opts = varargin{3};
        if ~isstruct(opts)
            error('TR_compat_dispatch:Options', ...
                'The final input must be an options structure.');
        end
    otherwise
        error('TR_compat_dispatch:Inputs', ...
            'Use the documented legacy or simplified solver interface.');
end
hasTest = ~isempty(PAGamma) || ~isempty(Gamma);
if hasTest && (isempty(PAGamma) || isempty(Gamma))
    error('TR_compat_dispatch:TestPair', ...
        'PAGamma and Gamma must either both be supplied or both omitted.');
end
end
