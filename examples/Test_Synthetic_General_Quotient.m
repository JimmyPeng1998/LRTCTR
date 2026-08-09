clear
clc

% General-order examples for TR-RGD(Q) and TR-RCG(Q).
% The fourth-order test uses nonuniform mode sizes and TR ranks. The
% fifth-order test verifies that the algorithms are not tied to d = 3 or 4.
% The training-set size is ten times the quotient-manifold dimension, and
% an independent test set measures recovery. All entries are evaluated
% directly from the TR cores; no full tensor is constructed.
%
% Reference: Quotient geometry of tensor ring decomposition,
%    Bin Gao, Renfeng Peng, and Ya-xiang Yuan,
%    arXiv preprint arXiv:2601.21874, 2026.
%    https://arxiv.org/abs/2601.21874
%
% Original author: Renfeng Peng, Aug. 05, 2026.

%% Creating problems
cases = { ...
    struct('n', [20 20 20 20], 'r', [2 3 2 2 2]), ...
    struct('n', [100 100 100 100 100], 'r', [2 2 2 2 2 2])};
results = cell(numel(cases), 1);

opts.maxiter = 2000;
opts.maxTime = 300;
opts.gradtol = 0;
opts.train_tol = 1e-12;
opts.tol = 0;
opts.verbosity = 1;
opts.minstepsize = eps;
oversampling = 10;
SizeGamma = 2000;

%% Run TR-RGDQ & TR-RCGQ
figure
layout = tiledlayout(1, numel(cases), 'TileSpacing', 'compact');
for k = 1:numel(cases)
    % Reset the stream so that each order is reproducible on its own and
    % does not depend on how many random numbers an earlier case consumed.
    rng(20260808, 'twister')
    n = cases{k}.n;
    r = cases{k}.r;
    d = numel(n);

    % Injectivity requires n(k) >= r(k)*r(k+1), so that the mode-2
    % unfolding of a generic core can have full column rank.
    assert(all(n >= r(1:d).*r(2:d+1)), ...
        'The selected dimensions do not satisfy the injectivity condition.')

    Xtrue = randomTR(n, r);
    X0 = Xtrue;
    for j = 1:d
        X0.core{j} = X0.core{j}+1e-2*randn(size(X0.core{j}));
    end

    dimM = sum(n.*r(1:d).*r(2:d+1)) - ...
        sum(r(1:d).^2)+1;
    SizeOmega = oversampling*dimM;
    subs = randomUniqueIndices(n, SizeOmega+SizeGamma);
    Omega = subs(1:SizeOmega, :);
    Gamma = subs(SizeOmega+1:end, :);
    PA = TR_sample(Xtrue, Omega);
    PAGamma = TR_sample(Xtrue, Gamma);

    [X_RGDQ, duration_RGDQ, errorOmega_RGDQ, errorGamma_RGDQ, info_RGDQ] = ...
        TR_RGDQ(X0, PA, Omega, PAGamma, Gamma, opts);
    [X_RCGQ, duration_RCGQ, errorOmega_RCGQ, errorGamma_RCGQ, info_RCGQ] = ...
        TR_RCGQ(X0, PA, Omega, PAGamma, Gamma, opts);

    final_errorOmega_RGDQ = norm(TR_sample(X_RGDQ, Omega)-PA)/norm(PA);
    final_errorOmega_RCGQ = norm(TR_sample(X_RCGQ, Omega)-PA)/norm(PA);
    final_errorGamma_RGDQ = norm(TR_sample(X_RGDQ, Gamma)-PAGamma)/norm(PAGamma);
    final_errorGamma_RCGQ = norm(TR_sample(X_RCGQ, Gamma)-PAGamma)/norm(PAGamma);
    assert(final_errorOmega_RGDQ < 1e-12 && final_errorOmega_RCGQ < 1e-12 && ...
            final_errorGamma_RGDQ < 3e-12 && final_errorGamma_RCGQ < 3e-12, ...
        'The order-%d completion test did not reach the requested tolerance.', d);

    results{k} = struct('d', d, 'n', n, 'r', r, ...
        'manifoldDimension', dimM, ...
        'oversampling', oversampling, 'trainingCount', SizeOmega, ...
        'testCount', SizeGamma, ...
        'duration_RGDQ', duration_RGDQ, 'errorOmega_RGDQ', errorOmega_RGDQ, ...
        'errorGamma_RGDQ', errorGamma_RGDQ, ...
        'duration_RCGQ', duration_RCGQ, 'errorOmega_RCGQ', errorOmega_RCGQ, ...
        'errorGamma_RCGQ', errorGamma_RCGQ, ...
        'final_errorOmega_RGDQ', final_errorOmega_RGDQ, ...
        'final_errorOmega_RCGQ', final_errorOmega_RCGQ, ...
        'final_errorGamma_RGDQ', final_errorGamma_RGDQ, ...
        'final_errorGamma_RCGQ', final_errorGamma_RCGQ, ...
        'iterationsRGDQ', numel(info_RGDQ)-1, ...
        'iterationsRCGQ', numel(info_RCGQ)-1);

    nexttile
    semilogy(duration_RGDQ, errorGamma_RGDQ, 'LineWidth', 1.5)
    hold on
    semilogy(duration_RCGQ, errorGamma_RCGQ, 'LineWidth', 1.5)
    xlabel('Time (seconds)')
    ylabel('Relative test error')
    title(sprintf('Order d = %d', d))
    grid on

    fprintf(['d=%d, dim=%d, |Omega|=%d: TR-RGD(Q) test %.3e ', ...
        '(%d iterations), TR-RCG(Q) test %.3e (%d iterations)\n'], ...
        d, dimM, SizeOmega, final_errorGamma_RGDQ, ...
        numel(info_RGDQ)-1, final_errorGamma_RCGQ, numel(info_RCGQ)-1)
end
legend(layout.Children(end), {'TR-RGD(Q)', 'TR-RCG(Q)'}, ...
    'Location', 'best')

exampleDir = fileparts(mfilename('fullpath'));
save(fullfile(exampleDir, 'Result_Synthetic_General_Quotient.mat'), ...
    'results')
exportgraphics(gcf, ...
    fullfile(exampleDir, 'Result_Synthetic_General_Quotient.png'), ...
    'Resolution', 180)

%% Auxiliary functions
function X = randomTR(n, r)
d = numel(n);
X.d = d;
X.n = n;
X.r = r;
X.core = cell(d, 1);
for k = 1:d
    X.core{k} = randn(r(k), r(k+1), n(k))/sqrt(n(k)*r(k));
end
end

function Omega = randomUniqueIndices(n, count)
% Draw uniformly without constructing the full Cartesian index set.
totalEntries = prod(n);
if count > totalEntries
    error('The requested sample count exceeds the number of tensor entries.')
end
linearInds = zeros(0, 1);
while numel(linearInds) < count
    remaining = count-numel(linearInds);
    candidates = randi(totalEntries, max(2*remaining, 100), 1);
    linearInds = unique([linearInds; candidates], 'stable');
end
linearInds = linearInds(1:count);
d = numel(n);
multiInds = cell(1, d);
[multiInds{:}] = ind2sub(n, linearInds);
Omega = zeros(count, d);
for k = 1:d
    Omega(:, k) = multiInds{k};
end
end
