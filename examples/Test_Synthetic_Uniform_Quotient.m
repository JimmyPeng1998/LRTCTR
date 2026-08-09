clear
clc

% General-order examples for uTR-RGD(Q) and uTR-RCG(Q).
% The fourth- and fifth-order cases verify that the uniform quotient
% algorithms are not restricted to d = 3. The training-set size is ten
% times the quotient-manifold dimension, and an independent test set
% measures recovery. No full tensor is constructed.
%
% Reference: Quotient geometry of tensor ring decomposition,
%    Bin Gao, Renfeng Peng, and Ya-xiang Yuan,
%    arXiv preprint arXiv:2601.21874, 2026.
%    https://arxiv.org/abs/2601.21874
%
% Original author: Renfeng Peng, Aug. 05, 2026.

%% Creating problems
cases = { ...
    struct('d', 4, 'n', 20, 'r', 2), ...
    struct('d', 5, 'n', 100, 'r', 2)};
results = cell(numel(cases), 1);

opts.maxiter = 5000;
opts.maxTime = 300;
opts.gradtol = 0;
opts.train_tol = 5e-13;
opts.tol = 0;
opts.verbosity = 1;
opts.minstepsize = eps;
oversampling = 10;
SizeGamma = 2000;

%% Run uTR-RGDQ & uTR-RCGQ
figure
layout = tiledlayout(1, numel(cases), 'TileSpacing', 'compact');
for k = 1:numel(cases)
    % Reset the stream so that each order is reproducible on its own.
    rng(20260808, 'twister')
    d = cases{k}.d;
    n = cases{k}.n;
    r = cases{k}.r;

    % Injectivity of the shared core requires n >= r^2.
    assert(n >= r^2, ...
        'The selected dimensions do not satisfy the injectivity condition.')

    Xtrue.d = d;
    Xtrue.n = n;
    Xtrue.r = r;
    Xtrue.U = randn(r, n, r)/sqrt(n*r);

    X0 = Xtrue;
    X0.U = X0.U+1e-2*randn(size(X0.U));

    dimM = n*r^2-r^2+1;
    SizeOmega = oversampling*dimM;
    subs = randomUniqueIndices( ...
        n*ones(1, d), SizeOmega+SizeGamma);
    Omega = subs(1:SizeOmega, :);
    Gamma = subs(SizeOmega+1:end, :);
    PA = uTR_sample(Xtrue, Omega);
    PAGamma = uTR_sample(Xtrue, Gamma);

    [X_RGDQ, duration_RGDQ, errorOmega_RGDQ, errorGamma_RGDQ, info_RGDQ] = ...
        uTR_RGDQ(X0, PA, Omega, PAGamma, Gamma, opts);
    [X_RCGQ, duration_RCGQ, errorOmega_RCGQ, errorGamma_RCGQ, info_RCGQ] = ...
        uTR_RCGQ(X0, PA, Omega, PAGamma, Gamma, opts);

    final_errorOmega_RGDQ = norm(uTR_sample(X_RGDQ, Omega)-PA)/norm(PA);
    final_errorOmega_RCGQ = norm(uTR_sample(X_RCGQ, Omega)-PA)/norm(PA);
    final_errorGamma_RGDQ = norm(uTR_sample(X_RGDQ, Gamma)-PAGamma)/norm(PAGamma);
    final_errorGamma_RCGQ = norm(uTR_sample(X_RCGQ, Gamma)-PAGamma)/norm(PAGamma);
    assert(final_errorOmega_RGDQ < 1e-12 && final_errorOmega_RCGQ < 1e-12 && ...
            final_errorGamma_RGDQ < 3e-12 && final_errorGamma_RCGQ < 3e-12, ...
        'The order-%d uniform completion test did not reach the tolerance.', d)

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

    fprintf(['d=%d, dim=%d, |Omega|=%d: uTR-RGD(Q) test %.3e ', ...
        '(%d iterations), uTR-RCG(Q) test %.3e (%d iterations)\n'], ...
        d, dimM, SizeOmega, final_errorGamma_RGDQ, ...
        numel(info_RGDQ)-1, final_errorGamma_RCGQ, numel(info_RCGQ)-1)
end
legend(layout.Children(end), {'uTR-RGD(Q)', 'uTR-RCG(Q)'}, ...
    'Location', 'best')

exampleDir = fileparts(mfilename('fullpath'));
save(fullfile(exampleDir, 'Result_Synthetic_Uniform_Quotient.mat'), ...
    'results')
exportgraphics(gcf, ...
    fullfile(exampleDir, 'Result_Synthetic_Uniform_Quotient.png'), ...
    'Resolution', 180)

%% Auxiliary functions
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
