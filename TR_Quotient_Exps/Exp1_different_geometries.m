clear
clc
close all

%% Add LRTCTR and Manopt to the MATLAB path
scriptDir = fileparts(mfilename('fullpath'));
packageDir = fileparts(scriptDir);
oldDir = pwd;
cd(packageDir)
run('install.m')
cd(oldDir)
resultsDir = fullfile(scriptDir, 'Results');

%% Initial settings
rng(20260801, 'twister')
d = 3;
n = 100*ones(1, d);
r = 2*ones(1, d+1);
repeats = 5;
SizeOmega = 30000;
SizeGamma = 100;
epsilon = 1e-3;
opts.maxiter = 5000;
opts.maxTime = inf;
opts.gradtol = 1e-13;
opts.train_tol = 1e-12;
opts.tol = 0;
opts.verbosity = 0;
opts.minstepsize = eps;

%% Tensor completion under the general TR representation
resultsTR = cell(repeats, 1);
for t = 1:repeats
    % Generate a random ground-truth TR tensor.
    Xtrue = TR_randn(n, d, r);

    % Prescribe an unbalanced spectrum for the mode-2 unfolding of the
    % third core. This is the same construction as in the original code.
    [Q, ~] = qr(rand(n(3), r(3)*r(1)), 0);
    spectrum = diag(max(1.5.^(0:r(3)*r(1)-1), eps));
    mode2Core = Q(:, 1:r(3)*r(1))*spectrum;
    Xtrue.core{3} = reshape(mode2Core', ...
        [r(3), r(1), n(3)]);

    % Normalize the represented tensor without forming the full tensor.
    Xtrue.core{3} = Xtrue.core{3}/TR_norm(Xtrue);

    % Generate the training and test sets and evaluate sampled entries.
    Omega = makeOmegaSet_mod(n, SizeOmega);
    Gamma = makeOmegaSet_mod(n, SizeGamma);
    PA = TR_sample(Xtrue, Omega);
    PAGamma = TR_sample(Xtrue, Gamma);

    % Construct the initial point. The first two unfoldings are
    % orthogonalized after perturbation, while the last core is unconstrained.
    X0 = Xtrue;
    mode2Core = reshape(permute(Xtrue.core{1}, [3 1 2]), n(1), []);
    [mode2Core, ~] = qr(mode2Core+epsilon*randn(size(mode2Core)), 0);
    X0.core{1} = permute(reshape(mode2Core, ...
        [n(1), r(1), r(2)]), [2 3 1]);
    mode1Core = reshape(permute(Xtrue.core{2}, [1 3 2]), ...
        r(2)*n(2), r(3));
    [mode1Core, ~] = qr(mode1Core+epsilon*randn(size(mode1Core)), 0);
    X0.core{2} = permute(reshape(mode1Core, ...
        [r(2), n(2), r(3)]), [1 3 2]);
    X0.core{3} = Xtrue.core{3}+epsilon*randn(size(Xtrue.core{3}));

    % Compare RGD and RCG under the quotient geometry.
    result = struct();
    [~, result.duration_RGDQ, ~, result.errorGamma_RGDQ] = ...
        TR_RGDQ(X0, PA, Omega, PAGamma, Gamma, opts);
    [~, result.duration_RCGQ, ~, result.errorGamma_RCGQ] = ...
        TR_RCGQ(X0, PA, Omega, PAGamma, Gamma, opts);

    % Compare RGD and RCG under the Euclidean geometry.
    [~, result.duration_RGDE, result.errorGamma_RGDE] = ...
        Exp1_run_TR_E('RGD', X0, PA, Omega, PAGamma, Gamma, opts);
    [~, result.duration_RCGE, result.errorGamma_RCGE] = ...
        Exp1_run_TR_E('RCG', X0, PA, Omega, PAGamma, Gamma, opts);
    resultsTR{t} = result;
end

%% Tensor completion under the uniform TR representation
nUniform = n(1);
rUniform = r(1);
resultsUTR = cell(repeats, 1);
for t = 1:repeats
    % A uniform TR tensor uses one shared core of size r-by-n-by-r.
    Xtrue.d = d;
    Xtrue.n = nUniform;
    Xtrue.r = rUniform;
    Xtrue.U = randn(rUniform, nUniform, rUniform);

    % TR_norm expects general TR cores of size r-by-r-by-n. Convert the
    % shared core only for computing the norm of the represented tensor.
    XasTR.d = d;
    XasTR.n = nUniform*ones(1, d);
    XasTR.r = rUniform*ones(1, d+1);
    XasTR.core = repmat({permute(Xtrue.U, [1 3 2])}, d, 1);
    Xtrue.U = Xtrue.U/TR_norm(XasTR)^(1/d);

    % Generate the training and test sets and evaluate sampled entries.
    Omega = makeOmegaSet_mod(nUniform*ones(1,d), SizeOmega);
    Gamma = makeOmegaSet_mod(nUniform*ones(1,d), SizeGamma);
    PA = uTR_sample(Xtrue, Omega);
    PAGamma = uTR_sample(Xtrue, Gamma);

    % Perturb the normalized shared core to obtain the initial point.
    X0 = Xtrue;
    X0.U = X0.U+epsilon*randn(size(X0.U));

    % Compare RGD and RCG under the quotient geometry.
    result = struct();
    [~, result.duration_RGDQ, ~, result.errorGamma_RGDQ] = ...
        uTR_RGDQ(X0, PA, Omega, PAGamma, Gamma, opts);
    [~, result.duration_RCGQ, ~, result.errorGamma_RCGQ] = ...
        uTR_RCGQ(X0, PA, Omega, PAGamma, Gamma, opts);

    % Compare RGD and RCG under the Euclidean geometry.
    [~, result.duration_RGDE, result.errorGamma_RGDE] = ...
        Exp1_run_uTR_E('RGD', X0, PA, Omega, PAGamma, Gamma, opts);
    [~, result.duration_RCGE, result.errorGamma_RCGE] = ...
        Exp1_run_uTR_E('RCG', X0, PA, Omega, PAGamma, Gamma, opts);
    resultsUTR{t} = result;
end

%% Save and plot the results
save(fullfile(resultsDir, 'Result_Exp1_geometries.mat'), ...
    'resultsTR', 'resultsUTR', 'n', 'r', 'd', 'repeats', 'opts')
Exp1_different_geometries_results
