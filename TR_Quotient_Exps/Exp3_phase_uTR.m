%EXP3_PHASE_UTR Estimate the phase transition of uTR-RCG(Q).

clear
clc

%% Add LRTCTR and Manopt to the MATLAB path
scriptDir = fileparts(mfilename('fullpath'));
packageDir = fileparts(scriptDir);
oldDir = pwd;
cd(packageDir)
run('install.m')
cd(oldDir)
resultsDir = fullfile(scriptDir, 'Results');

%% Initial settings
rng(20260804, 'twister')
d = 3;
r = 2;
ndims = 50:50:1500;
samples = 1000:1000:20000;
repeats = 10;
maxiter = 500;
opts = struct('maxiter', maxiter, 'maxTime', inf, 'gradtol', 1e-13, ...
    'train_tol', 1e-4, 'tol', 0, 'verbosity', 0, ...
    'minstepsize', eps);

%% Phase-transition experiment for uniform TR decomposition
% Each entry records the number of successful recoveries among all repeats.
successes = zeros(numel(ndims), numel(samples));
for j = 1:numel(ndims)
    n = ndims(j);
    for q = 1:numel(samples)
        % Skip sample sizes larger than the ambient tensor.
        if samples(q) > n^d
            continue
        end
        for t = 1:repeats
            % Generate an independent shared core and training set.
            Xtrue = struct('d', d, 'n', n, 'r', r, ...
                'U', randn(r, n, r));
            Omega = makeOmegaSet_mod(n*ones(1, d), samples(q));
            PA = uTR_sample(Xtrue, Omega);

            % Start uTR-RCG(Q) from an independent random shared core.
            X0 = struct('d', d, 'n', n, 'r', r, ...
                'U', randn(r, n, r));
            [~, ~, errorOmega] = uTR_RCGQ(X0, PA, Omega, opts);

            % Recovery is successful when the training error is below 1e-4.
            successes(j,q) = successes(j,q)+(errorOmega(end) < 1e-4);
        end

        % Save each completed grid point because the full test is expensive.
        save(fullfile(resultsDir, 'Result_Exp3_phase_uTR.mat'), ...
            'successes', 'ndims', 'samples', 'repeats', 'r', 'opts')
    end
end
