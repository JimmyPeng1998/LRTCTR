%EXP3_PHASE_TR Estimate the phase transition of TR-RCG(Q).

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
rng(20260803, 'twister')
d = 3;
r = 2*ones(1, d+1);
ndims = 50:10:200;
samples = 2000:2000:100000;
repeats = 10;
maxiter = 1000;
opts = struct('maxiter', maxiter, 'maxTime', inf, 'gradtol', 1e-13, ...
    'train_tol', 1e-4, 'tol', 0, 'verbosity', 0, ...
    'minstepsize', eps);

%% Phase-transition experiment for general TR decomposition
% Each entry records the number of successful recoveries among all repeats.
successes = zeros(numel(ndims), numel(samples));
for j = 1:numel(ndims)
    % Use equal mode sizes and fixed TR ranks.
    n = ndims(j)*ones(1, d);
    for q = 1:numel(samples)
        % Skip sample sizes larger than the ambient tensor.
        if samples(q) > prod(n)
            continue
        end
        for t = 1:repeats
            % Generate an independent ground-truth tensor and training set.
            Xtrue = TR_randn(n, d, r);
            Omega = makeOmegaSet_mod(n, samples(q));
            PA = TR_sample(Xtrue, Omega);

            % Start TR-RCG(Q) from an independent random initial point.
            X0 = TR_randn(n, d, r);
            [~, ~, errorOmega] = TR_RCGQ(X0, PA, Omega, opts);

            % Recovery is successful when the training error is below 1e-4.
            successes(j,q) = successes(j,q)+(errorOmega(end) < 1e-4);
        end

        % Save each completed grid point because the full test is expensive.
        save(fullfile(resultsDir, 'Result_Exp3_phase_TR.mat'), ...
            'successes', 'ndims', 'samples', 'repeats', 'r', 'opts')
    end
end
