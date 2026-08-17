%EXP2_SCALING Test the influence of tensor order and TR rank.
%
% IMPORTANT: The full experiment solves many large-scale tensor completion
% problems and may take several hours. Set local_mode to 1 to use the
% intermediate-size configuration.

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
localMode = 0;
plotOnlyMode = 0;
if localMode
    repeats = 1;
    SizeGamma = 1000;
    opts.maxiter = 500;
    resultFile = fullfile(resultsDir, 'Result_Exp2_scaling_local.mat');
else
    repeats = 1;
    SizeGamma = 5000;
    opts.maxiter = 2000;
    resultFile = fullfile(resultsDir, 'Result_Exp2_scaling.mat');
end
oversampling = 10;
epsilon = 1e-2;
opts.maxTime = 300;
opts.gradtol = 0;
opts.train_tol = 1e-12;
opts.tol = 0;
opts.verbosity = 0;
opts.minstepsize = eps;

if plotOnlyMode
    % Load an existing result without rerunning the experiments.
    load(resultFile)
else
    fprintf(['Exp2 contains several large-scale tests and may take ', ...
        'several hours to finish.\n'])

    %% Experiment configurations
    % Test 1 compares convergence for one fifth-order tensor. Tests 2 and 3
    % vary the tensor order and TR rank, respectively.
    dLarge = 5;
    rLarge = 4;
    dValues = 3:7;
    rOrder = 4;
    dRank = 5;
    if localMode
        nLarge = 100;
        nOrder = 50;
        rValues = 2:2:6;
        nRank = 50;
    else
        nLarge = 500;
        nOrder = 100;
        rValues = 2:2:8;
        nRank = 100;
    end

    % Allocate one cell for every parameter choice and repetition.
    resultsLarge = cell(repeats, 1);
    resultsOrder = cell(numel(dValues), repeats);
    resultsRank = cell(numel(rValues), repeats);
    % Identify the large-scale, tensor-order, and TR-rank cases.
    caseType = [ones(1,repeats), 2*ones(1,numel(dValues)*repeats), ...
        3*ones(1,numel(rValues)*repeats)];
    caseIndex = [ones(1,repeats), repelem(1:numel(dValues), repeats), ...
        repelem(1:numel(rValues), repeats)];
    repeatIndex = [1:repeats, repmat(1:repeats, 1, numel(dValues)), ...
        repmat(1:repeats, 1, numel(rValues))];

    %% Run all completion tests
    for q = 1:numel(caseType)
        % Select the dimensions and random seed for the current test.
        t = repeatIndex(q);
        if caseType(q) == 1
            d = dLarge;
            nScalar = nLarge;
            rScalar = rLarge;
            rng(20260812+t, 'twister')
            fprintf('Large-scale test, d=%d, n=%d, r=%d, repeat=%d/%d\n', ...
                d, nScalar, rScalar, t, repeats)
        elseif caseType(q) == 2
            j = caseIndex(q);
            d = dValues(j);
            nScalar = nOrder;
            rScalar = rOrder;
            rng(20260810+100*j+t, 'twister')
            fprintf('Order test, d=%d, n=%d, r=%d, repeat=%d/%d\n', ...
                d, nScalar, rScalar, t, repeats)
        else
            j = caseIndex(q);
            d = dRank;
            nScalar = nRank;
            rScalar = rValues(j);
            rng(20260810+200*j+t, 'twister')
            fprintf('Rank test, d=%d, n=%d, r=%d, repeat=%d/%d\n', ...
                d, nScalar, rScalar, t, repeats)
        end
        % Construct a random ground-truth TR tensor with unit norm.
        n = nScalar*ones(1, d);
        r = rScalar*ones(1, d+1);
        assert(all(n >= r(1:d).*r(2:d+1)), ...
            'The full-rank condition n(k) >= r(k)r(k+1) is not satisfied.')
        Xtrue = TR_randn(n, d, r);
        normalization = TR_norm(Xtrue)^(1/d);
        for k = 1:d
            Xtrue.core{k} = Xtrue.core{k}/normalization;
        end

        % Use ten times the manifold dimension as the training sample size.
        dimM = sum(n.*r(1:d).*r(2:d+1))-sum(r(1:d).^2)+1;
        SizeOmega = ceil(oversampling*dimM);
        Omega = makeOmegaSet_mod(n, SizeOmega);
        Gamma = makeOmegaSet_mod(n, SizeGamma);
        PA = TR_sample(Xtrue, Omega);
        PAGamma = TR_sample(Xtrue, Gamma);

        % Perturb every core relative to its Frobenius norm.
        X0 = Xtrue;
        for k = 1:d
            perturbation = randn(size(X0.core{k}));
            perturbation = perturbation/norm(perturbation(:));
            X0.core{k} = X0.core{k}+epsilon*norm(X0.core{k}(:))*perturbation;
        end

        % Compare RGD and RCG under the quotient and Euclidean geometries.
        [X_RGDQ, duration_RGDQ, errorOmega_RGDQ, errorGamma_RGDQ, info_RGDQ] = ...
            TR_RGDQ(X0, PA, Omega, PAGamma, Gamma, opts);
        [X_RCGQ, duration_RCGQ, errorOmega_RCGQ, errorGamma_RCGQ, info_RCGQ] = ...
            TR_RCGQ(X0, PA, Omega, PAGamma, Gamma, opts);
        [X_RGDE, duration_RGDE, errorOmega_RGDE, errorGamma_RGDE, info_RGDE] = ...
            Exp2_run_E('RGD', false, X0, PA, Omega, PAGamma, Gamma, opts);
        [X_RCGE, duration_RCGE, errorOmega_RCGE, errorGamma_RCGE, info_RCGE] = ...
            Exp2_run_E('RCG', false, X0, PA, Omega, PAGamma, Gamma, opts);

        % Store the complete histories and final TR representations.
        result = struct('d', d, 'n', n, 'r', r, 'dimM', dimM, ...
            'SizeOmega', SizeOmega, 'SizeGamma', SizeGamma, ...
            'X_RGDQ', X_RGDQ, 'duration_RGDQ', duration_RGDQ, ...
            'errorOmega_RGDQ', errorOmega_RGDQ, 'errorGamma_RGDQ', errorGamma_RGDQ, 'info_RGDQ', info_RGDQ, ...
            'X_RCGQ', X_RCGQ, 'duration_RCGQ', duration_RCGQ, ...
            'errorOmega_RCGQ', errorOmega_RCGQ, 'errorGamma_RCGQ', errorGamma_RCGQ, 'info_RCGQ', info_RCGQ, ...
            'X_RGDE', X_RGDE, 'duration_RGDE', duration_RGDE, ...
            'errorOmega_RGDE', errorOmega_RGDE, 'errorGamma_RGDE', errorGamma_RGDE, 'info_RGDE', info_RGDE, ...
            'X_RCGE', X_RCGE, 'duration_RCGE', duration_RCGE, ...
            'errorOmega_RCGE', errorOmega_RCGE, 'errorGamma_RCGE', errorGamma_RCGE, 'info_RCGE', info_RCGE);
        if caseType(q) == 1
            resultsLarge{t} = result;
        elseif caseType(q) == 2
            resultsOrder{j,t} = result;
        else
            resultsRank{j,t} = result;
        end

        % Save every completed case in case the long run is interrupted.
        save(resultFile, 'resultsLarge', 'resultsOrder', 'resultsRank', ...
            'dLarge', 'nLarge', 'rLarge', 'dValues', 'nOrder', 'rOrder', ...
            'rValues', 'dRank', 'nRank', 'repeats', 'oversampling', ...
            'SizeGamma', 'epsilon', 'opts')
    end

    %% Average iteration times
    % Take differences of cumulative solver times and average over iterations.
    fields = {'RGDQ', 'RCGQ', 'RGDE', 'RCGE'};
    for a = 1:numel(fields)
        name = fields{a};
        iterationTimeOrder.(name) = zeros(numel(dValues), repeats);
        iterationTimeRank.(name) = zeros(numel(rValues), repeats);
        for j = 1:numel(dValues)
            for t = 1:repeats
                iterationTimeOrder.(name)(j,t) = mean(diff(resultsOrder{j,t}.(['duration_' name])));
            end
        end
        for j = 1:numel(rValues)
            for t = 1:repeats
                iterationTimeRank.(name)(j,t) = mean(diff(resultsRank{j,t}.(['duration_' name])));
            end
        end
    end
    iterationTimeOrder_RGDQ = iterationTimeOrder.RGDQ;
    iterationTimeOrder_RCGQ = iterationTimeOrder.RCGQ;
    iterationTimeOrder_RGDE = iterationTimeOrder.RGDE;
    iterationTimeOrder_RCGE = iterationTimeOrder.RCGE;
    iterationTimeRank_RGDQ = iterationTimeRank.RGDQ;
    iterationTimeRank_RCGQ = iterationTimeRank.RCGQ;
    iterationTimeRank_RGDE = iterationTimeRank.RGDE;
    iterationTimeRank_RCGE = iterationTimeRank.RCGE;
    % Save the timing arrays used by the plotting script.
    save(resultFile)
end

%% Plot the convergence and scaling results
Exp2_scaling_results
