clear
clc
close all
scriptDir = fileparts(mfilename('fullpath'));
packageDir = fileparts(scriptDir);
oldDir = pwd;
cd(packageDir)
run('install.m')
cd(oldDir)
resultsDir = fullfile(scriptDir, 'Results');


%% Dataset
rng(20260824)
n = [6040 3952 150];
d = 3;
r = [6 10 3 6];

ratingsFile = fullfile(scriptDir, 'ratings.dat');
fileID = fopen(ratingsFile);
if fileID < 0
    error('Please place ratings.dat in the same folder as this script.')
end
data = textscan(fileID, '%d %d %d %d', 'Delimiter', '::');
fclose(fileID);
data1 = reshape(data{1}, [2 1000209])';
data2 = reshape(data{3}, [2 1000209])';
dat = double([data1 data2]);                       % user rating movie time
dat(:, 4) = floor((dat(:, 4)-datenum(1970,1,1))/604800)-1579;
dat = dat(:, [1 3 4 2]);                          % user movie time rating
assert(min(dat(:, 3)) == 1 && max(dat(:, 3)) == 150)

indices = randperm(size(dat, 1));
SizeOmega = 800000;
Omega = dat(indices(1:SizeOmega), 1:3);
PA = dat(indices(1:SizeOmega), 4);
Gamma = dat(indices(SizeOmega+1:end), 1:3);
PAGamma = dat(indices(SizeOmega+1:end), 4);
SizeGamma = size(Gamma, 1);


%% Initial setting
X0 = TR_randn(n, d, r);
PX0 = TR_sample(X0, Omega);
X0.core{1} = X0.core{1}*norm(PA)/norm(PX0);

opts.maxiter = 5;
opts.maxTime = 500;
opts.gradtol = 0;
opts.train_tol = 0;
opts.tol = 0;
opts.err = 0;
opts.lambda = 1e-10;
opts.verbosity = 1;


%% Run algorithms
disp('TR-RGD(Q) starts')
[X_RGDQ, duration_RGDQ, errorOmega_RGDQ, errorGamma_RGDQ] = ...
    TR_RGDQ(X0, PA, Omega, PAGamma, Gamma, opts);

disp('TR-RCG(Q) starts')
[X_RCGQ, duration_RCGQ, errorOmega_RCGQ, errorGamma_RCGQ] = ...
    TR_RCGQ(X0, PA, Omega, PAGamma, Gamma, opts);

disp('TR-RGD(precon) starts')
[X_RGDprecon, duration_RGDprecon, ...
    errorOmega_RGDprecon, errorGamma_RGDprecon] = ...
    TR_RGD_RBB2(X0, PA, Omega, PAGamma, Gamma, opts);

disp('TR-RCG(precon) starts')
[X_RCGprecon, duration_RCGprecon, ...
    errorOmega_RCGprecon, errorGamma_RCGprecon] = ...
    TR_RCG_HS(X0, PA, Omega, PAGamma, Gamma, opts);


%% Test RMSE
scaleGamma = norm(PAGamma)/sqrt(SizeGamma);
rmseGamma_RGDQ = scaleGamma*errorGamma_RGDQ;
rmseGamma_RCGQ = scaleGamma*errorGamma_RCGQ;
rmseGamma_RGDprecon = scaleGamma*errorGamma_RGDprecon;
rmseGamma_RCGprecon = scaleGamma*errorGamma_RCGprecon;


%% Full-rank condition
singularValues_RGDQ = cell(d, 1);
singularValues_RCGQ = cell(d, 1);
singularValues_RGDprecon = cell(d, 1);
singularValues_RCGprecon = cell(d, 1);

fprintf('\nSingular values of the mode-2 core unfoldings\n')
fprintf('\nTR-RGD(Q)\n')
for k = 1:d
    unfolding = reshape(permute(X_RGDQ.core{k}, [3 1 2]), n(k), []);
    singularValues_RGDQ{k} = svd(unfolding, 'econ');
    fprintf('Core %d:\n', k)
    fprintf('%.6e  ', singularValues_RGDQ{k})
    fprintf('\n')
end

fprintf('\nTR-RCG(Q)\n')
for k = 1:d
    unfolding = reshape(permute(X_RCGQ.core{k}, [3 1 2]), n(k), []);
    singularValues_RCGQ{k} = svd(unfolding, 'econ');
    fprintf('Core %d:\n', k)
    fprintf('%.6e  ', singularValues_RCGQ{k})
    fprintf('\n')
end

fprintf('\nTR-RGD(precon)\n')
for k = 1:d
    unfolding = reshape(permute(X_RGDprecon.core{k}, [3 1 2]), n(k), []);
    singularValues_RGDprecon{k} = svd(unfolding, 'econ');
    fprintf('Core %d:\n', k)
    fprintf('%.6e  ', singularValues_RGDprecon{k})
    fprintf('\n')
end

fprintf('\nTR-RCG(precon)\n')
for k = 1:d
    unfolding = reshape(permute(X_RCGprecon.core{k}, [3 1 2]), n(k), []);
    singularValues_RCGprecon{k} = svd(unfolding, 'econ');
    fprintf('Core %d:\n', k)
    fprintf('%.6e  ', singularValues_RCGprecon{k})
    fprintf('\n')
end


%% Save results
save(fullfile(resultsDir, 'Result_Exp4_movielens.mat'), ...
    'n', 'd', 'r', 'SizeOmega', 'SizeGamma', ...
    'X_RGDQ', 'duration_RGDQ', 'errorOmega_RGDQ', ...
    'errorGamma_RGDQ', 'rmseGamma_RGDQ', ...
    'X_RCGQ', 'duration_RCGQ', 'errorOmega_RCGQ', ...
    'errorGamma_RCGQ', 'rmseGamma_RCGQ', ...
    'X_RGDprecon', 'duration_RGDprecon', 'errorOmega_RGDprecon', ...
    'errorGamma_RGDprecon', 'rmseGamma_RGDprecon', ...
    'X_RCGprecon', 'duration_RCGprecon', 'errorOmega_RCGprecon', ...
    'errorGamma_RCGprecon', 'rmseGamma_RCGprecon', ...
    'singularValues_RGDQ', 'singularValues_RCGQ', ...
    'singularValues_RGDprecon', 'singularValues_RCGprecon')

Exp4_movielens_results