clear
clc

% Synthetic example for TR-RGD(Q) and TR-RCG(Q).
%
% Reference: Quotient geometry of tensor ring decomposition,
%    Bin Gao, Renfeng Peng, and Ya-xiang Yuan,
%    arXiv preprint arXiv:2601.21874, 2026.
%    https://arxiv.org/abs/2601.21874
%
% Original author: Renfeng Peng, Aug. 05, 2026.

%% Creating problem
rng(1)
n = [20 20 20];
d = 3;
r = [2 2 2 2];

Xtrue = TR_randn(n, d, r);

SizeOmega = 800;
SizeGamma = 200;
Omega = makeOmegaSet_mod(n, SizeOmega);
Gamma = makeOmegaSet_mod(n, SizeGamma);
PA = TR_sample(Xtrue, Omega);
PAGamma = TR_sample(Xtrue, Gamma);

%% Initial setting
X0 = Xtrue;
for k = 1:d
    X0.core{k} = X0.core{k} + 1e-2*randn(size(X0.core{k}));
end

opts.maxiter = 5000;
opts.maxTime = 1800;
opts.gradtol = 1e-14;
opts.train_tol = 1e-12;
opts.tol = 0;
opts.verbosity = 1;

%% Run TR-RGDQ & TR-RCGQ
[X_RGDQ, duration_RGDQ, errorOmega_RGDQ, errorGamma_RGDQ, info_RGDQ] = TR_RGDQ( ...
    X0, PA, Omega, PAGamma, Gamma, opts);
[X_RCGQ, duration_RCGQ, errorOmega_RCGQ, errorGamma_RCGQ, info_RCGQ] = TR_RCGQ( ...
    X0, PA, Omega, PAGamma, Gamma, opts);

%% Results
semilogy(duration_RGDQ, errorGamma_RGDQ, 'LineWidth', 1.5)
hold on
semilogy(duration_RCGQ, errorGamma_RCGQ, 'LineWidth', 1.5)
xlabel('Time (seconds)')
ylabel('Relative test error')
legend('TR-RGD(Q)', 'TR-RCG(Q)', 'Location', 'best')
grid on

fprintf('TR-RGD(Q) final relative test error: %.3e\n', errorGamma_RGDQ(end))
fprintf('TR-RCG(Q) final relative test error: %.3e\n', errorGamma_RCGQ(end))

resultFile = fullfile(fileparts(mfilename('fullpath')), ...
    'Result_Synthetic_Quotient.mat');
figureFile = fullfile(fileparts(mfilename('fullpath')), ...
    'Result_Synthetic_Quotient.png');
save(resultFile, 'X_RGDQ', 'duration_RGDQ', 'errorOmega_RGDQ', ...
    'errorGamma_RGDQ', 'info_RGDQ', 'X_RCGQ', 'duration_RCGQ', ...
    'errorOmega_RCGQ', 'errorGamma_RCGQ', 'info_RCGQ')
exportgraphics(gcf, figureFile, 'Resolution', 180)
