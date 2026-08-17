clear
clc
close all


%% Load results
scriptDir = fileparts(mfilename('fullpath'));
resultsDir = fullfile(scriptDir, 'Results');
load(fullfile(resultsDir, 'Result_Exp4_movielens.mat'))


%% Plot settings
colors = get(groot, 'defaultAxesColorOrder');
color_RCGQ = colors(1,:);
color_RCGprecon = colors(2,:);
color_RGDQ = colors(3,:);
color_RGDprecon = colors(4,:);
lwidth = 2.5;
msize = 8;
fontSize = 14;

%% Plot results
figure('Position', [100 100 1450 720])
layout = tiledlayout(3, 2, 'TileSpacing', 'compact', ...
    'Padding', 'compact');

nexttile(1, [3 1])
h1 = plot(duration_RGDQ, errorGamma_RGDQ, '-o', ...
    'MarkerSize', msize+2, ...
    'LineWidth', lwidth, 'Color', color_RGDQ);
hold on
h2 = plot(duration_RCGQ, errorGamma_RCGQ, '-+', ...
    'MarkerSize', msize, ...
    'LineWidth', lwidth, 'Color', color_RCGQ);
h3 = plot(duration_RGDprecon, errorGamma_RGDprecon, '-x', ...
    'MarkerSize', msize, ...
    'LineWidth', lwidth, 'Color', color_RGDprecon);
h4 = plot(duration_RCGprecon, errorGamma_RCGprecon, '->', ...
    'MarkerSize', msize, ...
    'LineWidth', lwidth, 'Color', color_RCGprecon);
h5 = plot(duration_RGDQ, errorGamma_RGDQ, '-o', ...
    'MarkerSize', msize+2, ...
    'LineWidth', lwidth, 'Color', color_RGDQ);
xlim([0,3000])
ylim([0.2,1.2])
xlabel('Time (s)')
ylabel('Relative test error')
title('MovieLens 1M,\quad $\mathbf{r}=(6,10,3)$', 'Interpreter', 'latex')
set(gca, 'FontSize', fontSize)
grid on

for k = 1:d
    nexttile(2*k)
    plot(1:numel(singularValues_RGDQ{k}), ...
        singularValues_RGDQ{k}, '-o', ...
        'MarkerSize', 4, 'LineWidth', 1.5, 'Color', color_RGDQ)
    hold on
    plot(1:numel(singularValues_RCGQ{k}), ...
        singularValues_RCGQ{k}, '-+', ...
        'MarkerSize', 5, 'LineWidth', 1.5, 'Color', color_RCGQ)
    plot(1:numel(singularValues_RGDprecon{k}), ...
        singularValues_RGDprecon{k}, '-x', ...
        'MarkerSize', 4, 'LineWidth', 1.5, 'Color', color_RGDprecon)
    plot(1:numel(singularValues_RCGprecon{k}), ...
        singularValues_RCGprecon{k}, '->', ...
        'MarkerSize', 4, 'LineWidth', 1.5, 'Color', color_RCGprecon)
    xlabel('Index')
    ylabel('Singular value')
    title(sprintf('Singular values of mode-2 unfolding $\\mathbf{W}_{%d}$', k), ...
        'Interpreter', 'latex')
    set(gca, 'FontSize', fontSize)
    grid on
end

hLegend = legend([h1 h2 h3 h4], ...
    'TR-RGD(Q)', 'TR-RCG(Q)', ...
    'TR-RGD(precon)', 'TR-RCG(precon)', ...
    'Orientation', 'horizontal');
hLegend.Layout.Tile = 'north';
hLegend.ItemTokenSize = [30, 0.5];

set(gcf, 'Renderer', 'painters')
print(gcf, fullfile(resultsDir, 'Exp4_movielens.eps'), ...
    '-depsc', '-painters')
