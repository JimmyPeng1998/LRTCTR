function Exp1_plot_comparison(data, prefix, colors, markerIndices, ...
        lwidth, msize, fontSize, figureFile)
%EXP1_PLOT_COMPARISON Plot iteration- and time-based convergence curves.

% Place the iteration and runtime comparisons in one figure.
figure('Position', [100 100 1050 470])
layout = tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% Test error versus iteration number.
nexttile
h1 = plotCurve(iterations(data.errorGamma_RGDE), data.errorGamma_RGDE, ...
    '-x', colors(4,:), markerIndices, msize, lwidth);
hold on
h2 = plotCurve(iterations(data.errorGamma_RGDQ), data.errorGamma_RGDQ, ...
    '-o', colors(3,:), markerIndices, msize+4, lwidth);
h3 = plotCurve(iterations(data.errorGamma_RCGE), data.errorGamma_RCGE, ...
    '->', colors(2,:), markerIndices, msize, lwidth);
h4 = plotCurve(iterations(data.errorGamma_RCGQ), data.errorGamma_RCGQ, ...
    '-+', colors(1,:), markerIndices, msize, lwidth);
plotCurve(iterations(data.errorGamma_RGDE), data.errorGamma_RGDE, ...
    '-x', colors(4,:), markerIndices, msize, lwidth, 'off');
xlabel('Iteration number')
ylabel('Test error')
set(gca, 'FontSize', fontSize)
grid on
hLegend = legend([h1 h2 h3 h4], ...
    [prefix '-RGD(E)'], [prefix '-RGD(Q)'], ...
    [prefix '-RCG(E)'], [prefix '-RCG(Q)'], ...
    'Orientation', 'horizontal');
hLegend.Layout.Tile = 'north';
hLegend.ItemTokenSize = [30, 0.5];

% Test error versus computational time.
nexttile
plotCurve(data.duration_RGDE, data.errorGamma_RGDE, ...
    '-x', colors(4,:), markerIndices, msize, lwidth);
hold on
plotCurve(data.duration_RGDQ, data.errorGamma_RGDQ, ...
    '-o', colors(3,:), markerIndices, msize+4, lwidth);
plotCurve(data.duration_RCGE, data.errorGamma_RCGE, ...
    '->', colors(2,:), markerIndices, msize, lwidth);
plotCurve(data.duration_RCGQ, data.errorGamma_RCGQ, ...
    '-+', colors(1,:), markerIndices, msize, lwidth);
plotCurve(data.duration_RGDE, data.errorGamma_RGDE, ...
    '-x', colors(4,:), markerIndices, msize, lwidth, 'off');
xlabel('Time (s)')
ylabel('Test error')
set(gca, 'FontSize', fontSize)
grid on
set(gcf, 'Renderer', 'painters')
print(gcf, figureFile, '-depsc', '-painters')
end

function values = iterations(errors)
values = 0:numel(errors)-1;
end

function h = plotCurve(xValues, errors, style, color, markerIndices, ...
        markerSize, lineWidth, handleVisibility)
if nargin < 8; handleVisibility = 'on'; end
indices = markerIndices(markerIndices <= numel(errors));
h = semilogy(xValues, errors, style, 'MarkerIndices', indices, ...
    'MarkerSize', markerSize, 'LineWidth', lineWidth, 'Color', color, ...
    'HandleVisibility', handleVisibility);
end
