% NOTE: The phase-plot code is intentionally commented out.
% Uncomment the code below after the phase-transition data have been generated.
%
% clear
% clc
% close all
% scriptDir = fileparts(mfilename('fullpath'));
% resultsDir = fullfile(scriptDir, 'Results');
% data = load(fullfile(resultsDir, 'Result_Exp3_phase_TR.mat'));
% figure
% h = heatmap(data.ndims, data.samples(end:-1:1), ...
%     data.successes(:,end:-1:1)'/data.repeats, ...
%     'CellLabelColor', 'none', 'GridVisible', 'off');
% colormap(gray)
% axisData = gca;
% labels = strings(size(axisData.YDisplayData));
% positions = 10:10:numel(labels);
% labels(positions) = string(data.samples(positions)/1e4);
% axisData.YDisplayLabels = labels(end:-1:1);
% labels = strings(size(axisData.XDisplayData));
% positions = 1:5:numel(labels);
% labels(positions) = string(data.ndims(positions));
% axisData.XDisplayLabels = labels;
% heatmapStruct = struct(h);
% heatmapStruct.Axes.XAxis.TickLabelRotation = 0;
% annotation('textbox', [.12 .7 .6 .3], 'String', '\times10^4', ...
%     'EdgeColor', 'none', 'FontSize', 18)
% set(gca, 'FontSize', 20)
% xlabel('Tensor size'); ylabel('Sample size'); title('Phase plot TR-RCG(Q)')
% saveas(gcf, fullfile(resultsDir, 'Exp3_phase_TR.eps'), 'epsc')
%
% data = load(fullfile(resultsDir, 'Result_Exp3_phase_uTR.mat'));
% figure
% h = heatmap(data.ndims, data.samples(end:-1:1), ...
%     data.successes(:,end:-1:1)'/data.repeats, ...
%     'CellLabelColor', 'none', 'GridVisible', 'off');
% colormap(gray)
% axisData = gca;
% labels = strings(size(axisData.YDisplayData));
% positions = 5:5:numel(labels);
% labels(positions) = string(data.samples(positions)/1e3);
% axisData.YDisplayLabels = labels(end:-1:1);
% labels = strings(size(axisData.XDisplayData));
% positions = 10:10:numel(labels);
% labels(positions) = string(data.ndims(positions));
% axisData.XDisplayLabels = labels;
% heatmapStruct = struct(h);
% heatmapStruct.Axes.XAxis.TickLabelRotation = 0;
% annotation('textbox', [.12 .7 .6 .3], 'String', '\times10^3', ...
%     'EdgeColor', 'none', 'FontSize', 18)
% annotation('textbox', [.092 .1 .1 .1], 'String', '0', ...
%     'EdgeColor', 'none', 'FontSize', 20)
% set(gca, 'FontSize', 20)
% xlabel('Tensor size'); ylabel('Sample size'); title('Phase plot uTR-RCG(Q)')
% saveas(gcf, fullfile(resultsDir, 'Exp3_phase_uTR.eps'), 'epsc')
