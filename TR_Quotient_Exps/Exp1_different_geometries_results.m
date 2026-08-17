clear
clc
close all

%% Load the numerical results
scriptDir = fileparts(mfilename('fullpath'));
resultsDir = fullfile(scriptDir, 'Results');
data = load(fullfile(resultsDir, 'Result_Exp1_geometries.mat'));
plotRepeat = 1;
colors = get(groot, 'defaultAxesColorOrder');
lwidth = 2.5;
msize = 10;
fontSize = 13;

%% General TR: mark every iteration
Exp1_plot_comparison(data.resultsTR{plotRepeat}, 'TR', colors, ...
    1:5000, lwidth, msize, fontSize, ...
    fullfile(resultsDir, 'Exp1_different_geometries_TR.eps'))

%% Uniform TR: mark every iteration
Exp1_plot_comparison(data.resultsUTR{plotRepeat}, 'uTR', colors, ...
    1:5000, lwidth, msize, fontSize, ...
    fullfile(resultsDir, 'Exp1_different_geometries_uTR.eps'))
