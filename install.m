clear
clc

disp('LRTCTR: Low-rank tensor completion in tensor ring decomposition ...')


disp('Adding paths ...')
addpath( cd )
addpath( [cd, filesep, 'examples'] )
addpath( [cd, filesep, 'mex'] )
addpath( [cd, filesep, 'solvers'] )
addpath( [cd, filesep, 'tools'] )

disp('Compiling mex files...')
cd mex
mex ComputeGradsAndPx_mex.c
mex ComputePx_mex.c
mex RGN_matrix_mex.c
cd ..


disp('Finished.')