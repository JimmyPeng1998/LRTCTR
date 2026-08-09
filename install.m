disp('LRTCTR: Low-rank tensor completion in tensor ring decomposition ...')

rootdir = fileparts(mfilename('fullpath'));

disp('Adding LRTCTR paths ...')
addpath(rootdir)
addpath(fullfile(rootdir, 'examples'))
addpath(fullfile(rootdir, 'manifolds'))
addpath(fullfile(rootdir, 'mex'))
addpath(fullfile(rootdir, 'solvers'))
addpath(fullfile(rootdir, 'tools'))
addpath(fullfile(rootdir, 'tools', 'quotient'))

if exist('manopt_version', 'file') ~= 2
    manopt_root = fullfile(rootdir, 'manopt');
    manopt_code = fullfile(manopt_root, 'manopt');
    if exist(manopt_code, 'dir') == 7
        disp('Loading Manopt ...')
        addpath(manopt_root)
        addpath(genpath(manopt_code))
    else
        warning('LRTCTR:ManoptNotFound', ...
            ['Manopt was not found. The preconditioned solvers remain available, ', ...
             'but quotient-geometry solvers require Manopt. Add Manopt to the ', ...
             'MATLAB path before using those solvers.'])
    end
end

mex_dir = fullfile(rootdir, 'mex');
mex_names = {'ComputeGradsAndPx_mex', 'ComputeGradsAndPxGeneral_mex', ...
    'ComputePx_mex', 'ComputePxGeneral_mex', 'RGN_matrix_mex'};
platform_extension = mexext;
fprintf('Checking MEX files for %s (%s) ...\n', computer, platform_extension)

available_mex = false(size(mex_names));
for k = 1:numel(mex_names)
    binary_file = fullfile(mex_dir, ...
        [mex_names{k} '.' platform_extension]);
    if exist(binary_file, 'file') ~= 0
        fprintf('  Using precompiled %s.%s\n', ...
            mex_names{k}, platform_extension)
        available_mex(k) = true;
        continue
    end

    source_file = fullfile(mex_dir, [mex_names{k} '.c']);
    fprintf('  Compiling %s.c ...\n', mex_names{k})
    try
        mex('-outdir', mex_dir, source_file)
        available_mex(k) = exist(binary_file, 'file') ~= 0;
        if available_mex(k)
            fprintf('  Compiled %s.%s\n', ...
                mex_names{k}, platform_extension)
        else
            warning('LRTCTR:MexOutputMissing', ...
                'Compilation did not create %s.', binary_file)
        end
    catch exception
        warning('LRTCTR:MexCompileFailed', ...
            'Could not compile %s.c: %s', mex_names{k}, exception.message)
    end
end

if all(available_mex)
    disp('All MEX routines are available.')
else
    missing = strjoin(mex_names(~available_mex), ', ');
    warning('LRTCTR:MexUnavailable', ...
        ['Unavailable MEX routines: %s. Solvers that use these routines ', ...
         'cannot run. Configure a supported C compiler with mex -setup, ', ...
         'then run install.m again.'], missing)
end

disp('Finished.')
