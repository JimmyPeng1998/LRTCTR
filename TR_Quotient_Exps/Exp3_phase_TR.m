clear
clc
scriptDir = fileparts(mfilename('fullpath'));
packageDir = fileparts(scriptDir);
oldDir = pwd;
cd(packageDir)
run('install.m')
cd(oldDir)
resultsDir = fullfile(scriptDir, 'Results');
rng(20260803, 'twister')
d = 3;
r = 2*ones(1,d+1);
ndims = 50:10:200;
samples = 2000:2000:100000;
repeats = 10;
maxiter = 1000;
opts = struct('maxiter',maxiter,'maxTime',inf,'gradtol',1e-13, ...
    'train_tol',1e-4,'tol',0,'verbosity',0,'minstepsize',eps);
successes = zeros(numel(ndims),numel(samples));
for j = 1:numel(ndims)
    n = ndims(j)*ones(1,d);
    for q = 1:numel(samples)
        if samples(q) > prod(n); continue; end
        for t = 1:repeats
            Xtrue = TR_randn(n,d,r);
            Omega = makeOmegaSet_mod(n,samples(q));
            PA = TR_sample(Xtrue,Omega);
            X0 = TR_randn(n,d,r);
            [~,~,errorOmega] = TR_RCGQ(X0,PA,Omega,opts);
            successes(j,q) = successes(j,q)+(errorOmega(end)<1e-4);
        end
        save(fullfile(resultsDir,'Result_Exp3_phase_TR.mat'), ...
            'successes','ndims','samples','repeats','r','opts')
    end
end
