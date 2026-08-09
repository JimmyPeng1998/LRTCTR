function M = TensorRingQuotient(n, r)
%TENSORRINGQUOTIENT Quotient manifold of a general tensor ring format.
%
% M = TensorRingQuotient(n, r) supports arbitrary tensor order d >= 2,
% nonuniform mode sizes n(1:d), and nonuniform cyclic ranks. The rank
% vector may contain either d entries or d+1 entries, with r(d+1)=r(1).
%
% A point and a tangent vector are represented by a field core, where
% core{k} has size r(k)-by-r(k+1)-by-n(k).
%
% Reference: Quotient geometry of tensor ring decomposition,
%    Bin Gao, Renfeng Peng, and Ya-xiang Yuan,
%    arXiv preprint arXiv:2601.21874, 2026.
%    https://arxiv.org/abs/2601.21874
%
% Original author: Renfeng Peng, Aug. 05, 2026.

[d, n, r] = TR_validate_dimensions(n, r, true);

coreSizes = n.*r(1:d).*r(2:d+1);
gaugeSizes = r(1:d).^2;
ambientDim = sum(coreSizes);
gaugeDim = sum(gaugeSizes);
gaugeOffsets = [0 cumsum(gaugeSizes)];

% Cache the projection data for the most recently used representative.
% MATLAB uses copy-on-write semantics, so retaining the cores does not copy
% their numerical data unless one of the arrays is subsequently modified.
cachedCores = [];
cachedProjectionData = [];

% Manopt manifold interface. We use the Euclidean metric on the total
% space; quotient geometry enters through the horizontal projection below.
M.name = @() sprintf('Quotient manifold of order-%d tensor ring decomposition', d);
M.dim = @() ambientDim-gaugeDim+1;
M.inner = @inner;
M.norm = @(X, eta) sqrt(M.inner(X, eta, eta));
M.dist = @(x, y) error('TensorRingQuotient:NotImplemented', ...
    'TensorRingQuotient.dist is not implemented.');
M.typicaldist = @(x, y) error('TensorRingQuotient:NotImplemented', ...
    'TensorRingQuotient.typicaldist is not implemented.');
M.egrad2rgrad = @(X, eucGrad) eucGrad;
M.ehess2rhess = @(X, eucGrad, eucHess, eta) eucHess;
M.proj = @projection;
M.tangent = M.proj;
M.tangent2ambient = @(X, eta) eta;
M.exp = @exponential;
M.retr = M.exp;
M.hash = @hashPoint;
M.rand = @random;
M.randvec = @randomvec;
M.lincomb = @lincomb;
M.zerovec = @zerovec;
M.transp = @(x1, x2, eta) projection(x2, eta);
M.vec = @vectorize;
M.mat = @matricize;
M.vecmatareisometries = @() true;

    function value = inner(~, xi, eta)
        % Euclidean product metric on the d TR cores.
        validateCoreContainer(xi);
        validateCoreContainer(eta);
        value = 0;
        for k = 1:d
            value = value + xi.core{k}(:)'*eta.core{k}(:);
        end
    end

    function projected = projection(X, eta)
        % Orthogonally remove the vertical (gauge) component of eta. This
        % implements the cyclic block system in the projection proposition
        % of the paper directly, without forming the larger vertical map.
        validateCoreContainer(X);
        validateCoreContainer(eta);

        % U1{k}, U3{k} and the coefficient matrix depend only on X. Reuse
        % them when Manopt projects multiple vectors at the same point.
        data = prepareProjectionData(X);

        % V1{k}, V3{k} are the mode-1 and mode-3 unfoldings of eta.
        V1 = cell(d, 1);
        V3 = cell(d, 1);
        for k = 1:d
            [V1{k}, V3{k}] = unfoldCore(eta.core{k}, k);
        end

        % Only b_k depends on eta; the cached matrix already contains all
        % A_k and B_k blocks from the projection formula.
        systemMatrix = data.systemMatrix;
        rhs = zeros(gaugeDim, 1);
        for k = 1:d
            kPrev = mod(k-2, d)+1;
            rowsK = gaugeOffsets(k)+(1:gaugeSizes(k));
            bk = V1{k}*data.U1{k}' - ...
                data.U3{kPrev}*V3{kPrev}';
            rhs(rowsK) = bk(:);
        end

        % The cached matrix already replaces the redundant equation by
        % trace(D_1)=0; set the matching right-hand side here.
        rhs(end) = 0;

        gauge = systemMatrix\rhs;
        D = cell(d, 1);
        for k = 1:d
            rowsK = gaugeOffsets(k)+(1:gaugeSizes(k));
            D{k} = reshape(gauge(rowsK), [r(k), r(k)]);
        end

        % Subtract the vertical component
        %   U_k x_1 D_k-U_k x_3 D_{k+1}^T
        % by two batched matrix products. The reshape/permute operations
        % apply the same multiplication to all n(k) slices simultaneously.
        projected.core = cell(d, 1);
        for k = 1:d
            kNext = mod(k, d)+1;
            leftPart = reshape( ...
                D{k}*reshape(X.core{k}, [r(k), r(k+1)*n(k)]), ...
                [r(k), r(k+1), n(k)]);
            coreTransposed = reshape(permute(X.core{k}, [2 1 3]), ...
                [r(k+1), r(k)*n(k)]);
            rightPart = permute(reshape(D{kNext}'*coreTransposed, ...
                [r(k+1), r(k), n(k)]), [2 1 3]);
            projected.core{k} = eta.core{k}-leftPart+rightPart;
        end
    end

    function data = prepareProjectionData(X)
        % Return cached unfoldings and coefficient matrix when X is exactly
        % the same representative as in the preceding projection call.
        if sameCachedCores(X.core)
            data = cachedProjectionData;
            return
        end

        data.U1 = cell(d, 1);
        data.U3 = cell(d, 1);
        for k = 1:d
            [data.U1{k}, data.U3{k}] = unfoldCore(X.core{k}, k);
        end

        data.systemMatrix = zeros(gaugeDim, gaugeDim);
        for k = 1:d
            kPrev = mod(k-2, d)+1;
            kNext = mod(k, d)+1;
            rowsK = gaugeOffsets(k)+(1:gaugeSizes(k));
            rowsNext = gaugeOffsets(kNext)+(1:gaugeSizes(kNext));

            Ak = kron(speye(r(k)), ...
                data.U3{kPrev}*data.U3{kPrev}') + ...
                kron(data.U1{k}*data.U1{k}', speye(r(k)));
            Bk = -kron(data.U1{k}, speye(r(k)))* ...
                kron(speye(r(k+1)), data.U3{k}');

            data.systemMatrix(rowsK, rowsK) = ...
                data.systemMatrix(rowsK, rowsK)+Ak;
            data.systemMatrix(rowsK, rowsNext) = ...
                data.systemMatrix(rowsK, rowsNext)+Bk;
            data.systemMatrix(rowsNext, rowsK) = ...
                data.systemMatrix(rowsNext, rowsK)+Bk';
        end

        % Replace the one redundant equation by trace(D_1)=0 once here.
        constraint = zeros(1, gaugeDim);
        constraint(1:r(1)+1:r(1)^2) = 1;
        data.systemMatrix(end, :) = constraint;
        cachedCores = X.core;
        cachedProjectionData = data;
    end

    function isSame = sameCachedCores(cores)
        isSame = iscell(cachedCores) && numel(cachedCores) == d;
        k = 1;
        while isSame && k <= d
            isSame = isequal(cachedCores{k}, cores{k});
            k = k+1;
        end
    end

    function [mode1, mode3] = unfoldCore(core, k)
        % Convert the stored r(k)-by-r(k+1)-by-n(k) core to the unfolding
        % convention of the paper, where the tensor is ordered as
        % r(k)-by-n(k)-by-r(k+1).
        paperCore = permute(core, [1 3 2]);
        mode1 = reshape(paperCore, [r(k), n(k)*r(k+1)]);
        mode3 = reshape(paperCore, [r(k)*n(k), r(k+1)])';
    end

    function Y = exponential(X, eta, t)
        % Addition retraction on the open set of full-rank TR cores.
        if nargin < 3
            t = 1;
        end
        validateCoreContainer(X);
        validateCoreContainer(eta);
        Y.core = cell(d, 1);
        for k = 1:d
            Y.core{k} = X.core{k}+t*eta.core{k};
        end
    end

    function key = hashPoint(X)
        % Lightweight cache key used by Manopt's store database.
        validateCoreContainer(X);
        summaries = zeros(1, d);
        for k = 1:d
            summaries(k) = sum(X.core{k}(:));
        end
        key = ['z' hashmd5(summaries)];
    end

    function X = random()
        % Draw an ambient Gaussian representative.
        X.core = cell(d, 1);
        for k = 1:d
            X.core{k} = randn(r(k), r(k+1), n(k));
        end
    end

    function eta = randomvec(X)
        % Project an ambient Gaussian vector and normalize it horizontally.
        validateCoreContainer(X);
        eta = random();
        eta = projection(X, eta);
        etaNorm = M.norm(X, eta);
        for k = 1:d
            eta.core{k} = eta.core{k}/etaNorm;
        end
    end

    function result = lincomb(~, a1, eta1, a2, eta2)
        validateCoreContainer(eta1);
        result.core = cell(d, 1);
        if nargin == 3
            for k = 1:d
                result.core{k} = a1*eta1.core{k};
            end
        elseif nargin == 5
            validateCoreContainer(eta2);
            for k = 1:d
                result.core{k} = ...
                    a1*eta1.core{k}+a2*eta2.core{k};
            end
        else
            error('TensorRingQuotient:Lincomb', ...
                'lincomb expects either three or five inputs.');
        end
    end

    function eta = zerovec(~)
        % Zero tangent vector in the core-cell representation.
        eta.core = cell(d, 1);
        for k = 1:d
            eta.core{k} = zeros(r(k), r(k+1), n(k));
        end
    end

    function vector = vectorize(~, eta)
        % Stack cores in cyclic order G_1,...,G_d using MATLAB ordering.
        validateCoreContainer(eta);
        vector = zeros(ambientDim, 1);
        offset = 0;
        for k = 1:d
            count = coreSizes(k);
            vector(offset+(1:count)) = eta.core{k}(:);
            offset = offset+count;
        end
    end

    function eta = matricize(~, vector)
        % Inverse operation of vectorize.
        if numel(vector) ~= ambientDim
            error('TensorRingQuotient:VectorSize', ...
                'The vector has an incompatible number of entries.');
        end
        eta.core = cell(d, 1);
        offset = 0;
        for k = 1:d
            count = coreSizes(k);
            eta.core{k} = reshape(vector(offset+(1:count)), ...
                [r(k), r(k+1), n(k)]);
            offset = offset+count;
        end
    end

    function validateCoreContainer(X)
        % Fail early when a point uses a different core order or TR rank.
        if ~isstruct(X) || ~isfield(X, 'core') || numel(X.core) ~= d
            error('TensorRingQuotient:PointFormat', ...
                'Points and tangent vectors must contain d cores.');
        end
        for k = 1:d
            if ~isequal(size(X.core{k}), [r(k), r(k+1), n(k)])
                error('TensorRingQuotient:CoreSize', ...
                    'Core %d has an incompatible size.', k);
            end
        end
    end
end
