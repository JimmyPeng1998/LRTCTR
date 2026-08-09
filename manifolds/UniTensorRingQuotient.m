function M = UniTensorRingQuotient(d,n,r)
%UNITENSORRINGQUOTIENT Quotient manifold of an order-d uniform TR format.
%
% Input:
%   d: tensor order (integer greater than or equal to 3)
%   n: common tensor mode size (positive integer scalar)
%   r: common TR rank (positive integer scalar)
%
% Output:
%   M: Manopt factory for the uniform TR quotient manifold (struct)
%
% A point is represented by one shared core X.U of size r-by-n-by-r. The
% core is repeated at all d positions of the tensor ring. The quotient
% action is simultaneous conjugation of this shared core and is therefore
% independent of d; d only determines how many times the core occurs in the
% represented tensor.
%
% Reference: Quotient geometry of tensor ring decomposition,
%    Bin Gao, Renfeng Peng, and Ya-xiang Yuan,
%    arXiv preprint arXiv:2601.21874, 2026.
%    https://arxiv.org/abs/2601.21874
%
% Original author: Renfeng Peng, Nov. 27, 2025.
% Last modified: Renfeng Peng, Aug. 05, 2026.

if ~isnumeric(d) || ~isreal(d) || ~isscalar(d) || ...
        ~isfinite(d) || d < 3 || d ~= floor(d)
    error('UniTensorRingQuotient:Order', ...
        'd must be an integer greater than or equal to three.');
end
if ~isnumeric(n) || ~isreal(n) || ~isscalar(n) || ...
        ~isfinite(n) || n < 1 || n ~= floor(n)
    error('UniTensorRingQuotient:ModeSize', ...
        'n must be a positive integer scalar.');
end
if ~isnumeric(r) || ~isreal(r) || ~isscalar(r) || ...
        ~isfinite(r) || r < 1 || r ~= floor(r)
    error('UniTensorRingQuotient:Rank', ...
        'r must be a positive integer scalar.');
end
if n < r^2
    error('UniTensorRingQuotient:InjectivityDimensions', ...
        'The quotient geometry requires n >= r^2.');
end


    M.name=@() sprintf( ...
        'Quotient manifold of order-%d uniform tensor ring decomposition.', d);

    M.dim=@() n*r^2-r^2+1;

    M.inner=@(X, xi, eta) xi.U(:)'*eta.U(:);

    M.norm = @(X, eta) sqrt(M.inner(X, eta, eta));

    M.dist = @(x, y) error('UniTensorRingQuotient.dist not implemented yet.');

    M.typicaldist = @(x, y) error('UniTensorRingQuotient.typicaldist not implemented yet.');

    % Compute the Riemannian gradient from the Euclidean gradient
    M.egrad2rgrad=@(X,egrad) egrad;

    M.ehess2rhess = @(X, egrad, ehess, eta) ehess;
    function Hess = ehess2rhess(X, egrad, ehess, eta)
        error('UniTensorRingQuotient.ehess2rhess not implemented yet.');
    end






    M.proj = @projection;
    function Proj = projection(X, eta)
        % Projection onto the horizontal space. The vertical vector is the
        % commutator U(i)D-DU(i), regardless of how many times U occurs in
        % the order-d tensor, so the projection system is independent of d.
        U1 = reshape(X.U, [r, n*r]);
        U2 = reshape(permute(X.U, [2 1 3]), [n, r*r]);
        U3 = reshape(X.U, [r*n, r])';
        
        V1 = reshape(eta.U, [r, n*r]);
        V3 = reshape(eta.U, [r*n, r])';
        
        A = kron(U1*U1', speye(r)) + kron(speye(r), U3*U3') ...
            - kron(speye(r), U3) * kron(U1', speye(r)) ...
            - kron(U1, speye(r)) * kron(speye(r), U3');
        b = V1*U1' - U3*V3';
        b = b(:);
        
        A(end,:) = reshape(eye(r),[1 r^2]);
        b(end) = 0;
        
        v = A\b;
        D = reshape(v,[r r]);
        
        temp = U2*(kron(speye(r),D) - kron(D',speye(r)))';
        Proj.U = eta.U - permute(reshape(temp, [n,r,r]), [2 1 3]);
    end


    M.tangent = M.proj;
    M.tangent2ambient = @(X, eta) eta;


    
    M.exp = @exponential;
    function Y = exponential(X, eta, t)
        if nargin < 3
            t = 1.0;
        end
        
        Y.U = X.U + t*eta.U;
    end
    
    M.retr = M.exp;

    M.hash = @(X) ['z' hashmd5(sum(X.U(:)))]; % Efficient, suggested by Bart Vandereycken.

    M.rand = @random;
    function X = random()
        X.U = randn(r,n,r);
    end

    M.randvec = @randomvec;
    function eta = randomvec(X)
        % A random vector on the tangent space
        eta.U = randn(r,n,r);
        eta = projection(X, eta);
        nrm = M.norm(X, eta);
        eta.U = eta.U / nrm;
    end

    M.lincomb = @lincomb;
    function d = lincomb(X, a1, d1, a2, d2) %#ok<INUSL>
        
        if nargin == 3
            d.U = a1*d1.U;
        elseif nargin == 5
            d.U = a1*d1.U + a2*d2.U;
        else
            error('Bad use of UniTensorRingQuotient.lincomb.');
        end
        
    end





    M.zerovec = @(X) struct('U', zeros(r,n,r));

    M.transp = @(x1, x2, d) projection(x2, d);

    M.vec = @(X, uMat) uMat.U(:);
    M.mat = @(X, uVec) struct('U', reshape(uVec, [r,n,r]));
    M.vecmatareisometries = @() true;



end
