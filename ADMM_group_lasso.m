function [z, history] = ADMM_group_lasso(A, b, lambda, ni, rho, alpha, max_iter,optimal_x, TOL)
% group_lasso  Solve group lasso problem via ADMM
%
% [x, history] = group_lasso(A, b, p, lambda, rho, alpha);
%
% solves the following problem via ADMM:
%
%   minimize 1/2*|| Ax - b ||_2^2 + \lambda sum(norm(x_i))
%
% The input p is a K-element vector giving the block sizes n_i, so that x_i
% is in R^{n_i}.
%
% The solution is returned in the vector x.
%
% history is a structure that contains the objective value, the primal and
% dual residual norms, and the tolerances for the primal and dual residual
% norms at each iteration.
%
% rho is the augmented Lagrangian parameter.
%
% alpha is the over-relaxation parameter (typical values for alpha are
% between 1.0 and 1.8).
%
%
% More information can be found in the paper linked at:
% http://www.stanford.edu/~boyd/papers/distr_opt_stat_learning_admm.html
%

t_start = tic;

QUIET    = 1;
MAX_ITER = max_iter;
ABSTOL   = 1e-4;
RELTOL   = 1e-4;
%% data processing
[m, n] = size(A);

% check that ni divides in to n
if (rem(n,ni) ~= 0)
    error('invalid block size');
end
% number of subsystems
N = n/ni;

%% ADMM Solver
x = zeros(ni,N);
z = zeros(m,1);
u = zeros(m,1);
Axbar = zeros(m,1);

zs = zeros(m,N);
Aixi = zeros(m,N);

if ~QUIET
    fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
      'r norm', 'eps pri', 's norm', 'eps dual', 'objective');
end
% pre-factor
for i = 1:N,
    Ai{i} = A(:,(i-1)*ni + 1:i*ni);
%     Ai{i} = Ai
    [Vi,Di] = eig(full(Ai{i}'*Ai{i}));
    V{i} = Vi;
    D{i} = diag(Di);

    % in Matlab, transposing costs space and flops
    % so we save a transpose operation everytime
    At{i} = Ai{i}';
end

t_start = tic;
history_t = 0;
distributed_t = 0;
consensus_t = 0;
history.x = [];
for k = 1:MAX_ITER
    % x-update (to be done in parallel)
    distributed_t_start = tic;
    for i = 1:N
        x(:,i) = x_update(Ai{i}, Aixi(:,i) + z - Axbar - u, lambda/rho, V{i}, D{i});
        Aixi(:,i) = Ai{i}* x(:,i);
    end
    distributed_t = distributed_t + toc(distributed_t_start);
    consensus_t_start = tic;
    % z-update
    zold = z;
    Axbar = 1/N*A*vec(x);

    Axbar_hat = alpha*Axbar + (1-alpha)*zold;
    z = (b + rho*(Axbar_hat + u))/(N+rho);

    % u-update
    u = u + Axbar_hat - z;
    consensus_t = consensus_t + toc(consensus_t_start);
    history_t_start = tic;
    %history.x = [history.x vec(x)];
    err = norm(optimal_x - vec(x),inf)/ norm(optimal_x,inf);
    history.err{k} = err;
    history.nIter = k;
    history_t = history_t + toc(history_t_start);
    
    if(max(vec(x) - optimal_x) < TOL)
         %fprintf("ADMM iter %d times\n",k);
         break;
    end
end
history.time = distributed_t + consensus_t;
history.dTime = distributed_t;
history.cTime = consensus_t;
% if ~QUIET
%     toc(t_start);
% end
%toc(t_start);
% fprintf("ADMM used %d iterations \n",history.nIter);
% fprintf("ADMM total time used %f s\n",distributed_t + consensus_t);
% fprintf("ADMM distributed time used %f s\n",distributed_t);
% fprintf("ADMM consensus time used %f s\n",consensus_t);
% fprintf("ADMM history time used %f s\n",history_t);
end


function p = objective(A, b, lambda, N, x, z)
    p = ( 1/2*sum_square(N*z - b) + lambda*sum(norms(x)) );
end

function x = x_update(A, b, kappa, V, D)
[m,n] = size(A);

q = A'*b;

if (norm(q) <= kappa)
   x = zeros(n,1);
else
    % bisection on t
    lower = 0; upper = 1e10;
    for i = 1:100,
        t = (upper + lower)/2;

        x = V*((V'*q)./(D + t));
        if t > kappa/norm(x),
            upper = t;
        else
            lower = t;
        end
        if (upper - lower <= 1e-6)
            break;
        end
    end
end

end