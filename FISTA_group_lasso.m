function [x, history] = FISTA_group_lasso(A, b, N ,gamma, ni,  varargin)
%  Reference: A. Becky and M. Teboulle, fast iterative shrinkage-thresholding algorithm 
%  for linear inverse problems,? SIAM journal on imaging sciences, 2009
%% parse the argument
p = inputParser;
addOptional(p,'x_opt',NaN);
addOptional(p,'tol',1e-8);
addOptional(p,'max_iter',500);
addOptional(p,'rho',1);
addOptional(p,'pg',false);
parse(p,varargin{:});
x_opt = p.Results.x_opt;
TOL   = p.Results.tol;
nIter = p.Results.max_iter;
rho   = p.Results.rho;
pg_flag = p.Results.pg; % proximal gradient flag
%% prepare
[m,n] = size(A);
assert(n == N*ni);
L =  rho * eigs(A'*A,1); % the lipschitz constant for group Lasso
x = zeros(n,1);
t = 1;
y = x;
tstart = tic;
for iter = 1:nIter
    xLast = x;
    x = prox(y - A'*(A*y - b)/L ,ni,N,L,gamma);
    xDiff = x - xLast;
    tLast = t;
    if (~pg_flag)
        t = (1 + sqrt(1+4*t^2))/2;
    end
    y = x + (tLast-1)/t * xDiff;
    % store the history information
    history.nIter = iter;
    err = norm(x_opt - x,inf);
    history.err{iter} = err;
    if(err < TOL)
        %fprintf("FISTA iter %d times (pg,%d)\n",iter,pg_flag);
        break;
    end
end
used_t = toc(tstart);
history.time = used_t;
%fprintf("FISTA total time used %d s\n",used_t);
end
 
function x = prox(y,ni,N,L ,gamma)
% the proximal operator for l2,1 norm
% x = argmin { \sum_{i=1}^N || x_i ||_2 + L/2 * || x - y + 1/L * (Ay - b )||^2  }
% L is the Liptchiz constant
x = zeros(size(y));
for ii = 1:N
    yi = y((ii-1)*ni+1:ii*ni);
    x((ii-1)*ni+1:ii*ni) = shrinkage(yi,gamma/L);
end

end

function z = l21_norm(x,ni,N)
z = 0;
for ii = 1:N
    xi = x((ii-1)*ni+1:ii*ni);
    z  = z + norm(xi,2);
end
end