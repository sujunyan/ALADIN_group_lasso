function [x1, history] = ALADIN_group_lasso(A, b, N, ni, gamma , rho0, varargin )
%  Solve group lasso problem via ALADIN
%
% ALADIN_group_lasso(A, b, N, ni, gamma, MAX_ITER, rho0, mu )
% A: mxn matrix that stores the information of A list
% b: mx1 vector
% n = N*ni
%
% The solution is returned in the vector x.
[m,n] = size(A);
assert(n == N*ni);
%% parse the argument
p = inputParser;
addOptional(p,'adaptive',false);
addOptional(p,'lam_bar',NaN);
addOptional(p,'diff_rho',false); %% determin if different rho is needed
addOptional(p,'x_opt',NaN);
addOptional(p,'tol',1e-8);
addOptional(p,'mu',0);
addOptional(p,'max_iter',500);
addOptional(p,'fixed',true); % if the Hessien matrix is fixed
parse(p,varargin{:});
adaptive = p.Results.adaptive;
lam_bar = p.Results.lam_bar;
diff_rho = p.Results.diff_rho;
x_opt = p.Results.x_opt;
TOL = p.Results.tol;
mu = p.Results.mu;
MAX_ITER = p.Results.max_iter;
fixed = p.Results.fixed;
%% 
for ii = 1:N
    Ai{ii} = A(:,(ii-1)*ni+1:ii*ni);
    Ati{ii} = Ai{ii}';
    xi{ii} = zeros(ni,1);
    yi{ii} = zeros(ni,1);
    gi{ii} = zeros(ni,1);
    if (diff_rho)
        rhoi{ii} = rho0 * sqrt(eigs(Ai{ii} * Ati{ii},1)); 
    else
        rhoi{ii} = rho0;
    end
    Hi{ii} = sparse(rhoi{ii} * eye(ni));
    Hi_inv{ii} = sparse(1/rhoi{ii} * eye(ni));
end
x0 = zeros(m,1);
u  = zeros(m,1);
[~,M_inv] = calculate_M(Ai,Ati,Hi_inv); M_inv = sparse(M_inv);
obj_opt = obj_F(x_opt,gamma,A,b,N,ni);
tstart = tic;
alpha = inf;
phi = inf;
history_t = 0;
distributed_t = 0;
consensus_t = 0;
update_flag = false;
for k = 1:MAX_ITER
    used_t = toc(tstart);
    tstart = tic;
    %fprintf("k %d used %f\n",k,used_t);
    % solve decoupled NLP
    distributed_t_start = tic;
    for ii = 1:N
        yi{ii} = solve_decouple(gamma,mu,Ai{ii},u,xi{ii},Hi{ii},rhoi{ii});
        if(fixed)
             gi{ii} = rho0*(xi{ii} - yi{ii}) - Ati{ii} * u;
        else
             gi{ii} = Hi{ii}*(xi{ii} - yi{ii}) - Ati{ii} * u;
        end
    end
    y0 = (u + x0 + b)/2;

    distributed_t = distributed_t + toc(distributed_t_start);
    consensus_t_start = tic;
    % check merit function
    %phi = merit_function();
%     if (k < 5)
%         alpha = phi;
%         update_flag = false;
%     elseif (phi < alpha && mu>0)
%         alpha = phi;
%         update_Hi();
%         [M, M_inv] = calculate_M(Ai,Ati,Hi_inv);
%         update_flag = false;
%     else
%         update_flag = false;
%     end
    % solve QP
    if (fixed) % if the Hi is fixed during the iterations
        [r, rx]= calculate_r();
        u_diff = M_inv * (r - rx);
        u = u + u_diff;
        for ii = 1:N
            xi{ii} = 2*yi{ii} - xi{ii} - 1/rho0 * Ati{ii} * u_diff;
        end
        x0 = 2*y0 - x0 + u_diff;
    elseif ~update_flag % if Hi are not updated during this iteration
        [r, rx]= calculate_r();
        u_diff = M_inv * (r - rx);
        u = u + u_diff;
        for ii = 1:N
            xi{ii} = 2*yi{ii} - xi{ii} - Hi_inv{ii} * Ati{ii} * u_diff;
        end
        x0 = 2*y0 - x0 + u_diff;
    else
        new_u = -(2 * y0 - x0 -u);
        for ii = 1:N
            new_u = new_u + Ai{ii} * (yi{ii} - Hi_inv{ii}*gi{ii});
        end
        u = M_inv * new_u;
        for ii = 1:N
            xi{ii} = yi{ii} - Hi_inv{ii} * (gi{ii} + Ati{ii} * u);
        end
    end

    consensus_t = consensus_t + toc(consensus_t_start);
    if(adaptive)
        update_mu();
    end
    %% store the history
    history_t_start = tic;
    %history.x{k} = vec([xi{:}]); % TODO, do not save trajectories to save memory
    %history.y{k} = vec([yi{:}]);
    history.update{k} = update_flag;
    history.alpha{k} = alpha;
    history.phi{k} = phi;
    %err = norm(x_opt - vec([yi{:}]),inf)/norm(x_opt,inf);
    err = norm(x_opt - vec([yi{:}]),inf);
    %obj_v = obj_F(vec([yi{:}]),gamma,A,b,N,ni);
    %err_func = norm(obj_v - obj_opt,inf);history.err_func{k} = err_func;
    history.err{k} = err;
    history_t = history_t + toc(history_t_start);
    history.nIter = k;
    if(err < TOL)
        %fprintf("ALADIN iter %d times\n",k);
        break;
    end
end
x1 = vec([xi{:}]);
history.time = distributed_t + consensus_t;
history.dTime = distributed_t;
history.cTime = consensus_t;
% fprintf("ALADIN used %d iterations \n",history.nIter);
% fprintf("ALADIN total time used %f s\n",distributed_t + consensus_t);
% fprintf("ALADIN distributed time used %f s\n",distributed_t);
% fprintf("ALADIN consensus time used %f s\n",consensus_t);
% fprintf("ALADIN history time used %f s\n",history_t);

% inside helper functions  -----------------------------
    function [M, M_inv] = calculate_M(Ai,Ati,Hi_inv)
         M = eye(m);
         for jj = 1:N
            % AH^{-1}A'
            M = M + Ai{jj} * (Hi_inv{jj}) * Ati{jj};
         end
         M_inv = inv(M);
    end
    function [r, rx]= calculate_r()
        r = -2*y0;
        rx = -x0;
        for jj = 1:N
            r = r + 2 * Ai{jj} * yi{jj};
            rx = rx + Ai{jj} * xi{jj};
        end
    end
    function phi = merit_function()
        phi = 1/2*norm(y0-b)^2;
        sum_of_Ay = - y0;
        for jj = 1:N
            yy = yi{jj};
            phi = phi + gamma*(norm([yy ; mu]) - mu);
            sum_of_Ay = sum_of_Ay + Ai{jj} * yi{jj};
        end
        phi = phi + lam_bar * norm(sum_of_Ay,1);
    end
    function update_Hi()
        assert(mu>0);
        for jj = 1:N
            yy = yi{jj};
            sqrt_yu = norm([yy;mu]);
            mul = gamma / sqrt_yu^3;
            Hi{jj} = sqrt_yu * eye(ni) - yy*yy';
            Hi_inv{jj} = inv(Hi{jj});
        end
    end
    function update_mu()
        if(mu < TOL)
             mu = 0;
        end
        mu = mu * 0.6;
    end
end

%%% helper functions outside ----------------------------
function y = solve_decouple(gamma,mu,A,u,x,H,rho)
    tstart = tic;
    ni = size(x,1);
    cvx_flag = false;
    
    %% solve by cvx
    if(cvx_flag)
        cvx_begin quiet
            variable y(ni) 
            minimize (gamma * (norm([y;mu]) -mu) + u' * A * y + 1/2 * quad_form(y-x,H)) ;
        cvx_end
        cvx_used_t = toc(tstart);
        tstart = tic;
    end
    if (mu == 0)
        y = shrinkage(x - 1/rho*A'*u, gamma/rho);
        return;
    end
    %% solved by fminunc 
    options = optimoptions('fminunc','SpecifyObjectiveGradient',false,'Display','off');
    y0 = ones(ni,1);
    y_tmp = fminunc(@(yy) obj_func(yy,gamma,mu,A,u,x,H),...
                    y0,options);
    fminunc_used_t = toc(tstart);
    if(cvx_flag)
        err = (y_tmp - y);
        fprintf("In sovlve_decouple err %e cvx uses %f fminfunc uses %f\n"...
            ,norm(err),cvx_used_t,fminunc_used_t);
    else
        y = y_tmp;
    end
end

function [f] = obj_func(y,gamma,mu,A,u,x,H)
        f = gamma * (norm([y;mu]) - mu);
        f = f + u'*A* y + (y-x)'*H*(y-x)/2;
end

function f = obj_F(x,gamma,A,b,N,ni)
    % the total objective function F = || Ax - b ||_2^2 + gamma * || x ||_{2,1}
    sumNorm = zeros(ni,1);
    for ii = 1:N
        xi = x( (ii-1)*ni+1:ii*ni , 1);
        sumNorm = sumNorm + norm(xi,2);
    end
    f = 1/2 * norm(A*x-b,2)^2 + gamma * sumNorm;
end

