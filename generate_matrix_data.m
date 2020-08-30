%% generate data
L = 10;
M = 100;  % size of each block
%% convert to vector form 
ni = M;         % size of each block
n  = ni * N;    % dimension of primal variables
m  = L*M;       % amount of data TODO
p = 10/N;       % sparsity density TODO : previously 10/N
% % generate block sparse solution vector
X = zeros(N,M);
for ii = 1:N
    if( rand() < p)
        X(ii,:) = randn(1,M);        % fill nonzeros
    end
end
Q = randn(L,N);                      % generate random data matrix
NN = normrnd(0,1e-2,L,M);
%Q = Q*spdiags(1./norms(Q)',0,N,N);  % normalize columns of Q
%NN = sqrt(1e-3) * randn(L,M);
Y = Q * X + NN;

%% convert matrix form to vector form
A   = jacobian_of_QTheta(Q,M);
x   = vec(X');
b   = vec(Y);
err = A*x - vec(Q*X);
norm(err);
% lambda max
nrmAitb = zeros(N,1);
for ii = 1:N
    Ai = A(:,(ii-1)*ni+1:ii*ni);
    nrmAitb(ii) = norm(Ai'*b);
end
lambda_max = max( nrmAitb );
lambda     = 0.5*lambda_max;      % regularization parameter
gamma      = 5e-1 * lambda_max;   % regularization parameter
gamma0     = gamma;
line_width = 1.5;

% % convert the problem to sparse matrix
A = sparse(A); 
b = sparse(b);
%% function jacobian_of_QTheta
function A = jacobian_of_QTheta(Q,M)
    % let x = vec(Theta), calculate the Jacobian J = d(Q*Theta)/dx
    [L,N] = size(Q);
    A = zeros(L*M,M*N);
    for ii = 1:N
        for jj = 1:M
            q = Q(:,ii);
            row = (jj-1)*L+1 : jj*L;
            col = jj + M * (ii-1);
            A(row,col) = q;
        end
    end 
end 