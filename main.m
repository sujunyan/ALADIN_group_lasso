%% compare the performance of different algorithms for group Lasso
% % initialization
clear variables;  close all; clc;
%rng(0);
% % problem setup
N = 200;
generate_matrix_data

% % algorithm setting
MAX_ITER = 1000;
rhoADMM  = 1e-1 * gamma; 
alpha    = 1.5;
TOL      = 1e-6; 

% % solve via CVX
fprintf("CVX start\n");
[x0,~] = solve_cvx(A, b, gamma, N , ni);

% % solve via ADMM
fprintf("ADMM start\n");
[xADMM, historyADMM] = ADMM_group_lasso(A, b, gamma, ni, rhoADMM, alpha ,MAX_ITER , x0, TOL);

% % solve via FISTA
fprintf("FISTA start\n");
[xFISTA, historyFISTA] = FISTA_group_lasso(A, b, N ,gamma, ni,'max_iter',MAX_ITER,...
                                           'x_opt', x0,'tol',TOL,'rho',8e-1);

% % solve via Proximal Gradient method
fprintf("Proximal Gradient start\n");
[xPG, historyPG] = FISTA_group_lasso(A, b, N ,gamma, ni,'max_iter',MAX_ITER,...
                                     'x_opt', x0,'tol',TOL,'rho',5e-1,'pg',true);
                                 
% % solve via ALADIN
rhoALADIN = 4e-1 * gamma;
fprintf("ALADIN start\n");
[xALADIN, historyALADIN] = ALADIN_group_lasso(A, b, N, ni, gamma, rhoALADIN, ...
                           'max_iter',MAX_ITER,'tol', TOL, 'x_opt', x0 );

% % residual record
errADMM  = historyADMM.err;
errFISTA = historyFISTA.err;
errPG    = historyPG.err;
errALADIN = historyALADIN.err;

% % plot
line_width = 2;
plot_result(errADMM,line_width);
hold on;
plot_result(errFISTA,line_width);
plot_result(errPG,line_width);
plot_result(errALADIN,line_width);
hold off;
legend('ADMM','FISTA','Proximal Gradient','ALADIN');
xlabel("k",'Interpreter','latex');
ylabel("||x^{k}-x^{*}||",'Interpreter','Tex');
function plot_result(err,line_width)
    err = [err{:}];
    t   = 0:size(err,2)-1;
    semilogy(t,err,'LineWidth',line_width);
end
