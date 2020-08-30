function [x,lambda]=solve_cvx(A,b,kappa_lasso,Num_group,ni)

    n = size(A,2);
    m = size(b,1);
    cvx_precision best
    cvx_begin quiet
        variables x(n) x_bar(m)
        expression xx(Num_group)
        dual variable lambda;
        for ii = 1:Num_group
            xx(ii) = kappa_lasso * norm(x((ii-1)*ni+1:ii*ni),2);
        end
        minimize (0.5*square_pos(norm(x_bar-b,2))+ sum(xx,1))
        subject to
        lambda : x_bar - A * x == 0;
    cvx_end        

end