function z = shrinkage(x, kappa)
   z = pos(1- (kappa + 1e-16)/(norm(x) + 1e-16))*x;
end