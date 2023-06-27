function picp = PICP(y_target, y_lower,  y_upper)

N = length(y_target);

C = (y_lower <= y_target) &  (y_target <= y_upper);

picp = 1/N*sum(C);

end