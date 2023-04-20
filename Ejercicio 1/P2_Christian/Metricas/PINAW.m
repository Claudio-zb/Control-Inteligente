function pinaw = PINAW(y_target, y_lower, y_upper)

N = length(y_target);

R = max(y_target) - min(y_target);

pinaw = 1/(N*R)*sum(y_upper-y_lower);

end