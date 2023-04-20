function [x, fval] = fuzzyNumberParams(X, Y, modelFuzzy)

eta1 = 0.1;
eta2 = 0.9;
alpha = 0.1;
reg = 5;

%Restricciones
Aeq = [];
beq = [];
A = [];
b = [];

%Limites
s0 = [modelFuzzy.g, modelFuzzy.g];
smin = s0-0.5;
smax = s0+0.5;

fun = @(s)lossInterval(s, eta1, eta2, alpha, reg, X, Y, modelFuzzy);
options = optimoptions('particleswarm', 'MaxIterations', 100);
[x, fval] = particleswarm(fun, numel(s0), smin, smax, options);

% fun = @(s)lossPINAW(s, reg, X, Y, modelFuzzy);
% const = @(s)constPICP(s, reg, X, Y, modelFuzzy, alpha);
% 
% options = optimoptions('fmincon', 'MaxIterations',20);
% [x, fval] = fmincon(fun, s0, A, b, Aeq, beq, smin, smax, const, options);

end

function loss = lossInterval(s, eta1, eta2, alpha,  reg, X, Y, modelFuzzy)

s = reshape(s, [], reg*2);
sl = s(:,1:reg);
su = s(:,reg+1:end);

[y_pred_upper, ~, y_pred_lower] = fuzzyNumberInterval(X, modelFuzzy.a,modelFuzzy.b,modelFuzzy.g, sl, su);

pinaw = PINAW(Y, y_pred_lower, y_pred_upper);
picp = PICP(Y, y_pred_lower, y_pred_upper);

loss = eta1*pinaw+exp(-eta2*(picp-(1-alpha)));

end

function loss = lossPINAW(s, reg, X, Y, modelFuzzy)
sl = s(:,1:reg);
su = s(:,reg+1:end);

[y_pred_upper, ~, y_pred_lower] = fuzzyNumberInterval(X, modelFuzzy.a,modelFuzzy.b,modelFuzzy.g, sl, su);
loss = PINAW(Y, y_pred_lower, y_pred_upper);
end

function [dummy, const] = constPICP(s, reg, X, Y, modelFuzzy, alpha)
sl = s(:,1:reg);
su = s(:,reg+1:end);

[y_pred_upper, ~, y_pred_lower] = fuzzyNumberInterval(X, modelFuzzy.a,modelFuzzy.b,modelFuzzy.g, sl, su);
picp = PICP(Y, y_pred_lower, y_pred_upper);
const = picp -(1-alpha);
dummy = 0;
end


