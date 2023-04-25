function x = fuzzyNumberParamsNeuronal(X, Y, modelNNStruct, alpha, sizeLast)

% %Restricciones
Aeq = [];
beq = [];
A = [];
b = [];

params0 = [modelNNStruct.LW, modelNNStruct.b2,modelNNStruct.LW, modelNNStruct.b2];
p0 = zeros(size(params0));
lb = p0-0.1;
ub = p0+0.1;


% eta1 = 0.1;
% eta2 = 0.9;
% fun = @(params)lossInterval(params, eta1, eta2, alpha,sizeLast, X, Y, modelNNStruct);
% 
% x = particleswarm(fun,numel(params0), lb, ub);

fun = @(params)lossPINAWNN(params, sizeLast, X, Y, modelNNStruct);
const = @(params)constPICPNN(params, sizeLast, X, Y, modelNNStruct, alpha);

options = optimoptions('fmincon', 'MaxIterations',1000);
[x, ~] = fmincon(fun, params0, A, b, Aeq, beq, lb, ub, const, options);

end

function loss = lossInterval(params, eta1, eta2, alpha, sizeLast, X, Y, modelNN)

LWl = params(:,1:sizeLast);
bl = params(:, sizeLast+1);
LWu = params(:, sizeLast+2:end-1);
bu = params(:,end);

[y_pred_upper, ~, y_pred_lower] = fuzzyNumberIntervalNeuronal(X', modelNN, LWl, bl, LWu, bu);

pinaw = PINAW(Y, y_pred_lower, y_pred_upper);
picp = PICP(Y, y_pred_lower, y_pred_upper);

loss = eta1*pinaw+exp(-eta2*(picp-(1-alpha)));

end



function loss = lossPINAWNN(params, sizeLast, X, Y, modelNN)
LWl = params(:,1:sizeLast);
bl = params(:, sizeLast+1);
LWu = params(:, sizeLast+2:end-1);
bu = params(:,end);

[y_pred_upper, ~, y_pred_lower] = fuzzyNumberIntervalNeuronal(X', modelNN, LWl, bl, LWu, bu);
loss = PINAW(Y, y_pred_lower, y_pred_upper);
end

function [dummy, const] = constPICPNN(params, sizeLast, X, Y, modelNN, alpha)
LWl = params(:,1:sizeLast);
bl = params(:, sizeLast+1);
LWu = params(:, sizeLast+2:end-1);
bu = params(:,end);

[y_pred_upper, ~, y_pred_lower] = fuzzyNumberIntervalNeuronal(X', modelNN, LWl, bl, LWu, bu);
picp = PICP(Y, y_pred_lower, y_pred_upper);
const = picp -(1-alpha);
loss = PINAW(Y, y_pred_lower, y_pred_upper);
dummy = -loss;
end