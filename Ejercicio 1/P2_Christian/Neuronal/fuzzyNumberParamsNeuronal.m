function x = fuzzyNumberParamsNeuronal(X, Y, modelNNStruct)

% %Restricciones
% Aeq = [];
% beq = [];
% A = [];
% b = [];
% 
% sizeLast = 15; %tamano ultima capa
% alpha = 0.2;
% 
% params0 = [modelNNStruct.LW-0.01, modelNNStruct.b2-0.01,modelNNStruct.LW+0.01, modelNNStruct.b2+0.01];
% 
% 
% fun = @(params)lossPINAWNN(params, sizeLast, X, Y, modelNNStruct);
% const = @(params)constPICPNN(params, sizeLast, X, Y, modelNNStruct, alpha);
% 
% options = optimoptions('fmincon', 'MaxIterations',500);
% [x, ~] = fmincon(fun, params0, A, b, Aeq, beq, params0-2, params0+2, const, options);

eta1 = 0.1;
eta2 = 0.9;
alpha = 0.1;
sizeLast = 15; %tamano ultima capa

params0 = [modelNNStruct.LW, modelNNStruct.b2,modelNNStruct.LW, modelNNStruct.b2];
lb = params0-0.5;
ub = params0+0.5;

fun = @(params)lossInterval(params, eta1, eta2, alpha,sizeLast, X, Y, modelNNStruct);

x = particleswarm(fun,(sizeLast+1)*2, lb, ub);

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
dummy = 0;
end