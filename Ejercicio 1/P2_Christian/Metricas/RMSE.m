function [rmse] = RMSE(y, y_pred)
% Calcula el RMSE

rmse = sqrt(mean((y_pred-y).^2));
end