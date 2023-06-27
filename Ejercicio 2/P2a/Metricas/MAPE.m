function [mape] = MAPE(y, y_pred)
%Calcula el MAPE

mape = mean(abs(y - y_pred) ./ abs(y)) * 100;

end