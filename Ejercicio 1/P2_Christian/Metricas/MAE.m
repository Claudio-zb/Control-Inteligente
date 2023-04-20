function [mae] = MAE(y, y_pred)
%Calcula el MAE

mae = mean(abs(y - y_pred));
end