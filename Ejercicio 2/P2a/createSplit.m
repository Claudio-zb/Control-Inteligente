function [X_train, Y_train, X_test, Y_test, X_val, Y_val] = createSplit(X, Y, porcentajes)
% Crea el split de los datos en train, test y val, de acuerdo a los
% porcentajes de cada uno
% Inputs:
%   X: matriz con los regresores
%   Y: vector con las salidas
%   porcentajes: vector con el porcentaje de cada conjunto

n = length(Y);
n_conjuntos = floor(n*porcentajes);
limits = cumsum(n_conjuntos);

X_train = X(1:limits(1), :);
Y_train = Y(1:limits(1), :);

X_test = X(limits(1)+1:limits(2), :);
Y_test = Y(limits(1)+1:limits(2), :);

X_val = X(limits(2)+1:limits(3), :);
Y_val = Y(limits(2)+1:limits(3), :);
end