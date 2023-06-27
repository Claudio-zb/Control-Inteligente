function [X, Y] = crearMatrizAutoregresores(y, ny)
% Crea las matrices con los regresores y las predicciones
% y: vector con la salida del sistema
% ny: cantidad de regresores de la salida

n_total = length(y); %Cantidad total de datos
n_generados = n_total-ny; %Cantidad posible de generar
X = zeros(n_generados, ny);
Y = zeros(n_generados, 1);

for i=1:n_generados
    Y(i) = y(i+ny);
    X(i, 1:ny) = y(i+ny-1:-1:i);
end