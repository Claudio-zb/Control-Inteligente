function [X, Y] = crearMatrices(u, y, nu, ny)
% Crea las matrices con los regresores y las predicciones
% u: vector con la entrada del sistema
% y: vector con la salida del sistema
% nu: cantidad de regresores de la entrada
% ny: cantidad de regresores de la salida

n_total = length(y); %Cantidad total de datos
n_generados = n_total-max(nu,ny); %Cantidad posible de generar
X = zeros(n_generados, nu+ny);
Y = zeros(n_generados, 1);

for i=1:n_generados
    Y(i) = y(i+max(nu, ny));
    X(i, 1:ny) = y(i+ny-1:-1:i);
    X(i, ny+1: ny+nu) = u(i+nu-1:-1:i);
end