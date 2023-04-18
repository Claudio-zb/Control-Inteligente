%% Configurar entorno
clear all
clc
addpath("../Toolbox/Toolbox difuso");

%% Cargar datos
load("DatosP2.mat"); %datos
u = datos(:, 1);
y = datos(:, 2);

%% Mostrar datos
n_mostrar = 500; %Cantidad de muestras a mostrar
figure()
stairs(u(1:n_mostrar));
hold on
stairs(y(1:n_mostrar));
xlabel('Tiempo k')
ylabel('Amplitud')
legend('u(k)', 'y(k)')
title('Serie de Chen')

%% Parametros del modelo
nu = 4; %Cantidad regresores u
ny = 4; %Cantidad regresores y
max_clusters = 30; %Cantidad de clusters
porcentajes = [0.6 0.2 0.2]; %Porcentaje train, test, val

%% Crear split
[X, Y] = crearMatrices(u, y, nu, ny);
[X_train, Y_train, X_test, Y_test, X_val, Y_val] = createSplit(X, Y, porcentajes);

split = struct('X_train', X_train, 'Y_train', Y_train, 'X_test', X_test, 'Y_test', Y_test, 'X_val', X_val, 'Y_val', Y_val);
save("split.mat", "split");
%% Clustering - Optimizar cantidad de clusters
[err_test, err_ent] = clusters_optimo(Y_test, Y_train, X_test, X_train, max_clusters);
figure()
plot(err_test, 'b')
hold on
plot(err_ent, 'r')
legend('Error de test', 'Error de entrenamiento')

%% Fijar clusters
n_clusters = 14; %Cantidad de clusters elegidos

%% Sensibilidad - Optimizar cantidad de regresores
[p, indices] = sensibilidad(Y_train, X_train, n_clusters);

%% Fijar regresores
reg = [1 2 5 6];
X_train_opt = X_train(:, reg);
X_test_opt = X_test(:, reg);
X_val_opt = X_val(:, reg);

%% Entrenar modelo
opciones = [1 2 1]; %Todos los datos, gaussiana, Gustafson-Kessel
[modelFuzzy, ~] = TakagiSugeno(Y_train, X_train_opt, n_clusters, opciones);

save("modelo_difuso.mat", "modelFuzzy");
