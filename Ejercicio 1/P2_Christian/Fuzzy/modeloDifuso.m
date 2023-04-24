%% Configurar entorno
clear all
clc
addpath("Toolbox/Toolbox difuso");

%% Cargar datos
load("split.mat");

%% Parametros del modelo
max_clusters = 30; %Cantidad de clusters

%% Clustering - Optimizar cantidad de clusters
[err_test, err_ent] = clusters_optimo(split.Y_test, split.Y_train, split.X_test, split.X_train, max_clusters);
%% Graficar
figure()
plot(err_test, 'b')
hold on
plot(err_ent, 'r')
legend('Error de test', 'Error de entrenamiento')
xlabel('Cantidad de clusters')
ylabel('RMSE')
title('Error con diferente cantidad de clusters')
saveas(gcf, 'Figuras/clustersOptimo.png')

%% Fijar clusters
n_clusters = 13; %Cantidad de clusters elegidos

%% Sensibilidad - Optimizar cantidad de regresores
[p, indices] = sensibilidad(split.Y_train, split.X_train, n_clusters);

%% Fijar regresores
reg = [1 2 5 6];
X_train_opt = split.X_train(:, reg);
X_test_opt = split.X_test(:, reg);
X_val_opt = split.X_val(:, reg);

%% Entrenar modelo
opciones = [1 2 2]; %Todos los datos, gaussiana, Fuzzy C-means
[modelFuzzy, ~] = TakagiSugeno(split.Y_train, X_train_opt, n_clusters, opciones);

save("Fuzzy/modelo_difuso.mat", "modelFuzzy");
