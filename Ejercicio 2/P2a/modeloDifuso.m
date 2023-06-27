%% Configurar entorno
clear all
clc
addpath("Toolbox difuso");

%% Cargar datos
load("temperatura_10min.mat");

%% Crear split
ny = 20; %Cantidad de regresores a considerar
[X, Y] = crearMatrizAutoregresores(temperatura, ny);

porcentajes = [0.6, 0.2, 0.2]; %Porcentajes del split
[X_train, Y_train, X_test, Y_test, X_val, Y_val] = createSplit(X, Y, porcentajes);

split = struct('X_train', X_train, 'Y_train', Y_train, 'X_test', X_test, 'Y_test', Y_test, 'X_val', X_val, 'Y_val', Y_val);
save("split.mat", "split");

%% Cargar split
clear all
clc
load("split.mat")

%% Sensibilidad - Optimizar cantidad de regresores
n_clusters = 15; %Cantidad de clusters iniciales
[p, indices] = sensibilidad(split.Y_train, split.X_train, n_clusters);

%% Fijar regresores
reg = [1 2 3 4 5];
split.X_train = split.X_train(:, reg);
split.X_test = split.X_test(:, reg);
split.X_val = split.X_val(:, reg);

%% Clustering - Optimizar cantidad de clusters
max_clusters = 30; %Cantidad de clusters maxima
[err_test, err_ent] = clusters_optimo(split.Y_test, split.Y_train, split.X_test, split.X_train, max_clusters);

%% Graficar
figure()
plot(err_test, 'b')
hold on
plot(err_ent, 'r')
legend('Error de prueba', 'Error de entrenamiento')
xlabel('Cantidad de clusters')
ylabel('RMSE')
%title('Error con diferente cantidad de clusters')
saveas(gcf, 'clustersOptimo.png')

%% Entrenar modelo
n_clusters = 6;
opciones = [1 2 2]; %Todos los datos, gaussiana, Fuzzy C-means
[modelFuzzy, ~] = TakagiSugeno(split.Y_train, split.X_train, n_clusters, opciones);

modelo = struct('modelFuzzy', modelFuzzy, 'n_clusters', n_clusters, 'reg', reg);

save("modelo.mat", "modelo");

%% Evaluar modelo
n_pasos = 1;
reg = modelo.reg;
ny = length(reg);
y_pred = predictFuzzy(split.X_test, modelFuzzy.a, modelFuzzy.b, modelFuzzy.g, n_pasos, ny, reg);

rmse = RMSE(split.Y_test, y_pred);
mae = MAE(split.Y_test, y_pred);
mape = MAPE(split.Y_test, y_pred);

display("RMSE: " + rmse + " | MAE: " + mae + " | MAPE: " + mape + "%")

cantidad = 500;

figure()
plot(split.Y_test(end-cantidad:end), '.b')
hold on
plot(y_pred(end-cantidad:end), 'r')

legend('Valor real', 'Valor esperado')
ylabel('Amplitud')
xlabel('Tiempo k')
xlim([0, cantidad])