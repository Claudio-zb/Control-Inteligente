%% Cargar modelo y datos
addpath("Toolbox/Toolbox difuso");

load("Fuzzy/modelo_difuso.mat");
load("Data/split.mat");

X_val = split.X_val;
Y_val = split.Y_val;

ny = 10; %Cantidad de regresores de y
reg = [1 2 11 12];
%% Predicciones
%y_pred = ysim(X_val_opt, modelFuzzy.a, modelFuzzy.b, modelFuzzy.g); %A un paso
n_pasos = 16;
y_pred = predictFuzzy(X_val, modelFuzzy.a, modelFuzzy.b, modelFuzzy.g, n_pasos, ny, reg);

figure()
plot(Y_val, '.b')
hold on
plot(y_pred, 'r')

legend('Valor real', 'Valor esperado')

%% Metricas
rmse = RMSE(Y_val, y_pred);
mae = MAE(Y_val, y_pred);
mape = MAPE(Y_val, y_pred);
