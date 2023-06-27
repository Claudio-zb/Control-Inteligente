%% Cargar modelo y datos
clear all
clc
addpath("Toolbox difuso");
addpath("Metricas");

load("modelo.mat");
load("split.mat");

X_val = split.X_val;
Y_val = split.Y_val;

modelFuzzy = modelo.modelFuzzy;
reg = modelo.reg;
ny = length(reg);
%% Predicciones
%y_pred = ysim(X_val_opt, modelFuzzy.a, modelFuzzy.b, modelFuzzy.g); %A un paso
n_pasos = 5;
y_pred = predictFuzzy(X_val, modelFuzzy.a, modelFuzzy.b, modelFuzzy.g, n_pasos, ny, reg);

%% Graficar
cantidad = 500;
figure()
plot(Y_val(end-500:end), '.b')
hold on
plot(y_pred(end-500:end), 'r')

legend('Valor real', 'Valor esperado')
ylabel('Amplitud')
xlabel('Tiempo k')
xlim([0, cantidad])
%title("Prediccion a " + n_pasos + " pasos" )

%% Metricas
rmse = RMSE(Y_val, y_pred);
mae = MAE(Y_val, y_pred);
mape = MAPE(Y_val, y_pred);

display("RMSE: " + rmse + " | MAE: " + mae + " | MAPE: " + mape + "%")