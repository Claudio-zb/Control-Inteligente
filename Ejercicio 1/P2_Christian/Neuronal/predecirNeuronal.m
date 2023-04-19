%% Cargar modelo y datos
addpath("Toolbox/Toolbox NN");

load("Neuronal/modelo_neuronal.mat");
load("Data/split.mat");

X_val = split.X_val;
Y_val = split.Y_val;

ny = 10; %Cantidad de regresores de y
reg = [1 2 11 12];

%% Prediccion
n_pasos = 1;
%y_pred = my_ann_evaluation(modelNNStruct, X_val_opt');
y_pred = predictNN(modelNNStruct, X_val, n_pasos, ny, reg);

figure()
plot(split.Y_val, '.b')
hold on
plot(y_pred, 'r')

legend('Valor real', 'Valor esperado')

%% Metricas
rmse = RMSE(Y_val, y_pred);
mae = MAE(Y_val, y_pred);
mape = MAPE(Y_val, y_pred);