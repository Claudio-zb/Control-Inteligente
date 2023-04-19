%% Cargar modelo y datos
addpath("../Toolbox/Toolbox NN");

load("modelo_neuronal.mat");
load("split.mat");

X_val = split.X_val;
Y_val = split.Y_val;

reg = [1 2 3 4 5 6 7 8];
X_val_opt = X_val(:, reg);

%% Prediccion
n_pasos = 1;
%y_pred = my_ann_evaluation(modelNNStruct, X_val_opt');
y_pred = predictNN(modelNNStruct, X_val_opt, n_pasos, 4);

figure()
plot(split.Y_val, '.b')
hold on
plot(y_pred, 'r')

legend('Valor real', 'Valor esperado')

%% Metricas
rmse = RMSE(Y_val, y_pred);
mae = MAE(Y_val, y_pred);
mape = MAPE(Y_val, y_pred);