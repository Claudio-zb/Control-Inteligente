%% Cargar modelo y datos
addpath("Toolbox/Toolbox NN");

load("Neuronal/modelo_neuronal.mat");
load("Data/split.mat");

ny = 10; %Cantidad de regresores de y
reg = [1 2 11 12];

X_train = split.X_train(:, reg);
Y_train = split.Y_train;

X_val = split.X_val(:, reg);
Y_val = split.Y_val;

%% Predicciones
alpha = 5;
[Z, sigma] = getCovParamsNeuronal(modelNNStruct, X_train', Y_train);
[y_pred_upper, y_pred, y_pred_lower] = covNeuronalInterval(modelNNStruct, X_val', Z, sigma, alpha);

%% Plot
figure()
plot(Y_val, '.b')
hold on
plot(y_pred, 'r')
x = 1:1:length(Y_val);
fill([x fliplr(x)], [y_pred_lower' fliplr(y_pred_upper')], 'k', 'FaceAlpha', 0.2);

legend('Valor real', 'Valor esperado')

%% Metricas
pinaw = PINAW(Y_val, y_pred_lower, y_pred_upper);
picp = PICP(Y_val, y_pred_lower, y_pred_upper);