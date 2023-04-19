%% Cargar modelo y datos
addpath("Toolbox/Toolbox difuso");

load("Fuzzy/modelo_difuso.mat");
load("Data/split.mat");

ny = 10; %Cantidad de regresores de y
reg = [1 2 11 12];

X_train = split.X_train(:, reg);
Y_train = split.Y_train;

X_val = split.X_val(:, reg);
Y_val = split.Y_val;

%% Predicciones
alpha = 1e-5;
[Psi_train, sigma] = getCovIntervalParams(X_train, Y_train, modelFuzzy.a,modelFuzzy.b,modelFuzzy.g);
[y_pred_upper, y_pred, y_pred_lower] = covFuzzyInterval(Psi_train, sigma, X_val, modelFuzzy.a, modelFuzzy.b, modelFuzzy.g, alpha);

%% Plot
figure()
plot(Y_val, '.b')
hold on
plot(y_pred, 'r')
x = 1:1:length(Y_val);
fill([x fliplr(x)], [y_pred_lower' fliplr(y_pred_upper')], 'k', 'FaceAlpha', 0.2);

legend('Valor real', 'Valor esperado')