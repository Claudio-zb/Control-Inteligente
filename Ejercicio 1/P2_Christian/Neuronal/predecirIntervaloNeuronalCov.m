%% Cargar modelo y datos
clear
clc
addpath("Toolbox/Toolbox NN");

load("Neuronal/modelo_neuronal.mat");
load("Data/split.mat");

ny = 2; %Cantidad de regresores de y
reg = [1 2 5 6];

X_train = split.X_train(:, reg);
Y_train = split.Y_train;

X_val = split.X_val(:, reg);
Y_val = split.Y_val;

%% Fit
alpha = 8.65;
[Z, sigma] = getCovParamsNeuronal(modelNNStruct, X_train', Y_train);

%% Prediccion

X = X_val;
Y = Y_val;

yp_up = zeros(length(Y), 1);
yp_low = zeros(length(Y), 1);
yp = zeros(length(Y), 1);
[Nd,~]=size(X);

p = 16; %Pasos

for k=1:Nd-p+1
    X_y = X(k, 1:ny); %Regresores de y
    for h=1:p
        X_u = X(k+h-1, ny+1:end);
        X_new = [X_y, X_u];
        [y_pred_upper, y_pred, y_pred_lower] = covNeuronalInterval(modelNNStruct, X_new', Z, sigma, alpha);

        X_y = circshift(X_y, 1); %Correr regresores de y
        X_y(1) = y_pred; %Agregar prediccion a los regresores

        if k+h-1<p
            yp_up(k+h-1) = y_pred_upper;
            yp_low(k+h-1) = y_pred_lower;
            yp(k+h-1) = y_pred;
        end
    end
    yp_up(k+p-1) = y_pred_upper;
    yp_low(k+p-1) = y_pred_lower;
    yp(k+p-1) = y_pred;
end

%% Metricas
pinaw = PINAW(Y, yp_low, yp_up)
picp = PICP(Y, yp_low, yp_up)
rmse = RMSE(Y, yp);
%% Plot
mostrar = 500;
figure()
plot(Y(1:mostrar), '.b')
hold on
plot(yp(1:mostrar), 'r')
x = 1:1:length(Y);
fill([x(1:mostrar) fliplr(x(1:mostrar))], [yp_low(1:mostrar)' fliplr(yp_up(1:mostrar)')], 'k', 'FaceAlpha', 0.2);

legend('Valor real', 'Valor esperado')
ylabel('Amplitud')
xlabel('Tiempo k')
title("Prediccion a " + p + " pasos" )
