%% Cargar modelo y datos
clc
clear
addpath("Toolbox/Toolbox NN");

load("Neuronal/modelo_neuronal.mat");
load("Data/split.mat");

ny = 2; %Cantidad de regresores de y
reg = [1 2 5 6];

X_train = split.X_train(:, reg);
Y_train = split.Y_train;

X_val = split.X_val(:, reg);
Y_val = split.Y_val;

sizeLast = 35;

%% Fit
alpha = 0.1;
params = fuzzyNumberParamsNeuronal(X_train, Y_train, modelNNStruct, alpha, sizeLast);

%% Prediccion
LWl = params(:,1:sizeLast);
bl = params(:, sizeLast+1);
LWu = params(:, sizeLast+2:end-1);
bu = params(:,end);

X = X_val;
Y = Y_val;
p = 16; %Pasos

yp_up = zeros(length(Y), 1);
yp_low = zeros(length(Y), 1);
yp = zeros(length(Y), 1);
[Nd,~]=size(X);

for k=1:Nd-p+1
    X_y = X(k, 1:ny); %Regresores de y
    for h=1:p
        X_u = X(k+h-1, ny+1:end);
        X_new = [X_y, X_u];
        [y_pred_upper, y_pred, y_pred_lower] = fuzzyNumberIntervalNeuronal(X_new', modelNNStruct, LWl, bl, LWu, bu);
        
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

%% Plot
mostrar = 500;
figure()
plot(Y(1:mostrar), '.b')
hold on
plot(yp(1:mostrar), 'r')
x = 1:1:length(Y);
fill([x(1:mostrar) fliplr(x(1:mostrar))], [yp_low(1:mostrar)' fliplr(yp_up(1:mostrar)')], 'k', 'FaceAlpha', 0.2);

legend('Valor real', 'Valor esperado')
legend('Valor real', 'Valor esperado')
ylabel('Amplitud')
xlabel('Tiempo k')
title("Prediccion a " + p + " pasos" )


