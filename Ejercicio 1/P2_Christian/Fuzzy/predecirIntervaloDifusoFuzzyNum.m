%% Cargar modelo y datos
addpath("Toolbox/Toolbox difuso");

load("Fuzzy/modelo_difuso.mat");
load("Data/split.mat");

ny = 2; %Cantidad de regresores de y
reg = [1 2 5 6];

X_train = split.X_train(:, reg);
Y_train = split.Y_train;

X_test = split.X_test(:, reg);
Y_test = split.Y_test;

X_val = split.X_val(:, reg);
Y_val = split.Y_val;

%% Fit
alpha = 0.1;
p = 1;

[x, ~] = fuzzyNumberParams(X_test, Y_test, modelFuzzy, length(reg), alpha);

%[Psi_train, sigma] = getCovIntervalParams(X_train, Y_train, modelFuzzy.a,modelFuzzy.b,modelFuzzy.g);

%% Prediccion

X = X_val;
Y = Y_val;

[Nv,n]=size(X);
NR = size(modelFuzzy.a,1);
factor = zeros(NR, n+1, n+1);

for j=1:NR
    factor(j, :, :) = inv(squeeze(Psi_train(j, :, :))*squeeze(Psi_train(j, :, :))');
end

yp_up = zeros(length(Y), 1);
yp_low = zeros(length(Y), 1);
yp = zeros(length(Y), 1);
[Nd,~]=size(X);

p = 1; %Pasos

for k=1:Nd-p+1
    X_y = X(k, 1:ny); %Regresores de y
    for h=1:p
        X_u = X(k+h-1, ny+1:end);
        X_new = [X_y, X_u];
        [y_pred_upper, y_pred, y_pred_lower] = covFuzzyInterval(factor, sigma, X_new, modelFuzzy.a, modelFuzzy.b, modelFuzzy.g, alpha);
        yp_up(k) = y_pred_upper;
        yp_low(k) = y_pred_lower;
        yp(k) = y_pred;

        X_y = circshift(X_y, 1); %Correr regresores de y
        X_y(1) = y_pred; %Agregar prediccion a los regresores

        if k+h-1<p
            yp_up(k+h-1) = y_pred_upper;
            yp_low(k+h-1) = y_pred_lower;
            yp(k+h-1) = y_pred;
        end
    end
end

%% Metricas
pinaw = PINAW(Y, yp_low, yp_up);
picp = PICP(Y, yp_low, yp_up);

%% Plot
mostrar = 500;
figure()
plot(Y(1:500), '.b')
hold on
plot(y_pred, 'r')
x = 1:1:length(Y);
fill([x(1:500) fliplr(x(1:500))], [yp_low(1:500)' fliplr(yp_up(1:500)')], 'k', 'FaceAlpha', 0.2);

legend('Valor real', 'Valor esperado')

