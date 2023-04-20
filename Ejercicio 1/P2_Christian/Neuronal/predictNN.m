function y_pred = predictNN(net, X, p, ny, regresores)
% Works with only single INPUT vector
% Matrix version can be implemented
% net: Estructura de la red entrenada
% X: matriz con los datos
% p: pasos a predecir
% ny: regresores de y
% regresores: lista de los regresores a usar

%Parametros de normalizacion
ymax = net.input_ymax;
ymin = net.input_ymin;
xmax = net.input_xmax;
xmin = net.input_xmin;

ymax_out = net.output_ymax;
ymin_out = net.output_ymin;
xmax_out = net.output_xmax;
xmin_out = net.output_xmin;

%Numero de datos y regresores
[Nd,~]=size(X);

y_pred=zeros(Nd,1);

for k=1:Nd-p+1
    X_y = X(k, 1:ny); %Regresores de y
    for h=1:p
        X_u = X(k+h-1, ny+1:end); %Regresores de u
        X_new = [X_y, X_u]; %todos los regresores
        X_new = X_new(1, regresores); %Sacar regresores que no se usan
        X_new = X_new';
        input_preprocessed = (ymax-ymin) * (X_new-xmin) ./ (xmax-xmin) + ymin;

        %Pasarlo por la red
        y1 = tanh(net.IW * input_preprocessed + net.b1);
        y = net.LW * y1 + net.b2;
        y = (y-ymin_out) .* (xmax_out-xmin_out) /(ymax_out-ymin_out) + xmin_out;

        X_y = circshift(X_y, 1); %Correr regresores de y
        X_y(1) = y; %Agregar prediccion a los regresores

        if k+h-1<p
            y_pred(k+h-1) = y; %Arreglar primeras predicciones
        end
    end
    y_pred(k+p-1) = y; %Agregar prediccion p step
end
end