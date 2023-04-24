%% Configurar entorno
clear all
clc
addpath("Toolbox/Toolbox NN");

%% Cargar datos
load("Data/split.mat");

%% Parametros modelo
max_hlayer = 5:5:40;

%% Optimizar capas ocultas
errores_test = zeros(1, length(max_hlayer)); %Errores de cada red con distintos layers
errores_train = zeros(1, length(max_hlayer)); 
net.dummy = 0; %Guardar redes
i = 1;
for h=max_hlayer
    net_ent = fitnet(h); % h neuronas en capa oculta
    net_ent.trainFcn = 'trainscg'; % Funcion de entrenamiento
    net_ent.trainParam.showWindow=0; % Evita que se abra la ventana de entrenamiento
    net_ent = train(net_ent,split.X_train',split.Y_train', 'useParallel','yes');
    
    y_p_test = net_ent(split.X_test')'; % Se genera una prediccion en conjunto de test
    errtest = RMSE(split.Y_test, y_p_test);%(sqrt(sum((y_p_test-split.Y_test).^2)) )/length(split.Y_test); % Se guarda el error de test
    y_p_train = net_ent(split.X_train')'; % Se genera una prediccion en conjunto de train
    errtrain= RMSE(split.Y_train, y_p_train);%(sqrt(sum((y_p_train-split.Y_train).^2)))/length(split.Y_train); % Se guarda el error de train
    
    errores_test(i) = errtest; %Guardar error
    errores_train(i) = errtrain; %Guardar error
    i = i + 1;

    net.("H"+h) = net_ent;
end
%% Graficar
figure()
plot(max_hlayer, errores_test, 'b')
hold on
plot(max_hlayer, errores_train, 'r')
legend('Error de test', 'Error de entrenamiento')
xlabel('Cantidad de neuronas capa oculta')
ylabel('RMSE')
title('Error con diferente cantidad de neuronas')
saveas(gcf, 'Figuras/hiddenOptimo.png')
%% Fijar modelo
hlayer = 35; %Capas ocultas mejor modelo
%% Guardado
net_ent = net.("H"+hlayer);
%% Entrenar de cero
net_ent = fitnet(hlayer); % h neuronas en capa oculta
net_ent.trainFcn = 'trainscg'; % Funcion de entrenamiento
net_ent.trainParam.showWindow=0; % Evita que se abra la ventana de entrenamiento
net_ent = train(net_ent,split.X_train',split.Y_train', 'useParallel','yes');
%% Sensibilidad - Optimizar cantidad de regresores
[p, indices] = sensibilidad_nn(split.X_train, net_ent);

%% Fijar regresores
reg = [1 2 5 6];
X_train_opt = split.X_train(:, reg);
X_test_opt = split.X_test(:, reg);
X_val_opt = split.X_val(:, reg);

%% Entrenar modelo final
modelNN = fitnet(hlayer);
modelNN.trainFcn = 'trainscg';  
modelNN.trainParam.showWindow=0;
modelNN = train(modelNN,X_train_opt',split.Y_train', 'useParallel','yes');

y_p_test = modelNN(X_test_opt')'; 
errtest = RMSE(split.Y_test, y_p_test);
y_p_train = modelNN(X_train_opt')';
errtrain= RMSE(split.Y_train, y_p_train);

%% Guardar modelo
modelNNStruct = my_ann_exporter(modelNN);

save("Neuronal/modelo_neuronal.mat", "modelNNStruct");