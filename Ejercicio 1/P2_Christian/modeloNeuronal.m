%% Configurar entorno
clear all
clc
addpath("../Toolbox/Toolbox NN");

%% Cargar datos
load("split.mat");

%% Parametros modelo
max_hlayer = 5:5:30;

%% Optimizar capas ocultas
errores = zeros(1, length(max_hlayer)); %Errores de cada red con distintos layers
net.dummy = 0; %Guardar redes
i = 1;
for h=max_hlayer
    net_ent = fitnet(h); % h neuronas en capa oculta
    net_ent.trainFcn = 'trainscg'; % Funcion de entrenamiento
    net_ent.trainParam.showWindow=1; % Evita que se abra la ventana de entrenamiento
    net_ent = train(net_ent,split.X_train',split.Y_train', 'useParallel','yes');
    
    y_p_test = net_ent(split.X_test')'; % Se genera una prediccion en conjunto de test
    errtest= (sqrt(sum((y_p_test-split.Y_test).^2)))/length(split.Y_test); % Se guarda el error de test
    
    errores(i) = errtest; %Guardar error
    i = i + 1;

    net.("H"+h) = net_ent;
end
%% Fijar modelo
hlayer = 5; %Capas ocultas mejor modelo
%% Guardado
net_ent = net.("H"+hlayer);
%% Entrenar de cero
net_ent = fitnet(hlayer); % h neuronas en capa oculta
net_ent.trainFcn = 'trainscg'; % Funcion de entrenamiento
net_ent.trainParam.showWindow=1; % Evita que se abra la ventana de entrenamiento
net_ent = train(net_ent,split.X_train',split.Y_train', 'useParallel','yes');
%% Sensibilidad - Optimizar cantidad de regresores
[p, indices] = sensibilidad_nn(split.X_train, net_ent);

%% Fijar regresores
reg = [1 2 3 4 5 6 7 8];
X_train_opt = split.X_train(:, reg);
X_test_opt = split.X_test(:, reg);
X_val_opt = split.X_val(:, reg);

%% Entrenar modelo final
modelNN = fitnet(hlayer);
modelNN.trainFcn = 'trainscg';  
modelNN.trainParam.showWindow=0;
modelNN = train(modelNN,X_train_opt',split.Y_train', 'useParallel','yes');

%% Guardar modelo
modelNNStruct = my_ann_exporter(modelNN);

save("modelo_neuronal.mat", "modelNNStruct");