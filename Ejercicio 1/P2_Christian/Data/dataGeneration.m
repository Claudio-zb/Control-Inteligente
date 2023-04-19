%% Parametros
rng(0);
fmax = 1; %Frecuencia maxima
fmin = 0.2; %Frecuencia minima
fs = 100; %Frecuencia de sampleo
Ndatos = 10000; %Cantidad de datos
rango = [-2, 2]; %Rango de la APRBS
n_niveles = 11; %Cantidad de niveles de la APRBS

%% Generacion de la entrada
u = aprbsGenerator(fmax, fmin, fs, Ndatos, rango, n_niveles); %Entrada APRBS
ruido = 0.1*randn(size(u)); %ruido gaussiano con dev std 0.1
u = u+ruido; %Entrada con ruido

%% Generacion de la salida
n_reg = 2; %Cantidad de regresores
e = zeros(Ndatos + n_reg, 1); %Ruido del sistema
y = zeros(Ndatos + n_reg, 1); %Salida
beta = randn(Ndatos + n_reg, 1); %Ruido blanco con dev std 1

for k=n_reg+1:Ndatos+n_reg
    e(k) = 0.5*exp(-(y(k-1)^2))*beta(k);
    y(k) = (0.5-0.3*exp(-(y(k-1)^2)))*y(k-1)...
            - (0.2+0.8*exp(-(y(k-1)^2)))*y(k-2)...
            + u(k-1) + 0.2*u(k-2) + 0.1*u(k-1)*u(k-2)+e(k);
end

%% Mostrar graficos
n_mostrar = 500; %Cantidad de muestras a mostrar
figure()
stairs(u(1:n_mostrar));
hold on
stairs(y(1:n_mostrar));
xlabel('Tiempo k')
ylabel('Amplitud')
legend('u(k)', 'y(k)')
title('Serie de Chen')

%% Guardar datos
filename = 'Data/DatosP2.mat';
datos = [u(1:size(y))', y];
save(filename, 'datos');

%% Generar split
nu = 10; %Cantidad regresores u
ny = 10; %Cantidad regresores y

porcentajes = [0.6 0.2 0.2]; %Porcentaje train, test, val

[X, Y] = crearMatrices(u, y, nu, ny);
[X_train, Y_train, X_test, Y_test, X_val, Y_val] = createSplit(X, Y, porcentajes);

split = struct('X_train', X_train, 'Y_train', Y_train, 'X_test', X_test, 'Y_test', Y_test, 'X_val', X_val, 'Y_val', Y_val);
save("Data/split.mat", "split");