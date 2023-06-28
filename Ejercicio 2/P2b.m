% Cargar modelo difuso Ta
clear all
clc

addpath("P2a");
addpath("P2a/Toolbox difuso");

load("P2a/modelo.mat");
load("temperatura_10min.mat")

% Deberia ocuparse como temperatura real la de validacion, o si no el
% modelo difuso entregaria predicciones sobreajustadas.
% PARA ELLO DEJAR DESCOMENTADA LAS SIGUIENTES LINEAS
load("P2a/split.mat");
temperatura = split.Y_val;
%TAMBIEN SE PUEDE CORRER LA TEMPERATURA AMBIENTAL USADA
inicio = 721-9;
temperatura = temperatura(inicio:end);

%Cargar modelo difuso
modelFuzzy = modelo.modelFuzzy;
reg = modelo.reg;
ny = length(reg);

%Parametros de la simulacion
pasos1 = 6;
pasos2 = 6;
pasos3 = 6;
r1 = 20;
r2 = 18;
r3 = 25;
dt = 600; %En segundos
horizonte = 5;

pasos = pasos1+pasos2+pasos3;

Ta = temperatura(1:pasos+1);

%Datos
T1all = zeros(1, pasos+1);
T2all = zeros(1, pasos+1);
%Temperatura inicial
T1all(1) = 10.12;
T2all(1) = 10.12;

uall = zeros(1, pasos+1);

r = zeros(1, pasos+1);
r(1:pasos1) = r1;
r(pasos1+1:pasos1+pasos2) = r2;
r(pasos1+pasos2+1:pasos+1) = r3;

tiempos = zeros(1, pasos);
iterationsk = zeros(1, pasos);
for k=1:1:pasos
    tic
    [u, iterations] = mpc(horizonte, T1all(k), T2all(k), flip(temperatura(k:k+ny-1))', dt, modelFuzzy, reg, r(k), 1);
    tiempos(k) = toc;
    iterationsk(k) = iterations;
    uall(k) = u;
    [T1, T2] = predict(1, T1all(k), T2all(k), flip(temperatura(k:k+ny-1))', u, dt, modelFuzzy, reg);
    T1all(k+1) = T1 + randn()*0.001;
    T2all(k+1) = T2 + randn()*0.001;
end
uall(end) = uall(end-1);

% Esfuerzo computacional
display("Tiempo total: " + sum(tiempos) + " | Tiempo promedio: " + sum(tiempos)/length(tiempos))
display("Iteraciones total: " + sum(iterationsk) + " | Iteraciones promedio: " + sum(iterationsk)/length(iterationsk))

%Graficos
x = 0:dt/60:pasos*dt/60;
figure

subplot(2,1,1)
plot(x, T1all)
hold on
stairs(x, r)
plot(x, Ta)
legend('T1 real', 'Referencia', 'Ta', 'Location', 'southeast')
ylabel('Temperatura [°C]')
xlabel('Tiempo [min]')
title('Temperaturas')
ylim([min(T1all)-3, max(T1all)+3])

subplot(2,1,2)
plot(x, uall)
xlim([0, pasos*dt/60])
title("Entrada dms/dt")
ylabel('Flujo másico de aire [kg/s]')
xlabel('Tiempo [min]')
ylim([min(uall)-0.3, max(uall)+0.3])

function [T1, T2] = predict(horizonte, T1k, T2k, Taanteriores, u, dt, modelFuzzy, reg)
assert(horizonte==length(u), "Entrada no apta para el horizonte de prediccion")
ny = length(reg);
assert(length(Taanteriores)==ny, "Datos de Ta no aptos")

%Parametros
c1=2.508*10^6;
c2=4.636*10^7;
cp=1012;
R=1.7*10^(-3);
Ra=1.3*10^(-3);
delta=0.7;
DeltaT=13;

a1=(cp*(1-delta))/c1;
a2=1/(R*c1);
a3=1/(Ra*c1);
a4=(DeltaT*cp)/c1;
a5=1/c1;
a6=1/(R*c2);
a7=1/c2;

%Predecir Ta
Ta = zeros(1, horizonte);
Ta(1) = Taanteriores(1);
Tahist = Taanteriores;
for i=2:1:horizonte
    Ta(i) = predictFuzzy(Tahist, modelFuzzy.a, modelFuzzy.b, modelFuzzy.g, 1, ny, reg);
    Tahist = circshift(Tahist, 1);
    Tahist(1) = Ta(i);
end

%Calculo
T1 = zeros(1, horizonte+1);
T1(1) = T1k;
T2 = zeros(1, horizonte+1);
T2(1) = T2k;

for k=1:1:horizonte
    T1(k+1) = ( a1*(Ta(k)-T1(k))*u(k) + a2*(T2(k)-T1(k)) + a3*(Ta(k)-T1(k)) + a4*u(k) )*dt + T1(k);
    T2(k+1) = ( a6*(T1(k)-T2(k)) )*dt + T2(k);
end

T1 = T1(2:end);
T2 = T2(2:end);

end

function j = costo(r, T1)
j = sum((T1-r).^2);
end

function [u, iterations] = mpc(horizonte, T1k, T2k, Taanteriores, dt, modelFuzzy, reg, r, type)
assert(type==0||type==1, "Tipo no soportado")
fun = @(x)costo(r, predict(horizonte, T1k, T2k, Taanteriores, x, dt, modelFuzzy, reg));
if (type==0)
    options = optimoptions('fmincon', 'Algorithm','interior-point');
    [us,fval,exitflag,output] = fmincon(fun, zeros(1, horizonte), [], [], [], [], zeros(1,horizonte)+0.1, zeros(1,horizonte)+2, [], options);
    iterations = output.iterations;
elseif (type==1)
    options = optimoptions('particleswarm','SwarmSize', min(100, horizonte*10), 'MaxIterations', 200*horizonte);
    [us,fval,exitflag,output] = particleswarm(fun, horizonte,zeros(1,horizonte)+0.1,zeros(1,horizonte)+2,options);
    iterations = output.iterations;
end

u = us(1);
end
