% Cargar modelo difuso Ta
clear all
clc

addpath("P2a");
addpath("P2a/Toolbox difuso");

load("P2a/modelo.mat");
load("temperatura_10min.mat")

modelFuzzy = modelo.modelFuzzy;
reg = modelo.reg;

pasos = 1000;

T1all = zeros(1, pasos+1);
T2all = zeros(1, pasos+1);
T1all(1) = 10.12;
T2all(1) = 10.12;
dt = 10;
horizonte = 5;
uall = zeros(1, pasos);
for k=1:1:pasos
    u = mpc(horizonte, T1all(k), T2all(k), flip(temperatura(k:k+4))', dt, modelFuzzy, reg, 20);
    uall(k) = u;
    [T1, T2] = predict(1, T1all(k), T2all(k), flip(temperatura(k:k+4))', u, dt, modelFuzzy, reg);
    T1all(k+1) = T1 + randn()*0.001;
    T2all(k+1) = T2 + randn()*0.001;
end



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

function u = mpc(horizonte, T1k, T2k, Taanteriores, dt, modelFuzzy, reg, r)
fun = @(x)costo(r, predict(horizonte, T1k, T2k, Taanteriores, x, dt, modelFuzzy, reg));
us = fmincon(fun, zeros(1, horizonte), [], [], [], [], zeros(1,horizonte)+0.1, zeros(1,horizonte)+2, []);
u = us(1);
end
