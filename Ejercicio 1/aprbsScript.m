clear all
clc
%% Tiempo de establecimiento 
% tiempo estabilización: tiempo que demora la salida en estar dentro
% del 5% al rededor de y_ss (estado estacionario)

%load('simulacion1.mat')
%in = motor_data3(:,3);
%out = motor_data3(:,2);

figure(1)
plot(tout, in, tout, out)
%yline([9.55],'--')
%yline([9.35],'--',{'2%'})
grid on
xlabel('Tiempo [s]');ylabel('Amplitud');
title('Salida del sistema a entrada escalón')
legend({'Escalón','Salida'},'Location','northwest')

%% Señal APRBS 
% Una vez conocido el tiempo de estabilización (2 seg), podemos definir el 
% ancho mínimo del pulso pseudo aleatorio.
clear all

rng(2); % semilla random
tf = 1000; % tiempo final de señal
ts = 0.05; % tiempo sampleo
f_max = 0.0175; %0.11 rad/s max ; %relacionado al ancho mínimo del pulso
u = idinput(tf/ts,'prbs',[f_max*0.8 f_max*1.2],[1,2]);

a = 1; % lower amp limit
b = 4.5; % upper amp limit
d = diff(u);
idx = find(d) + 1; % changed to find(d)
idx = [1;idx];
for ii = 1:length(idx) - 1
     amp = (b-a).*rand + a;
     u(idx(ii):idx(ii+1)-1) = amp*u(idx(ii));
end
    % removed the normalization step
u = iddata([],u,1);
t = linspace(0, tf, tf/ts);
aprbs = timeseries(u.InputData, t);
figure(2);clf
plot(u)
title('Entrada APRBS')
ylabel('Amplitud');xlabel('Tiempo [s]')