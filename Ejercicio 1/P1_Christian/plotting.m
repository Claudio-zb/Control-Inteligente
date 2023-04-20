t = out.time;
V = out.inV;
T = out.outV;

figure(1);
plot(t, V);
title("Entrada V")
xlabel("Tiempo [s]")
ylabel("Flujo [l/s]")
figure(2);
plot(t, T, "b");
xlabel("Tiempo [s]")
ylabel("Temperatura salida [°C]")
title("Temperatura de salida del aceite")

%% Respuesta escalon
hold on
yline(264+11*0.1, "r--");
yline(264+11*0.9, "r--");
hold off

%% Escalon irradiancia
t = out.time;
Irr = out.inIrr;
Tirr = out.outIrr;

figure(1);
plot(t, Irr);
title("Perturbación Irradiancia")
xlabel("Tiempo [s]")
ylabel("Radiación solar [W/m2]")
figure(2);
plot(t, Tirr, "b");
xlabel("Tiempo [s]")
ylabel("Temperatura salida [°C]")
title("Temperatura de salida del aceite")