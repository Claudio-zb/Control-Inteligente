function [aprbs] = aprbsGenerator(fmax, fmin, fs, Ndatos, rango, n_niveles)
%fmax: frecuencia maxima de interes
%fmin: frecuencia minima de interes
%fs: frecuencia de sampleo del sistema (se debe cumplir fs>2fmax)
%Ndatos: cantidad de datos
%rango: Rango de la senal generada
%n_niveles: cantidad de niveles de amplitud (se divide el rango [0,1])

fc = 2.5*fmax; %Frecuencia de bit
Ns = floor(fs/fc); %Muestras por bit (si cambio cada 5s y muestreo a 2.5s,
           % se tienen dos muestras por bit) No estoy seguro si va con
           % floor
n = ceil(log(fc/fmin+1)/log(2)); %Orden del prbs
periodo = 2^n-1; %periodo de la prbs
n_periodos = ceil(Ndatos/(Ns*periodo)); %Un periodo completo es de P=Ns*periodo.
                                    % En N muestras hay N/P periodos
B = 1/Ns; %inverso de la cantidad de muestras en donde la senal no cambia

prbs = idinput([periodo*Ns 1 n_periodos], 'prbs', [0 B], rango);

niveles = linspace(0, 1, n_niveles); %Dividir rango [0,1] en niveles
n_variaciones = periodo*n_periodos; %largo/Ns

for i=1:n_variaciones
    aleatorio = randi(n_niveles, 1); %Nivel aleatorio
    amp = niveles(aleatorio);
    aprbs((i-1)*Ns+1:i*Ns) = amp*prbs((i-1)*Ns+1:i*Ns); %Asignar nivel aleatorio en Ns muestras
end

end