%% Parametros de la PRBS
n = 5;            %Orden (CAMBIAR)
a = [0 1 0 0 1];  %Coeficientes (CAMBIAR)
N = 2^n-1;        %Periodo
reg = ones(1, n); %Registro

%% Parametros de la amplitud
minimo = -2;    %Amplitud minima
maximo = 2;     %Amplitud maxima
n_niveles = 11; %Cantidad de niveles de amplitud (si el rango es simetrico, realmente es el doble)
niveles = linspace(0, 1, n_niveles); %Niveles de amplitud

%% Generacion de la APRBS

muestras = 100; %Cantidad de muestras a generar (CAMBIAR)
salidas = zeros(1, muestras); %Vector con las salidas
amp_mod = true; %Activar la modificacion de la amplitud

for i=1:muestras
    out = mod(sum(a.*reg), 2); %Calcular la salida
    reg = circshift(reg, 1); %Correr el registro
    reg(1) = out; %Agregar salida calculada al registro

    if amp_mod
        out = (maximo-minimo)*out + minimo; %Dejarlo entre el rango
        amplitud = niveles(randi(n_niveles)); %Nivel de amplitud aleatorio. Moverlo en el rango
        out = amplitud*out; %Modificar la amplitud de la prbs
    end
    salidas(i) = out; %Agregar salida generada
end

%% Graficar
stairs(salidas);