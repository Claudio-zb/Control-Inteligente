fs = 100;
Ndatos = 0.5*fs; %1 segundos

u = zeros(Ndatos + n_reg,1);
u(floor(Ndatos/2):end) = 2;

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

stairs(y)
hold on
stairs(u)