function [y_pred_upper, y_pred, y_pred_lower, Psi_train] = covFuzzyInterval(X_train, Y_train, X_pred,a,b,g, alpha)
% Genera la prediccion de intevalos del modelo difuso por metodo de
% covarianza
%
% X_train: Matriz con los datos de entrenamiento
% Y_train: Vector con las salidas de entrenamiento
% X_pred: Matriz con los datos sobre los que se quiere predecir
% a: the cluster's Std^-1 
% b: the cluster's center
% g: the consecuence parameters

% Nt numero de datos de entrenamiento
% n is the number of regressors of the TS model
[Nt,n]=size(X_train);

% NR is the number of rules of the TS model
NR=size(a,1);

Psi_train = zeros(NR, n+1, Nt); %Psi_j
yj_train = zeros(NR, Nt);

for k=1:Nt

    % W(r) is the activation degree of the rule r
    % mu(r,i) is the activation degree of rule r, regressor i
    W=ones(1,NR);
    mu=zeros(NR,n);
    for r=1:NR
     for i=1:n
       mu(r,i)=exp(-0.5*(a(r,i)*(X_train(k,i)-b(r,i)))^2);  
       W(r)=W(r)*mu(r,i);
     end
    end

    % Wn(r) is the normalized activation degree
    if sum(W)==0
        Wn=W;
    else
        Wn=W/sum(W);
    end

    Psi_train(:,:, k) = Wn'*[1, X_train(k, :)]; % (NRx1) * (1xn+1) = (NRxn+1)
    yj_train(:, k) = Wn'*Y_train(k);  

end

e_train = zeros(Nt, NR); %e_j

for j=1:NR
    e_train(:, j) = yj_train(j, :)'-squeeze(Psi_train(j, :, :))'*[1; X_train(k, :)'];
end

sigma = std(e_train, 0, 1); %(1xNR)


%Datos de evaluacion
[Nv,~]=size(X_pred);
y_pred = zeros(Nv, 1);
y_pred_upper = zeros(Nv, 1);
y_pred_lower = zeros(Nv, 1);

for k=1:Nv

    % W(r) is the activation degree of the rule r
    % mu(r,i) is the activation degree of rule r, regressor i
    W=ones(1,NR);
    mu=zeros(NR,n);
    for r=1:NR
     for i=1:n
       mu(r,i)=exp(-0.5*(a(r,i)*(X_pred(k,i)-b(r,i)))^2);  
       W(r)=W(r)*mu(r,i);
     end
    end

    % Wn(r) is the normalized activation degree
    if sum(W)==0
        Wn=W;
    else
        Wn=W/sum(W);
    end

    psi_i = (Wn'*[1, X_pred(k, :)])'; % ((NRx1) * (1xn+1)) = (n+1xNR)

    dy = zeros(1,NR);
    
    for j=1:NR
        dy_ij = sigma(j)*sqrt(1+psi_i(:, j)'/( squeeze(Psi_train(j, :, :))*squeeze(Psi_train(j, :, :))' )*psi_i(:, j));
        dy(1, j) = dy_ij;
    end

    % Now we evaluate the consequences
    yr=g*[1 ;X_pred(k,:)'];  
    
    % Finally the output
    pred = Wn*yr;
    y_pred(k,1)=pred;

    I = sum(Wn.*dy);

    y_pred_upper(k, 1)= pred + alpha*I;
    y_pred_lower(k, 1) = pred - alpha*I;

end

end