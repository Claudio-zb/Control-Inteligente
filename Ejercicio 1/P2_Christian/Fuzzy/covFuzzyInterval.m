function [y_pred_upper, y_pred, y_pred_lower] = covFuzzyInterval(factor, sigma, X_pred,a,b,g, alpha)
% Genera la prediccion de intevalos del modelo difuso por metodo de
% covarianza
%
% Psi_train, sigma: Parametros del intervalo difuso
% X_pred: Matriz con los datos sobre los que se quiere predecir
% a: the cluster's Std^-1 
% b: the cluster's center
% g: the consecuence parameters

% NR is the number of rules of the TS model
NR=size(a,1);

%Datos de evaluacion
[Nv,n]=size(X_pred);
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

    dy = zeros(1,NR);

    for j=1:NR
        psi_kj = Wn(j)*[1; X_pred(k, :)'];
        dy_ij = sigma(j)*sqrt(1+psi_kj'*( squeeze(factor(j,:,:)))*psi_kj);
        dy(1, j) = dy_ij;
    end

    %psi_i = (Wn'*[1, X_pred(k, :)])'; % ((NRx1) * (1xn+1)) = (n+1xNR)

    % for j=1:NR
    %     dy_ij = sigma(j)*sqrt(1+psi_i(:, j)'/( squeeze(Psi_train(j, :, :))*squeeze(Psi_train(j, :, :))' )*psi_i(:, j));
    %     dy(1, j) = dy_ij;
    % end

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