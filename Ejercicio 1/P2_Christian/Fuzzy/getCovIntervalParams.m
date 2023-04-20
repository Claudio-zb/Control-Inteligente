function [Psi_train, sigma] = getCovIntervalParams(X_train, Y_train,a,b,g)
% Encuenta los parametros necesario para realizar el intervalo por
% covarianza
%
% X_train: Matriz con los datos de entrenamiento
% Y_train: Vector con las salidas de entrenamiento
% a: the cluster's Std^-1 
% b: the cluster's center
% g: the consecuence parameters

% Nt numero de datos de entrenamiento
% n is the number of regressors of the TS model
[Nt,n]=size(X_train);

% NR is the number of rules of the TS model
NR=size(a,1);

Psi_train = zeros(NR, n+1, Nt); %Psi_j
e_train = zeros(Nt, NR); %e_j

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

    for j=1:NR
        y_kj = Wn(j)*Y_train(k);
        psi_kj = Wn(j)*[1, X_train(k, :)];
        e_train(k, j) = y_kj - psi_kj*g(j,:)';

        Psi_train(j, :, k) = psi_kj;
    end

    %Psi_train(:,:, k) = Wn'*[1, X_train(k, :)]; % (NRx1) * (1xn+1) = (NRxn+1)
    %yj_train(:, k) = Wn'*Y_train(k);  
    
end



% for j=1:NR
%     e_train(:, j) = yj_train(j, :)'- squeeze(Psi_train(j, :, :))'*g(j, :)';
% end

sigma = std(e_train, 0, 1); %(1xNR)

end