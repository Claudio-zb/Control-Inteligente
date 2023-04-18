function y_pred = predictFuzzy(X,a,b,g,p, ny)
% Creates the model's predicction
% y is the vector of outputs when evaluating the TS defined by a,b,g
% X is the data matrix
% a is the cluster's Std^-1 
% b is the cluster's center
% g is the consecuence parameters
% p: pasos a predecir
% ny: cantidad de regresores de y
%
% WARNING: Aqui se hace la suposicion de que los regresores de y son los
% ultimos (i.e., y(k-1), y(k-2), ...) y no estan saltados (e.g., y(k-2), y(k-5)...)

% Nd number of point we want to evaluate
% n is the number of regressors of the TS model
[Nd,n]=size(X);

% NR is the number of rules of the TS model
NR=size(a,1);         
y_pred=zeros(Nd,1);
         
     
for k=1:Nd-p+1

    X_y = X(k, 1:ny); %Regresores de y
    

    for h=1:p
        X_u = X(k+h-1, ny+1:end); %Regresores de u
        X_new = [X_y, X_u]; %todos los regresores

        % W(r) is the activation degree of the rule r
        % mu(r,i) is the activation degree of rule r, regressor i
        W=ones(1,NR);
        mu=zeros(NR,n);
        for r=1:NR
         for i=1:n
           mu(r,i)=exp(-0.5*(a(r,i)*(X_new(i)-b(r,i)))^2);  
           W(r)=W(r)*mu(r,i);
         end
        end

        % Wn(r) is the normalized activation degree
        if sum(W)==0
            Wn=W;
        else
            Wn=W/sum(W);
        end

        % Now we evaluate the consequences
        yr=g*[1 ;X_new'];  

        % Finally the output
        y=Wn*yr;

        X_y = circshift(X_y, 1); %Correr regresores de y
        X_y(1) = y; %Agregar prediccion a los regresores
    end

    y_pred(k+p-1) = y; %Agregar prediccion p step

end
end