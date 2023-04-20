function [thetaL, thetaU] = minMaxParams(X, Y, modelFuzzy)


%Restricciones
Aeq = [];
beq = [];
A = [];
b = [];

%Limites
thetaL0 = modelFuzzy.g;
thetaU0 = modelFuzzy.g;

fun = @(theta)abs(lossMinMax(theta, X, Y, modelFuzzy));
constLf = @(theta)constL(theta, X, Y, modelFuzzy);
constUf = @(theta)constU(theta, X, Y, modelFuzzy);

options = optimoptions('fminimax','AbsoluteMaxObjectiveCount',1,'MaxIterations',10);
[thetaL, ~] = fminimax(fun, thetaL0, A, b, Aeq, beq, [], [], constLf, options);
[thetaU, ~] = fminimax(fun, thetaU0, A, b, Aeq, beq, [], [], constUf, options);

end

function prediction = predict(X, a,b,g)
% Creates the model's predicction
% y is the vector of outputs when evaluating the TS defined by a,b,g
% X is the data matrix
% a is the cluster's Std^-1 
% b is the cluster's center
% g is the consecuence parameters

% Nd number of point we want to evaluate
% n is the number of regressors of the TS model

[Nd,n]=size(X);

% NR is the number of rules of the TS model
NR=size(a,1);         
prediction=zeros(Nd,1);

     
for k=1:Nd 
    
    % W(r) is the activation degree of the rule r
    % mu(r,i) is the activation degree of rule r, regressor i
    W=ones(1,NR);
    mu=zeros(NR,n);
    for r=1:NR
     for i=1:n
       mu(r,i)=exp(-0.5*(a(r,i)*(X(k,i)-b(r,i)))^2);  
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
    yr=g*[1 ;X(k,:)'];  
    
    % Finally the output
    prediction(k,1)=Wn*yr;


end


end

function loss = lossMinMax(theta, X, Y, modelFuzzy)

prediction = predict(X, modelFuzzy.a,modelFuzzy.b,theta);
loss = Y-prediction;
end

function [val, dummy] = constL(theta, X, Y, modelFuzzy)
loss = lossMinMax(theta, X, Y, modelFuzzy);
val = -loss;
dummy = 0;
end

function [val, dummy] = constU(theta, X, Y, modelFuzzy)
loss = lossMinMax(theta, X, Y, modelFuzzy);
val = loss;
dummy= 0;
end