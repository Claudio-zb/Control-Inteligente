function [y_pred_upper, y_pred, y_pred_lower] = fuzzyNumberInterval(X, a,b,g, sl, su)
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
y_pred=zeros(Nd,1);
y_pred_upper=zeros(Nd,1);
y_pred_lower=zeros(Nd,1);
         
     
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
    y_pred(k,1)=Wn*yr;

    %Upper
    yr_upper = g*[1 ;X(k,:)'] + su*abs([1 ;X(k,:)']);
    y_pred_upper(k,1)=Wn*yr_upper;

    %lower
    yr_lower = g*[1 ;X(k,:)'] - sl*abs([1 ;X(k,:)']);
    y_pred_lower(k,1) = Wn*yr_lower;


end

end