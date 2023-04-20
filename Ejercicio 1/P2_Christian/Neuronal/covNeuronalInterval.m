function [y_pred_upper, y_pred, y_pred_lower] = covNeuronalInterval(net, X, Z, sigma, alpha)
%Parametros de normalizacion
ymax = net.input_ymax;
ymin = net.input_ymin;
xmax = net.input_xmax;
xmin = net.input_xmin;

ymax_out = net.output_ymax;
ymin_out = net.output_ymin;
xmax_out = net.output_xmax;
xmin_out = net.output_xmin;


input_preprocessed = (ymax-ymin) * (X-xmin) ./ (xmax-xmin) + ymin;
% Pass it through the ANN matrix multiplication
y1 = tanh(net.IW * input_preprocessed + net.b1);

y2 = net.LW * y1 + net.b2;

res = (y2-ymin_out) .* (xmax_out-xmin_out) /(ymax_out-ymin_out) + xmin_out;
res = res'; %Nx1

sigmak = zeros(length(res), 1);
for k=1:length(res)
    zk = y1(:, k);
    sigmak(k) = sigma*sqrt((1+zk'/(Z*Z')*zk));
end

y_pred = res;
y_pred_upper = y_pred + alpha*sigmak;
y_pred_lower = y_pred - alpha*sigmak;

end