function [Z, sigma] = getCovParamsNeuronal(net, X, Y)
%Parametros de normalizacion
ymax = net.input_ymax;
ymin = net.input_ymin;
xmax = net.input_xmax;
xmin = net.input_xmin;

ymax_out = net.output_ymax;
ymin_out = net.output_ymin;
xmax_out = net.output_xmax;
xmin_out = net.output_xmin;

%Numero de datos y regresores
[Nd,n]=size(X);

input_preprocessed = (ymax-ymin) * (X-xmin) ./ (xmax-xmin) + ymin;
% Pass it through the ANN matrix multiplication
y1 = tanh(net.IW * input_preprocessed + net.b1);

y2 = net.LW * y1 + net.b2;

res = (y2-ymin_out) .* (xmax_out-xmin_out) /(ymax_out-ymin_out) + xmin_out;
res = res';

sigma = mean((Y-res).^2);
Z = y1;
end