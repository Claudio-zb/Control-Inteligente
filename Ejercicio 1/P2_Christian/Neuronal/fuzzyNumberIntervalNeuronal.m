function [y_pred_upper, y_pred, y_pred_lower] = fuzzyNumberIntervalNeuronal(X, net, LWl, bl, LWu, bu)
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
y2l = LWl * y1 + bl;
y2u = LWu * y1 + bu;

res = (y2-ymin_out) .* (xmax_out-xmin_out) /(ymax_out-ymin_out) + xmin_out;
resl = (y2l-ymin_out) .* (xmax_out-xmin_out) /(ymax_out-ymin_out) + xmin_out;
resu = (y2u-ymin_out) .* (xmax_out-xmin_out) /(ymax_out-ymin_out) + xmin_out;

y_pred = res';
y_pred_lower = resl';
y_pred_upper = resu';

end