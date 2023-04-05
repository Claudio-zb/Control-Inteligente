i = 1;
ts = 0.001;
ttd = 20.0;
input = [];
output = [];
while i <= 18
    num = int2str(i);
    load(strcat("APRBS_signals\APRBS", num))
    input = [input; Salida(ttd/ts:end,3)];
    output = [output; Salida(ttd/ts:end,2)];
    i=1+i;
end