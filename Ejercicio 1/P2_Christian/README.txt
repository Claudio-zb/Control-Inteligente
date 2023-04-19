Ejecutar todos los scripts desde esta carpeeta P2_Christian

Generacion de datos:

Utilizar respuestaEscalon.m para configurar parametros de APRBS.
Usar dataGeneration.m con los parametros deseados para generar los datos y 
las particiones. En este ultimo caso, se deben configurar los porcentajes
y cantidad de regresores maximos a ocupar por los modelos (esto ultimo es
optimizado por cada uno de ellos).
Al final de la ejecución, se crea DatosP2.mat y split.mat dentro de Data.


Modelo difuso:

Ejecutar modeloDifuso.m
Aquí se debe configurar el numero de clusters y regresores de acuerdo a
los experimentos que se realicen. Al finalizar, se guarda el modelo en
Fuzzy/modelo_difuso.mat.
Para probar sobre validacion, usar predecirDifuso.m. Aquí se debe configurar
la cantidad de regresores de y que tienen los datos y los regresores que
finalmente se utilizan. Además, se configura el número de pasos a los que
predecir. Se mostrará la predicción y las metricas.


Modelo neuronal:

Ejecutar modeloNeuronal.m
Aquí se debe configurar el numero de neuronas de la capa oculta. Para ello
se entrena con distinta cantidad y se tiene el error alcanzado en test
para cada uno, en la variable errores.
Luego de esto, se deben escoger los regresores. Se puede ir sacando uno a uno
iterativamente.
Finalmente, se guarda el mejor modelo en Neuronal/modelo_neuronal.mat.
Para probar sobre validación, usar predecirNeuronal.m. Aquí se debe configurar
la cantidad de regresores de y que tienen los datos y los regresores que
finalmente se utilizan. Además, se configura el número de pasos a los que
predecir. Se mostrará la predicción y las metricas.