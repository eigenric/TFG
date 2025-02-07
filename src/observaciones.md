# Miércoles 20/11/2024

Debido al largo contenido del primer Notebook, que incluye un analisis exploratorio de los
datos, y teniendo en cuenta que hay que restringirse a la región de Pampaneira para realizar el 
el objetivo principal de generar datos mediante modelos transformer, se ha creado un Notebook
Pipeline2.

Tenemos dos dataset, df_orig y df_int. df_int es un subconjunto de df_orig donde tienen todas
las columnas de este más las columnas de contaminación.
Hemos modificado la columna date para que tengan el mismo nombre (en lugar de Date en el dset 
intersección) y el mismo tipo.
Esto permite realizar el merge.

## Interpolación

Interpolación: https://medium.com/@abhishekjainindore24/handling-missing-data-in-pandas-part-2-8d06c4075bb6

La idea es interpolar los valores faltantes mediante un método de spline. Para ello debe haber
suficiente valores no nulos. La columna eBC_tot para la interseccion en pampaneira tiene todos 
los valores nulos.

¿Transformaciones logarítmicas?

¿Es siempre eBC_tot suma de eBC_ff y eBC_bb?

## Normalizción

Normalización: https://medium.com/@prasadmahamulkar/feature-scaling-using-standardization-normalization-and-robust-scaling-cb333f943f70

Usaremos RobustScaler.
Duda, ¿Es normal que deje valores negativos?

## Aumentar la muestra con Transformers

Se optó en primer lugar por transformar la fecha a timestamp
pero ofrecía errores muy altos de perdida.
Por tanto se entrena sin la columna fecha y luego se generan
fechas desde 2023-06-27 01:00:00+00:00 hasta 200 más con periodo de 1 hora.

- Mediante un GAN
- Mediante Transformer

## Realizar el merge

## Matriz de correlación entre variables de contaminación y tráfico

Ver eBC_tot y eBC_ff, eBC_bb

## Entrenar modelo de regrresión o clasificación un subconjunto de datos

Pendiente

Entrenar un modelo de regresión o clasificación para las variables elegida 