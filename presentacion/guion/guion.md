Buenos días, soy Ricardo Ruiz Fernández de Alba, y voy a presentar
mi TFG: "Imputación de Datos Faltantes de Variables Meteorológicas y de
Contaminación en Series Temporales Multivariantes IoT con Modelos Transformer"

La presentación se estructura en los siguientes siete puntos, que voy a ir
desarrollando.  

En primer lugar, el contexto: Para estudiar la sostenibilidad ambiental de la
región de Poqueira, en la Alpujarra Granadina, se tomaron datos de tráfico en
las localidades de Pampaneira, Bubión y Capileira y de variables meteorológicas y
de contaminación cerca de Pampaneira mediante un Sistema IoT.

Al analizar los datos registrados cada hora de 114 variables en dos periodos
de 2023: invierno y verano, se encontró un 28.4% de datos faltantes.

Por ejemplo, esta gráfica de una de las 114 variables: PM10, que mide las partículas suspendidas en el aire, representa una serie temporal univariante parcialmente observada.


Frente a este problema, nos planteamos la siguiente pregunta de investigación:
¿En qué medida los diversos métodos de imputación pueden recuperar eficazmente los valores faltantes? ¿Cómo se posicionan los modelos Transformers en este contexto?
Para responder a este interrogante, este trabajo plantea como objetivo principal el estudio de los modelos transformer para imputar series temporales.

Y como objetivos específicos:
1. Estudio del estado del arte.
2. Fundamentación informática-matemática del Transformer.
3. Elaboración de un Pipeline de Imputación y de un paquete de Python.
4. Evaluación comparativa del rendimiento.

Así pues, comenzamos por presentar la forma general del problema de Imputación.
Partimos de la Serie Temporal Original Parcialmente Observada X original.
De esta serie, se elimina deliberadamente un porcentaje de valores observados produciendo X missing. Este proceso genera una máscara binaria M, con unos en la posiciones eliminadas (puntos rojos) y ceros en caso contrario

Un método de imputación f se aplica a X missing y M, generando la serie temporal completa

Ahora bien, ¿Por qué eliminamos datos? Porque esto permite comparar los 
valores imputados con su valor verdadero (ground truth) mediante distintas
métricas de error: error absoluto medio, error cuadrático medio, y error relativo medio.
Observamos que aparece la norma 1 y el Producto de Hadamard (punto a punto).

Continuamos con el estudio del estado del arte, estableciendo una taxonomía,
que diferencia entre métodos estacionarios, métodos con información local, basados en matrices y de Deep Learning entre los que se encuentran los modelos Transformer.

Como métodos estacionarios, la media y la mediana pueden sustituir a los valores faltantes si se consideran invariantes en el tiempo.

Por otro lado, los métodos de información local estiman los valores faltantes
basándose en los puntos adyacentes: En particular, Forward Fill
propaga el último valor observado hacia adelante, y Backward Fill rellena con el siguiente valor observado hacia atrás.

En el caso de la interpolación lineal, se tienen en cuenta ambos valores adyacentes, que se unen mediante una línea recta, completando los valores.

Como método basado en matrices, la Imputación Hankel transforma la serie
temporal en una matriz de tamaño (n - k + 1) x (k+d), donde n: número de filas, k parámetro retardo que por defecto toma la parte entera de n+1 medios  y d: número de variables. 
La imputación se resuelve minimizando la norma nuclear, lo que permite obtener la serie temporal completa.  


La evolución del ML al DL progresó desde el Perceptrón básico hacia el
Perceptrón Multicapa (MLP) Y Redes Feed-Forward (FFN) para capturar relaciones
no lineales. Las limitaciones con datos secuenciales llevaron a la adopción de
Redes Recurrentes (RNN, LSTM), que aunque efectivas, presentaron problemas de
memoria y paralelización. Esta evolución culminó en las arquitecturas
Transformer. 

La arquitectura Transformer, introducida en "Attention Is All You Need" 
(Vaswani et al, 2017), revolucionó el procesamiento del lenguaje natural al
permitir paralelización completa durante el entrenamiento. Su diseño
encoder-decoder utiliza N capas apiladas. El encoder procesa simultáneamente
toda la secuencia de entrada generando una representación contextual que envía
al decoder.  Este focaliza su atención sobre tal representación generando una salida
de forma autorregresiva.

En principio, la paralelización de la que hablábamos no respetaría el orden de
la secuencia de embeddings si no fuera por el componente Positional encoding.
Este utiliza funciones trigonométricas para proporcionar  información de orden y posición.

El núcleo del Transformer es su mecanismo de Atención, que toma como entrada
las matrices de Consulta (Q), Claves (K) y Valores (V), construidas a partir de los embeddings con información posicional.
La atención calcula la similitud entre Q y K mediante el producto
escalar escalado y seguidamente la función Softmax obtiene los pesos de atención que ponderan los valores.
Técnicamente se ejecutan múltiples mecanismos de atención en paralelo (uno por
cabeza), que se concatenan en los componentes Multihead-Attention.

Tras cada componente de Multihead-Attention, se encuentra un Add & Norm, es
decir las Conexiones Residuales y la Normalización de Capas. Las primeras
facilitan el flujo de gradientes mientras que la segunda estabiliza el entrenamiento.

Por otro lado, la Red Neuronal Feed-Forward aplica transformaciones no lineales
mediante la función de activación ReLU entre dos capas lineales. Tras una capa
lineal final y Softmax se obtiene una distribución de probabilidad categórica sobre
los posibles valores que pueden tomar las posiciones faltantes en la serie
temporal.

A diferencia del Transformer, SAITS emplea una arquitectura especializada en
imputación de series temporales con dos bloques DMSA (Diagonal Masked Self-Attention) y un tercer bloque.

El primer bloque (en color verde) se asemeja al encoder, mientras que el segundo (en color azul) al decoder. Entonces,  ¿cuáles son las particularidades de este modelo respecto a Transformer?

Por un lado, el enmascaramiento diagonal (en color naranja oscuro) y un tercer bloque de combinación ponderada (en color marrón claro) 

¿En qué consiste este enmascaramiento diagonal? 

Pues que mientras que en Transformer cada punto puede atender a
todos los demás, y a sí mismo, en SAITS la máscara diagonal bloquea la
auto-atención, asignando menos infinito a los elementos de la diagonal principal. 
Tras aplicar Softmax los pesos diagonales se anulan, lo que significa que, al multiplicar por la matriz de valores, V, cada punto sólo atiende a los demás.


El procesamiento matemático de SAITS se desarrolla en tres etapas:

El primer bloque DMSA produce tras su última capa lineal, la salida
X1. Una primera imputación a partir de esta se toma como entrada para el segundo bloque,
que tras una función de activación ReLu entre dos capas lineales, produce X2.

El tercer bloque efectúa una combinación ponderada de X1 y X2,
dando lugar a X3.

Por último, utilizando la serie original X, la salida del bloque combinado X3 y la
máscara de datos faltantes M, obtenemos la serie temporal completa X gorro.

Este trabajo ha utilizado como herramienta la biblioteca PyPOTS, que
desarrollaron los autores de SAITS en 2023.

Ahora bien, ha sido necesario elaborar un paquete de Python llamado
pampaneira_imputation para integrar las implementaciones de PyPOTS con los
diversos métodos definidos en la taxonomía. 

Este paquete consta de 4 módulos dedicados a: la carga, preprocesamiento,
imputación y evaluación, junto con dos módulos para utilidades y configuraciones

Este paquete figura en el repositorio indicado arrriba con licencia
de código abierto, para hacerlo reproducible.


Utilizando los módulos del paquete desarrollado, se elaboró un Pipeline o Flujo de
Trabajo, usando la Plataforma Jupiter cuyo Outline o esquema es el
correspondiente a la imagen.

Dentro del preprocesamiento, destacamos que se realiza un escalado previo usando las medias y desviaciones tipicas de las características de entrenamiento que se aplican posteriormente a validación y test.

Por otro lado, la representación de la serie temporal se cambia para adaptarla al modelo de Deep Learning. En concreto, mediante un proceso de ventaneo deslizante: que las convierte en arrays tridimensionales (número de ventanas x tamaño de ventana x número de características). Se escoge un tamaño de 24 horas por ventana lo que permite seguir los patrones diarios de las variables.   N_windows = N_rows – 24 + 1

Recordemos el ejemplo original de la variable PM10 parcialmente observada.

Tras el flujo de Trabajo, podemos observar las gráficas con los valores
imputados (línea verde de puntos) por SAITS: 

La gráfica superior con el patrón puntual 'point’: que elimina puntos de forma aleatoria.
La inferior con el patrón 'subseq’: que elimina ráfagas de puntos.
La comparación con el ground truth (puntos rojos) proporciona la tabla de errores.

Esta es una de las catorce tablas de errores de evaluación que hay en la memoria. En concreto, la del período 1 con 10% de ausencia y patrón puntual.

Para visualizar mejor los resultados, se construyó este gráfico 3D.
Observándose que:
La interpolación lineal es la mejor de los métodos simples.
Y que os modelos DL: Transformer & SAITS son superiores al resto de modelos. 
De hecho, SAITS mejora ligeramente a Transformer. 

Se replicó el Pipeline con variaciones en los porcentajes y patrones de ausencia. 
Y se observó que, en la mayoría de los casos,  conforme aumenta el porcentaje aumenta el error.
Por otro lado, la ausencia con patrón ráfaga da peor resultado que el correspondiente puntual.


A la pregunta de investigación planteada inicialmente: 

“¿En qué medida los métodos de imputación pueden recuperar eficazmente los
valores faltantes? ¿Y Transformer?”

Podemos responder:

La eliminación artificial nos ha permitido comparar los valores imputados con
los originales (ground truth) mediante métricas de evaluación.
Transformer y SAITS son claramente superiores,  según estas métricas, a los
demás modelos.  
SAITS mejora ligeramente a Transformer.
Aunque la ausencia por ráfaga da mayor error que la puntual, modela los fallos de los sensores de forma más realista.

Así, estas series temporales, ya completadas, pueden utilizarse para mejorar el estudio de la sostenibilidad ambiental, de acuerdo con los Objetivos de Desarrollo Sostenible.

Gracias por su Attention.
