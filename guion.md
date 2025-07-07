# D1

[Buenos días/tardes], soy Ricardo Ruiz Fernández de Alba, y voy a mi presentar
mi TFG: "Imputación de Datos Faltantes de Variables Meteorológicas y de
Contaminación en Series Temporales Multivariantes IoT con Modelos Transformer"

# D2

La presentación de la estructura en los siguientes siete puntos, que voy a ir
desarrollando.  

# D3

En primer lugar, el contexto: Para estudiar la sostenibilidad ambiental de la
región de Poqueira, en la Alpujarra Granadina, se tomaron datos de tráfico en
las localidades de Pampaneira, Bubión y Capilera y de variables meteorológicas y
de contaminación cerca de Pampaneira mediante un Sistema IoT.

# D4

Al analizar los datos registrados cada hora por de 114 variables en dos periodos
de 2023: invierno y verano, se encontró un 28.4% de datos faltantes.

# D5

Por ejemplo, en esta gráfica de una de las 114 variables: PM10, que mide las partículas
suspendidas en el aire, los huecos en la serie temporal indican los datos faltantes.

# D6

Frente a este problema, nos plenamos la siguiente pregunta de investigación:

¿en qué medidas los diversos métodos de imputación pueden recuperar eficazmente
los valores faltantes? ¿Cómo se posicionan los modelos Transformers en este contexto?
Para responder a este interrogante, este trabajo plantea cuatro objetivos específicos:
1. Estudio del estado del arte.
2. Fundamentación informática-matemática del Transformer.
3. Elaboración de un Pipeline de Imputación y de un paquete de Python.
4. Evaluación comparativa del rendimiento.

# D7

Así pues, comenzamos por presentar la forma general del problema de Imputación.
Partimos de la Serie Temporal Original Parcialmente Observada X_original.
X_original = (X1, ..., Xn)T, donde n es el número de puntos temporales y Xi ∈ R^d

donde d es el número de variables. En este caso, d = 114.

Respecto a esta serie, se elimina deliberadamente un porcentaje de valores observados
produciendo X_missing. Este proceso genera una máscara binaria M, con 1s en la posiciones
eliminadas (puntos rojos) y 0s en caso contrario

Un método de imputación f se aplica a X_missing y M, generando la serie temporal completa
X̂c = f(X_missing, M). 

# D8

Ahora bien, ¿Por qué eliminamos datos? Porque esto permite comparar los 
valores imputados con su valor verdadero (ground truth) mediante distintas
métricos de error. Observamos que aparece la norma 1 y el Producto de Hadamar (punto a punto).

# D9

Continuamos con el estudio del estado del arte, estableciendo una taxonomía,
diferenciando métodos estadísticos, con información local, modelos regresivos,
autorregresivos, basados en matrices y de Deep Learning. 

# D10

Como métodos estacionarios, la media puede sustituir a los valores faltantes si se consideran
invariantes en el tiempo.

# D11

Estos dos métodos de información local estiman los valores faltantes
basándose en los puntos temporalmente adyacentes: En particular, Forward Fill
propaga el último valor observado hacia adelante, y backward rellena con el
siguiente valor observado hacia atrás.

# D12

En este caso, se tiene en cuenta ambos valores adyacentes, que se unen mediante
una línea recta, proporcionando los valores interpolados.


# D13

Como método basado en matrices, la Imputación Hankel transforma la serie
temporal en una matriz de tamaño (n-k+1) x (k+d), donde k parámetro: retardo y d
el número de variables. La imputación se resuelve minimizando la norma nuclear,
lo que permite obtener la serie temporal completa.  K = floor((n+1)/2)


# D14

La evolución del ML al DL progresó desde el Perceptrón básico hacia el
Perceptrón Multicapa (MLP) Y Redes Feed-Forward (FFN) para capturar relaciones
no lineales. Las limitaciones con datos secuenciales llevaron a la adopción de
Redes Recurrentes (RNN, LSTM), que aunque efectivas, presentaron problemas de
memoria y paralelización. Esta evolución culminó en las arquitecturas
Transformer. 

# D15

La arquitectura Transformer, introducida en "Attention Is All You Need" 
(Vaswani et al, 2017), revolucionó el procesamiento del lenguaje natural al
permitir paralelización completa durante el entrenamiento. Su diseño
encoder-decoder utiliza N capas apiladas. El encoder procesa simultáneamente
toda la secuencia de entrada generando una representación contextual que envía
al decoder.  Este focaliza su atención sobre tal representación generando una salida
de forma autorregresiva.

# D16

En principio, la paralelización de la que hablábamos no respetaría el orden de
la secuencia de embeddings si no fuera por el componente Positional encoding.
Este utiliza funciones trigonométrica para proporcionar 
información de orden y posición.


# D17

El núcleo del Transformer es su mecanismo de Atención, que toma entrada
las matrices de Query (Q), Keys (K) y Value (V), construidas a partir de los
embeddings con información posicional.

La atención calcula la similitud entre contextos y claves mediante el producto
escalar escalado y seguidamente la función Softmax obtiene los pesos de atención
que ponderan los valores.
Técnicamente se ejecutan múltiples mecanismos de atención en paralelo (uno por
cabeza), que se concatenan en los componentes Multihead-Attention.

# D18 

Tras cada componente de Multihead-Attention, se encuentra un Add & Norm, es
decir las Conexiones Residuales y la Normalización de Capas. Las primeras
facilitan el flujo de gradientes mientras que la segunda estabiliza el entrenamiento.

Por otro lado, las Red Neuronal Feed-Forward aplica transformaciones no lineales
mediante la función de activación ReLU entre dos capas lineales. Tras una capa
lineal final y Softmax se obtiene una distribución de probabilidad categórica sobre
los posibles valores que pueden tomar las posiciones faltantes en la serie
temporal.

Yn ~ Categórica(p) p = (p_1, ..., p_k)^T

K inicializada Pc: probabilidad de que el resultado sea la categoría Σ Pi = 1,
Pi ≥ 0 ∀i = 1, ..., K

# D19 

A diferencia del Transformer, saits emplea una arquitectura especializada en
imputación de series temporales con dos bloques DMSA (Diagonal Masked Self-Attention).

El primer bloque (color verde) se asemeja al encoder, mientras que el segundo
(color azul) al decoder. Entonces,  ¿cuáles serán las particularidades de este
modelo respecto a Transformer?

Por un lado, el enmascaramiento diagonal (color naranja oscuro) y un tercer bloque de
combinación ponderada de color marrón. 

# D20 

¿En qué consiste este enmascaramiento diagonal? 

pues que mientras que en Transformer cada punto puede atender a
todos los demás, y a sí mismo, en saits la máscara diagonal bloquea la
auto-atención, asignando -infinito a los elementos de la diagonal principal. 

Tras  aplicar Softmax estos valores diagonales se anulan, indicando que los valores
solo atienden al resto, no a sí mismos


# D21

El procesamiento matemático de saits se desarrolla en tres etapas:

El primer bloque DMSA produce tras su última capa lineal, la salida
X1. Una primera imputación a partr de esta se toma como entrada para el segundo bloque,
que tras una función de activación ReLu entre dos capas lineales, produce X2.

El tercer bloque (color marrón) efecta una combinación ponderada de X1 y X2,
dando lugar a X3.

Por último, utilizando la serie original X, la salida del bloque combinado X3 y la
máscara de datos faltantes M, obtenemos la serie temporal completa X̂c.

# D22

Este trabajo ha utilizado como herramienta la biblioteca PyPOTS, que
desarrollaron los autores de saits en 2023.

# D23

Ahora bien, ha sido necesario elaborar un paquete de Python llamado
pampaneira_imputation para integrar las implementaciones de PyPOTS con los
diversos métodos definidos en la taxonomía. 

Este paquete consta de 4 módulos dedicados a: la carga, preprocesamiento,
imputación y evaluación, junto con dos módulos para utilidad utilidades y configuraciones


# D24 

Utilizando los módulos del paquete desarrollado, se elaboró un Pipeline Flujo de
Trabajo, usando la Plataforma Jupiter cuyo Outline o esquema es el
correspondiente a la imagen.

Destacamos el proceso de ventaneo deslizante: Nwindows = Nsamples - 24 + 1, que convierte
la series temporales en tensores de dimensiones (Nwindows, 24, 114).

Este Pipeline se ha replicado para distintos porcentajes de eliminación.
En concreto, el patrón puntual que elimina puntos de forma aleatoria y el patrón 'subseq'
que elimina ráfagas de puntos.

# D25

Recordemos el ejemplo original de la variable PM10 parcialmente observada.

# D26

Tras el flujo de Trabajo, podemos observar las gráfica con los valores
imputados (línea verde de puntos) por saits: 
- la gráfica superior con el patrón puntual 'point' 
- la inferior con el patrón ráfaga 'subseq'. 


La comparación con el ground truth (puntos rojos) proporciona la tabla de errores.

# D27

Esta es una de las tablas de errores de evaluación + métricas. En concreto, la del período 1 con 10% de observar en patrón puntual.

# D28

Construyendo con los valores de la tabla, este gráfico 30, comprobamos que
- La interpolación lineal es la mejor de los métodos simples  
- Pero los modelos DL: Transformer & saits son superiores al resto de modelos. 
- saits mejora ligeramente a Transformer. 

# D29

A la pregunta de investigación planteada inicialmente: 

“¿En qué medida los métodos de imputación pueden recuperar eficazmente los
valores faltantes? ¿Y Transformer?”


- La eliminación artificial nos ha permitido comparar los valores imputados con
los originales (ground truth) mediante métricas de evaluación (RMSE, MRE, etc.).
- Transformer y saits son claramente superiores,  según estas métricas, a los
demás modelos.  
- saits mejora ligeramente a Transformer
- Las series temporales completas constituyen las bases para el estudio de la
sostenibilidad ambiental.