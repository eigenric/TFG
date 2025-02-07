**VARIABLES DE CONTAMINACIÓN**

CO  
Concentraciones medias horarias de monóxido de carbono medidas con el analizador de monóxido de carbono (mod. 48i, Thermo Scientific). Unidades: miligramos por metro cúbico (mg/m3).

**NO y NO2 son trazadores de tráfico diesel.**

NO  
Concentraciones medias horarias de monóxido de nitrógeno medidas con el analizador de NO/NO2/NOX (mod. 42i, Thermo Scientific). Unidades: microgramos por metro cúbico (µg/m3).

NO2  
Concentraciones medias horarias de dióxido de nitrógeno medidas con el analizador de NO/NO2/NOX (mod. 42i, Thermo Scientific). Unidades: microgramos por metro cúbico (µg/m3).

O3  
Concentraciones medias horarias de ozono troposférico medidas con el analizador de ozono (mod. 49i, Thermo Scientific). Unidades: microgramos por metro cúbico (µg/m3).

SO2  
Concentraciones medias horarias de dióxido de azufre medidas con el analizador de dióxido de azufre (mod. 43i, Thermo Scientific). Unidades: microgramos por metro cúbico (µg/m3).

PM10  
Concentraciones medias horarias de material particulado en suspensión PM10 obtenidas con el Monitor TEOM 1405 de partículas PM10 (Thermo Scientific). Unidades: microgramos por metro cúbico (µg/m3).

eBC\_tot  
Concentraciones medias horarias de black carbon u hollín total obtenidas con fotómetro de absorción (Aethalometer AE31). Unidades: microgramos por metro cúbico (µg/m3).

**eBC\_ff trazador primario de tráfico de coches**

eBC\_ff  
Concentraciones medias horarias de black carbon atribuíble a la combustión de combustibles fósiles. Se obtienen como resultado de aplicar el modelo del etalómetro (Sandradewi et al., 2008). Unidades: microgramos por metro cúbico (µg/m3).

eBC\_bb  
Concentraciones medias horarias de black carbon atribuíble a la quema de biomasa. Se obtienen como resultado de aplicar el modelo del etalómetro (Sandradewi et al., 2008). Unidades: microgramos por metro cúbico (µg/m3).

**Temperaturas altas: situaciones que en invierno son más anticiclónicas (se estanca). La temperatura es para saber si viene un Frente (en calimas sube y habrá muchas partículas). TB puede ser que haya mucha calefacción y más partículas. ALTA temperatura \-\> peor porque se estanca.**

TEMP  
Temperatura del aire (promedio horario) obtenida con el sensor de temperatura de la torre meteorológica (Transmisor meteorológico YOUNG 92000 ResponseONE). Unidades: grados Celsius.

**más Humedad \-\> más estancamiento. La humedad depende de la lluvia.**

RH  
Humedad relativa del aire (promedio horario) obtenida con el sensor de humedad de la torre meteorológica (Transmisor meteorológico YOUNG 92000 ResponseONE). Unidades: porcentaje (%).

**De forma combinada: presión atmosférica con velocidad viento. Si hay presión alta y velocidad baja (como CAP) se trata de un episodio de estancamiento. Los contaminantes no se dispersan.**

WS  
Velocidad media del viento (promedio horario) medida con el anemómetro de la torre meteorológica. Unidades: metros por segundo (m/s).  
WD  
Dirección media del viento (promedio horario) obtenida en la torre meteorológica. 0° corresponde al Norte. Unidades: grados (°).

PRES  
Presión atmosférica a nivel de estación (promedio horario) medida con la estación de la torre meteorológica. Unidades: hectopascales (hPa).

**VARIABLE CAMIÓN:**

pos\_truck \= \[identificador de la cámara más cercana al camión de meteorología\]

**VARIABLES VEHÍCULOS:**

Están formadas por vehicles\_ID\_camera\_IN/OUT

IN: entrada  
OUT: salida

ID\_camara \= \[PAM\_1, PAM\_2, BUB, CAP\]

Cuando ponga nan, significa que esos vehículos contabilizados no tienen la variable indicada.

**ZONAS GEOGRÁFICAS:**

Zona\_Catalunia\_y\_Otras \= \['Catalunya', 'La Rioja', 'Aragón', 'Aragon', 'Illes Balears', 'Euskadi', 'Navarra \- Nafarroa', 'Galicia', 'Asturias / Asturies', 'Cantabria', 'Castilla y León', 'Castilla y Leon'\]:

Zona\_Comunidad\_de\_Madrid \= \[Comunidad de Madrid'\]

Zona\_Andalucia\_no\_GR \= \[Provincias de Andalucía sin Granada\]

Zona\_Granada \= \[Granada\]

Zona\_Extremadura\_y\_Otras \= \['Extremadura', 'Castilla-La Mancha', 'Región de Murcia', 'Murcia', 'Region de Murcia', 'Comunitat Valenciana', 'Comunidad Valenciana', 'Canarias', 'Ceuta', 'Melilla'\]

**INTERVALOS CONTAMINACIÓN:**

co2 del vehículo concreto

   elif co2 \<= 100:  
        return '0-100'  
    elif 100 \< co2 \<= 200:  
        return '101-200'  
    elif 200 \< co2 \<= 300:  
        return '201-300'  
    else:  
        return '+300'

**NÚMERO DE ASIENTOS:**

5\_seats: 5 asientos  
\+5\_seats  
\-5\_seats:

**ETIQUETAS ECO:**

Sin Etiqueta, C, B, ECO, 0 (las que proporciona DGT)

Obtenidas a partir de web scrapping de la página oficial.

**PERIODO DE DATOS:**

**CAP: Colegio Público Rural. En el patio \- Barranco de Poqueira (Detrás del Ayto)**  
**PAM: Dentro del cementerio municipal, en el patio de acceso, pasado el pueblo**

**Camión en CAPILEIRA: 16/11/2022-15/01/2023**  
**Camión en Pampaneira 2:17/01/2023-14/03/2023**  
**Camión en Pampaneira 2: 06/06/2023-27/06/2023**  
