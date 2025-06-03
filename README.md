# TFG: Generación de Datos Sintéticos a partir de un Sistema IoT de Detección de Vehículos utilizando Modelos Transformer con Atención en Python

## Resumen

Este Trabajo de Fin de Grado se centra en la generación de datos sintéticos para sustituir valores faltantes, en series temporales multivariantes, en el contexto de sistemas IoT. El propósito es implementar y evaluar la eficacia de arquitecturas de aprendizaje profundo, especialmente modelos Transformer como SAITS, para la imputación de datos. La investigación se aplica a un caso de estudio de monitorización de tráfico y contaminación del proyecto Smart Poqueira, buscando restaurar la completitud de los datos.

La metodología se basa en una sólida **fundamentación teórica en matemáticas:** álgebra lineal,  métodos numéricos, análisis, probabilidad y estadística; y **aprendizaje profundo (deep learning)**, complementada con un estudio del estado del arte en imputación. En la parte práctica, se ha elaborado un [Jupyter Notebook](https://github.com/eigenric/TFG/blob/main/notebooks/Notebook_Imputation.ipynb) y un paquete de Python, **pampaneira\_imputation**, que integra diversos métodos de imputación, desde elementales hasta Transformer y SAITS. El flujo experimental incluye preprocesamiento y generación controlada de ausencias para una evaluación rigurosa.

Las tecnologías clave son Python, su ecosistema científico (Pandas, NumPy, Seaborn) y PyPOTS. El código, gestionado con Git, está disponible con licencia de código abierto en GitHub para fomentar la transparencia y reproducibilidad.

Los resultados experimentales, mediante métricas estándar de error, demuestran la superioridad de los modelos Transformer sobre los convencionales, logrando una reducción notable y consistente del error de imputación en los dos períodos de estudio analizados.

En conclusión, este estudio confirma que los modelos basados en autoatención, y SAITS en particular, son herramientas precisas y potentes para la imputación en series temporales complejas, generando datos más completos. Se cumplieron los objetivos de fundamentación, análisis, aplicación, desarrollo y evaluación. El trabajo aporta una solución contrastada y sienta bases para futuras investigaciones en IoT, orientadas a la gestión ambiental y la sostenibilidad.



## Historias de Usuario y Milestones

En los siguientes documentos, se detallan las historias de usuario y milestones que se han definido para el
desarrollo del proyecto:

- [Historias de Usuario](docs/user-stories.md)
- [Milestones](docs/milestones.md)

# License

[GPLv3](LICENSE) para el código y [CC-BY-SA](https://creativecommons.org/licenses/by-sa/4.0/) para las memorias.
