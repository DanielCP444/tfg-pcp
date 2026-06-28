# pcp-tfg

Implementación experimental de verificadores PCP para el problema CLIQUE, desarrollada como parte del Trabajo de Fin de Grado en la Universidad Complutense de Madrid.

## Descripción

Este repositorio contiene el código fuente de los experimentos descritos en el TFG. Se implementan distintos verificadores PCP para el problema CLIQUE, junto con una construcción basada en la telaraña (*SpiderWeb*) y un algoritmo genético para la búsqueda de certificados.

## Estructura

```
├── constructions.py   # GraphInstance, SpiderWeb, PureGeneticAlgorithm
├── tests.py           # Clases de verificadores PCP
├── experiments.py     # Funciones de experimentación y visualización
└── requirements.txt
```

## Verificadores implementados

- `test_t2grupos_solovencedor` — verificador binario por grupos, solo vencedor
- `test_t2grupos_diferenciaexacta` — verificador con diferencia exacta entre particiones
- `test_t2grupos_bits_parciales` — verificador con ventana de bits parciales
- `test3_cantexacta` — verificador con muestreo probabilístico (cantidad exacta)
- `test3_margencant` — verificador con muestreo probabilístico (margen de bits)
- `test_4_reciprocidad_logn` — verificador de reciprocidad con subconjuntos de tamaño log n
- `test_4_reciprocidad_logn_margen` — variante con representación por margen

## Contexto académico

Este código acompaña al TFG *[título del TFG]*, dirigido por Ismael Rodríguez y Fernando Rubio, del Departamento de Sistemas Informáticos y Computación de la UCM.
