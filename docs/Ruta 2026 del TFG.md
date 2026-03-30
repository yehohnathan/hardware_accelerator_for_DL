> Las Etapas A y B corresponden a la preparación del dataset y de los pares FF; el entrenamiento en hardware inicia formalmente en la Etapa C, cuando el modelo pasa a poseer parámetros mutables, continúa con el forward local, el cálculo de la señal de aprendizaje y la actualización de pesos en las Etapas D–F, y culmina con el control de entrenamiento en la Etapa G. Las Etapas H–J corresponden a prueba, evaluación y experimentación del acelerador.
> 
> El Python actual de generación del `.bin`, dentro de `ff_bnn_stage/mnist/src`, **sí binariza** los píxeles y empaqueta `[label_onehot || imagen_binaria]` en 25 palabras de 32 bits.  

# Etapa A. Preprocesamiento externo y congelamiento del dataset hardware
---
**Estado:** completada.
**Objetivo:** transformar `train.csv` en un dataset hardware estable, compacto y sintetizable.

**Qué incluye:**
- lectura del CSV,
- validación del dataset,
- binarización de píxeles,
- codificación one-hot,
- concatenación FF,
- empaquetado en 25 `uint32`,
- verificación pack/unpack,
- guardado del `.bin` y archivos de depuración.

**Resultado formal de la etapa:**  
cada muestra queda como un vector fijo de 800 bits con el layout ya acordado.

**Archivos principales:**
- `main.py`
- `mnist_binarize_onehot_pack.py`
- `read_bin.py`

# Etapa B. Preparación hardware de pares FF
---
**Estado:** completada y validada.
**Objetivo:** construir el par supervisado Forward-Forward:
$$ 
x^{+} = (x, y_{true}), \qquad x^{-} = (x, y_{neg})  
$$

preservando la misma imagen y cambiando únicamente la etiqueta.

**Qué incluye:**
- lectura del binario,
- reconstrucción de la muestra de 800 bits,
- desempaquetado,
- decodificación de la etiqueta,
- generación de etiqueta negativa excluyente con LFSR,
- reempaquetado de positivo y negativo,
- validaciones en testbench.

**Archivos principales:** `git checkout preprocessing` 
- `forward_fw.cpp/.hpp`
- `train_tb.cpp`
- `debug_utils.cpp/.hpp`  
    `debug_utils` se conserva como soporte de inspección, no como núcleo del algoritmo.

# Etapa C. Definición del modelo entrenable en hardware
---
**Estado:** nueva.
**Objetivo:** introducir por primera vez el **estado entrenable del modelo** dentro de la FPGA.

Esta etapa es la que marca la diferencia entre:
- _preparar datos_,
- y _empezar a entrenar de verdad_.

**Qué debe definir:**
- forma de la capa base: `794 -> 64 (modifiable)`,
- paralelismo base: `8 (modifiable)` neuronas en paralelo,
- representación de parámetros,
- memoria de pesos y bias **de lectura/escritura**,
- formato numérico de activaciones, goodness y actualización. La recomendación de 64 neuronas totales y 8 en paralelo ya estaba fijada para la primera implementación.

**Decisión arquitectónica clave:**  
No conviene que el entrenamiento actualice directamente máscaras binarias “duras”.  
El notebook almacena **pesos latentes reales** y los binariza en el forward.

Por eso, en hardware la opción más coherente es:
- guardar **pesos latentes** en BRAM, con tipo fijo pequeño, por ejemplo `ap_fixed` o entero pequeño,
- derivar el signo en el forward para obtener la máscara `W_pos/W_neg`,
- mantener el `bias` como fijo o entero pequeño,
- aplicar `weight_clip` en la actualización.

# Etapa D. Forward local de una capa FF
---
**Estado:** nueva.
**Objetivo:** calcular el forward local de la capa usando la representación coherente con el proyecto:

$$  
z_j = \text{popcount}(x \land W^{+}_j) - \text{popcount}(x \land W^{-}_j) + b_j  
$$

$$  
h_j = \max(0, z_j)  
$$

$$  
g = \text{mean}(h^2)  
$$

Esta formulación ya fue definida como la versión correcta para entrada `0/1`, pesos binarios `{-1,+1}`, ReLU y goodness local.

# Etapa E. Cálculo de la señal de aprendizaje FF
---
**Estado:** nueva.
**Objetivo:** convertir `g_pos`, `g_neg` y `threshold` en una señal local de corrección.

En el paper de Hinton, el criterio es claro: cada capa debe aumentar la goodness de positivos y disminuir la de negativos, con referencia a un umbral.

El notebook implementa una pérdida FF suave basada en `softplus(threshold - g_pos)` y `softplus(g_neg - threshold)`.

**Recomendación hardware mínima funcional:**  
para una primera versión sintetizable, no arrancar con `softplus`.  
Conviene primero usar una señal más simple y barata, por ejemplo una versión por tramos:

$$ 
e_{pos} = \max(0, \theta - g_{pos})  
$$  
$$  
e_{neg} = \max(0, g_{neg} - \theta)  
$$

Eso preserva la lógica FF, evita exponenciales/logaritmos y reduce muchísimo costo en LUT/DSP.

# Etapa F. Actualización local de pesos y bias
---
> **Esta es la primera etapa de entrenamiento real.**

**Estado:** nueva.
**Objetivo:** modificar parámetros en hardware.

Mientras no exista una operación de escritura sobre pesos/bias, todavía no hay entrenamiento; solo hay forward y medición.

El notebook usa una actualización local simplificada que depende de `g_pos`, `g_neg`, `threshold`, `lr` y `weight_clip`.

**Propuesta realista para FPGA:**
- actualizar **pesos latentes** y no máscaras binarias directas,
- aplicar `clip` a los pesos latentes,
- recalcular el signo para el siguiente forward,
- actualizar bias con una regla local simple.

**Ventaja de esta decisión:**  
sigue la lógica del notebook y evita el problema de intentar “aprender” directamente sobre una máscara binaria rígida.

**Trade-off principal:**
- más BRAM para pesos latentes,
- más lógica de actualización,
- mayor latencia por muestra,
- pero por fin tienes entrenamiento FF auténtico.

# Etapa G. Control del entrenamiento hardware
---
**Estado:** nueva.
**Objetivo:** envolver las etapas D–F dentro de un flujo completo de entrenamiento.

**Qué debe hacer:**
- recorrer muestras o mini-batches,
- llamar a la preparación de pares,
- ejecutar forward positivo/negativo,
- calcular señal FF,
- actualizar parámetros,
- contar épocas,
- exponer métricas mínimas.

El notebook ya tiene esta lógica a nivel algorítmico cuando entrena por batches y registra `loss`, `goodness`, `val_acc`, `val_f1` y `goodness_gap`.

**Versión mínima recomendada en hardware:**
- primero **online por muestra** o mini-batch muy pequeño,
- luego escalar a mini-batch real si los recursos lo permiten.

# Etapa H. Inferencia multiclase
---
**Estado:** nueva.
**Objetivo:** una vez entrenado el modelo, clasificar una imagen probando todas las etiquetas posibles.

El notebook hace exactamente eso: prueba la imagen con cada label, calcula `total_goodness` y escoge la clase con mayor puntaje.

**Implementación hardware:**
- para una imagen, generar 10 variantes etiquetadas,
- ejecutar forward con cada una,
- acumular goodness total,
- hacer `argmax`.

**Observación importante:**  
esta etapa depende de que D ya esté estable.  
No conviene implementarla antes de que el entrenamiento local funcione.

# Etapa I. Evaluación del acelerador
---
**Estado:** nueva.
**Objetivo:** medir si el sistema entrena y clasifica correctamente, y cuánto cuesta hacerlo.

**Debe incluir:**
- accuracy,
- F1 macro,
- goodness gap,
- latencia por muestra,
- throughput,
- uso de LUT, FF, BRAM y DSP,
- consumo energético si ya estás en etapa de board.

Tu documento de tesis ya plantea como objetivos validar entrenamiento en BNN y luego evaluar rendimiento y eficiencia energética del acelerador.

**Dónde vive esta etapa:**  
principalmente en host/testbench y reportes, no dentro del kernel HLS.

# Etapa J. Experimentación y escalamiento
---
**Estado:** nueva.
**Objetivo:** explorar configuraciones y preparar la transición a versiones más grandes o a CNN.

**Variables naturales a barrer:**
- número total de neuronas,
- paralelismo 1/4/8/16,
- precisión de pesos latentes,
- `threshold`,
- `label_scale`,
- `learning_rate`,
- política de clip,
- una capa vs múltiples capas.

El notebook ya usa `threshold`, `label_scale`, `goodness_gap`, accuracy/F1 y experimentos de entrenamiento/evaluación como ejes de análisis.

# Lectura correcta de esta ruta
---
#### Bloque 1: **Preprocesar**
---
- **A**
- **B**
#### Bloque 2: **Entrenar**
---
- **C**
- **D**
- **E**
- **F**
- **G**
#### Bloque 3: **Probar / inferir**
---
- **H**
- **I**
- **J**