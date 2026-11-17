# Perceptrón para aprobar una experiencia curricular

Aplicación educativa en Python y Streamlit. Implementa el perceptrón manualmente, con las fórmulas y el factor de aprendizaje fijados en el ejercicio.

## Archivos

- `app.py`: aplicación Streamlit, entrenamiento, historial y evaluación de estudiantes.
- `desarrollo_matematico.md`: desarrollo patrón por patrón, incluidos los parámetros anteriores y posteriores a cada operación.
- `requirements.txt`: dependencias necesarias.

## Requisitos

Python 3.10 o posterior.

## Instalación y ejecución

Desde esta carpeta, crea y activa un entorno virtual e instala las dependencias:

```bash
python -m venv .venv
```

En Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

En macOS o Linux:

```bash
source .venv/bin/activate
```

Instala las dependencias y ejecuta la aplicación:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Streamlit abrirá la aplicación en el navegador. La evaluación acepta promedios entre 0 y 20; cada nota se convierte en 1 si es al menos 14 y en 0 si es menor que 14.

## Parámetros del ejercicio

- Entradas iniciales: `w1=0.5`, `w2=0.4`, `w3=0.5` (el valor de `w3` fue confirmado por el usuario).
- Bias inicial: `b=1.5`.
- Factor de aprendizaje fijo: `fa=0.5`.
- Activación: salida 1 cuando `NET >= 0`; salida 0 cuando `NET < 0`.
- Orden de patrones: `000`, `001`, `010`, `011`, `100`, `101`, `110`, `111`.
- Convergencia: una época completa sin errores de clasificación.

La salida deseada se asigna a los patrones de entrenamiento según la regla de dos unidades aprobadas. La predicción de un estudiante se obtiene aplicando NET y la función de activación con los pesos aprendidos.
