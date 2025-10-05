<h1>Proyecto Final - Aprendizaje profundo</h1>  

<h4><u>Autores:</u></h4> 

   <p>1. Pablo Gonzalez </p> 
    <p>2. Araceli Sanchez</p>

<h4><u>Fecha:</u> 05-10-25</h4>  

<h2>Resumen ejecutivo CNN</h2>  

<p>Desarrollamos un modelo de clasificacion de imagenes del dataset cifar10 utilizando una arquitectura de red neuronal convolucional (CNN): compuesta por capas Conv2D, MaxPooling2D, Flatten y Dense.  

El modelo fue entrenado durante 35 epocas y evaluado utilizando TensorFlow y Keras, logrando una:  
- Precision en entrenamiento: 0.825  
- Precision en validacion: 0.79  
- Perdida en entrenamiento: 0.49  
- Perdida en validacion: 0.63  

El modelo logra una buena capacidad de aprendizaje y generalización, aunque existe una pequeña diferencia entre entrenamiento y validación que podría indicar margen para mejorar con técnicas de regularización o ajuste de hiperparámetros.  

**Alcance actual:** clasificación de imágenes en 10 categorías del dataset cifar10.  
**Stack técnico:** Tensorflow, Keras, Python.    
**Próximos pasos:** Aplicar regularización como Dropout, ajustar el learning rate para mejorar la precisión en validación.</p>  

<h2>Resumen ejecutivo GAN</h2>  

<p>Desarrollamos un modelo de generacion de imagenes utilizando una arquitectura de red neuronal generativa adversarial (GAN) utilizando el dataset de cifar10 para entrenamiento: compuesta por un generador y un discriminador y un modelo GAN que los une.  
El discriminador utiliza capas Conv2D, LeakyReLU, Dropout y Dense, mientras que el generador utiliza capas Dense, LeakyReLU, Reshape y Conv2DTranspose.

El combinado utiliza un optimizador Adam con tasa de aprendizaje 0.0002 y beta_1 0.5.  


El modelo fue entrenado durante 90 epocas debido a la limitacion de hardware, alguna de las imagenes generadas se encuentran en la carpeta results/gan.    

Si bien las imagenes generadas aun no son claras, se pueden observar las primeras estructuras que el modelo va aprendiendo. Con mas tiempo de entrenamiento y ajustes en los hiperparametros se podrian obtener mejores resultados. El generador toma un vector de ruido como entrada y genera una imagen, mientras que el discriminador toma una imagen como entrada y determina si es real o generada. Durante el entrenamiento, el generador intenta engañar al discriminador para que clasifique sus imágenes como reales, mientras que el discriminador intenta mejorar su capacidad para distinguir entre imágenes reales y generadas. Este proceso de competencia entre los dos modelos es lo que permite al generador aprender a crear imágenes cada vez más realistas.  

**Alcance actual:** generacion de imágenes similares a las del dataset cifar10.  
**Stack técnico:** Tensorflow, Keras, Python.    
**Próximos pasos:** Aplicar regularizaciones, tambien entrenar el generador y el discriminador por mas epocas para poder visualizar los resultados.</p> 

<h2>Resumen ejecutivo RAG</h2>  

<p>Desarrollamos un sistema RAG (Retrieval-Augmented Generation) para consultar sinopsis de películas a partir de documentos propios, para este caso realizamos la carga de dos documentos: superheroes.txt y romance.txt.

Flujo de trabajo: se cargan de los documentos, luego pasan por el proceso de chunking, se vectorizan con un modelo de embedding (en este caso elegimos uno de HuggingFaceEmbeddings), posterior al embedding, se indexan a un vector store (FAISS en este caso) y luego ante una pregunta del usuario el sistema recupera los k (3 para este trabajo) mas relevantes con lo cual arma un contexto que el LLM (via langchain + ollama) utiliza para responder.

**Alcance actual:** carga de documentos, búsqueda semántica y respuesta contextual.  
**Stack técnico:** LangChain, Ollama, HuggingFaceEmbeddings, FAISS.  
**Próximos pasos:** sumar más géneros, mejorar evaluación de calidad de respuestas y persistir/optimizar el índice para colecciones grandes con por ejemplo PGVector.</p>

<h2>Resumen ejecutivo LSTM</h2>  

<p>Desarrollamos un modelo de prediccion de fallas de maquinas a partir de datos generados por sensores (serie temporal) utilizando una arquitectura de red neuronal LSTM (Long Short-Term Memory): compuesta por capas LSTM, Dropout y Dense.  
El modelo fue entrenado durante 4 epocas debido al early stopping implementado logrando un:

Reporte de entrenamiento:

                precision    recall  f1-score   support

             0      0.985     0.985     0.985     17431
             1       0.92     0.917     0.918      3100
    
        accuracy     0.975                        20531
     macro avg       0.953                        20531  

Reporte de validacion(test):

                precision    recall  f1-score   support

             0       0.99      0.99      0.99     12664
             1       0.86      0.71      0.78       332
    
        accuracy     0.988                        12996
     macro avg       0.989                        12996


Como vemos el rendimienot del modelo es bueno en entrenamiento y validacion, aunque existe una diferencia en la clase 1 (fallas) que podria mejorarse con tecnicas de oversampling o ajuste de hiperparametros.  
Debido a que la cantidad de datos de las clases con falla es mucho menor que la de las clases sin falla, el modelo tiende a predecir mas o tener mas sesgo hacia la clase sin falla. Por eso la precicion y recall de la clase 1 es menor.  
**Alcance actual:** prediccion de fallas en maquinas a partir de datos de sensores.
**Stack técnico:** Tensorflow, Keras, Python.    
**Próximos pasos:** Realizar un balanceo de clases con tecnicas de oversampling, ajustar el learning rate y probar con regularizaciones de manera a obtener mas epocas de entrenamiento.
</p> 

<h2>Resumen ejecutivo Transformer</h2>  

<p>Desarrollamos un modelo de clasificacion de texto utilizando una arquitectura de Transformer compuesta por un Possitional Embbeding, y un Decoder, asi como sus parametros de atencion, y longitudes de sequencias previamente definidas en la carga del dataset de IMDB esto nos ayuda a clasificar las reseñas de peliculas en positivas y negativas.  
El modelo fue entrenado por 3 epocas debido a la limitacion de hardware aun asi se lograron buenos resultados con los datos de prueba:  


               precision    recall  f1-score   support
    Negativo       0.85      0.88      0.87     12500
    Positivo       0.88      0.84      0.86     12500

    accuracy                           0.86     25000
    macro avg       0.86      0.86     0.86     25000
    weighted avg    0.86      0.86     0.86     25000

Segun la grafica reportada en la notebook de precision y perdida el modelo puede mejorar aun mas con mas epocas de entrenamiento, por lo que se recomienda aumentar la cantidad de epocas y ajustar el learning rate para mejorar la precision en validacion. Tambien se podria probar con regularizaciones a parte del dropout ya implementado.  

**Alcance actual:** clasificacion de reseñas de texto en positivas y negativas del dataset IMDB.    
**Stack técnico:** Tensorflow, Keras, Python.    
**Próximos pasos:** Aplicar regularizaciones, tambien entrenar por mas epocas y continuar con el fine tuning del numero de cabezas de atencion por ejemplo </p> 