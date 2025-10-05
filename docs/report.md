<h1>Examen Final - Aprendizaje profundo</h1>  

<h4><u>Autores:</u></h4> 

   <p>1. Pablo Gonzalez </p> 
    <p>2. Araceli Sanchez</p>

<h4><u>Fecha:</u> 05-10-25</h4>  

<h2>Resumen ejecutivo RAG</h2>  

<p>Desarrollamos un sistema RAG (Retrieval-Augmented Generation) para consultar sinopsis de películas a partir de documentos propios, para este caso realizamos la carga de dos documentos: superheroes.txt y romance.txt.

Flujo de trabajo: se cargan de los documentos, luego pasan por el proceso de chunking, se vectorizan con un modelo de embedding (en este caso elegimos uno de HuggingFaceEmbeddings), posterior al embedding, se indexan a un vector store (FAISS en este caso) y luego ante una pregunta del usuario el sistema recupera los k (3 para este trabajo) mas relevantes con lo cual arma un contexto que el LLM (via langchain + ollama) utiliza para responder.

**Alcance actual:** carga de documentos, búsqueda semántica y respuesta contextual.  
**Stack técnico:** LangChain, Ollama, HuggingFaceEmbeddings, FAISS.  
**Próximos pasos:** sumar más géneros, mejorar evaluación de calidad de respuestas y persistir/optimizar el índice para colecciones grandes con por ejemplo PGVector.</p>