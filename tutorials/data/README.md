# Datos de los tutoriales

## `sbwce_slim.txt`

Versión reducida de los embeddings **SBWCE** (*Spanish Billion Words Corpus and Embeddings*), usada en el tutorial 3 (word embeddings).

- **Contenido:** las 10.000 palabras más frecuentes del archivo original, con sus vectores de 300 dimensiones, en formato de texto de word2vec. Los vectores no se modificaron: solo se conservaron las primeras 10.000 líneas (el original viene ordenado por frecuencia) y se ajustó el encabezado (`10000 300`).
- **Original:** 1.000.653 palabras, entrenadas con word2vec skip-gram (gensim) sobre un corpus de cerca de 1.500 millones de palabras en español.
  - Página del proyecto: https://crscardellino.github.io/SBWCE/
  - Archivo original: https://cs.famaf.unc.edu.ar/~ccardellino/SBWCE/SBW-vectors-300-min5.txt.bz2

**Cita:** Cristian Cardellino: *Spanish Billion Words Corpus and Embeddings* (March 2016), https://crscardellino.github.io/SBWCE/

**Licencia:** el recurso original de Cristian Cardellino se distribuye bajo [Creative Commons Atribución-CompartirIgual 4.0 Internacional (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/deed.es). Esta versión reducida se distribuye bajo la misma licencia.
