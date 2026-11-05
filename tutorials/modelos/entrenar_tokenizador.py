"""Entrena el tokenizador BPE de la clase 5 con la Wikipedia en español.

Reproduce `5-tokenizador.json`: BPE sobre bytes, 16.384 tokens, entrenado
con 64.389 artículos del volcado del 1 de noviembre de 2023
(`wikimedia/wikipedia`, configuración `20231101.es`, CC BY-SA 4.0).

La muestra es la misma que usa `entrenar_clase5.py`: de cada uno de los 13
archivos del volcado se toma el 35 % de los artículos al azar (semilla 0),
y de esos, uno de cada diez.

Uso:
    python entrenar_tokenizador.py [carpeta con los train-*.parquet]

Sin carpeta, descarga los archivos de Hugging Face (unos 2,7 GB).
"""
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

N_TOKENS = 16384
FRACCION = 0.35
SEMILLA = 0
ESPECIALES = ["<fin>", "<mascara>"]


def carpeta_volcado(argumentos: list[str]) -> Path:
    """La carpeta con los archivos del volcado; la descarga si no se da.

    Parameters
    ----------
    argumentos : list[str]
        Los argumentos de la línea de comandos.

    Returns
    -------
    Path
        Carpeta con los 13 archivos `train-*.parquet`.
    """
    if argumentos:
        return Path(argumentos[0])
    from huggingface_hub import snapshot_download
    raiz = snapshot_download("wikimedia/wikipedia", repo_type="dataset",
                             allow_patterns="20231101.es/*")
    return Path(raiz) / "20231101.es"


def muestra_wikipedia(carpeta: Path) -> list[str]:
    """Título y texto de los artículos de la muestra.

    Parameters
    ----------
    carpeta : Path
        Carpeta con los archivos `train-*.parquet`.

    Returns
    -------
    list[str]
        Uno de cada diez artículos de la muestra del 35 %.
    """
    generador = np.random.default_rng(SEMILLA)
    textos = []
    for archivo in sorted(carpeta.glob("train-*.parquet")):
        tabla = pq.read_table(archivo, columns=["title", "text"])
        elegidos = generador.random(tabla.num_rows) < FRACCION
        muestra = tabla.take(np.flatnonzero(elegidos))
        for titulo, texto in zip(muestra["title"].to_pylist(),
                                 muestra["text"].to_pylist()):
            textos.append(titulo + "\n" + texto)
    return textos[::10]


def main() -> None:
    textos = muestra_wikipedia(carpeta_volcado(sys.argv[1:]))
    print(f"{len(textos):,} artículos")

    # BPE sobre bytes: ningún texto queda fuera del vocabulario
    tokenizador = Tokenizer(models.BPE())
    tokenizador.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False)
    tokenizador.decoder = decoders.ByteLevel()
    entrenador = trainers.BpeTrainer(
        vocab_size=N_TOKENS, special_tokens=ESPECIALES,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tokenizador.train_from_iterator(textos, entrenador)
    tokenizador.save("5-tokenizador.json")
    print("Listo:", tokenizador.get_vocab_size(), "tokens")


if __name__ == "__main__":
    main()
