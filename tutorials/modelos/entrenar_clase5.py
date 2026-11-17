"""Entrena el Transformer del tutorial 5 con la Wikipedia en español.

El tutorial construye el modelo pieza por pieza y carga los pesos que deja
este script. Las clases del modelo son las mismas que muestra el tutorial.

Uso, en orden:

    python entrenar_clase5.py datos      # tokenizador y tokens (una vez)
    python entrenar_clase5.py todo       # todo el volcado, mismo tokenizador
    python entrenar_clase5.py medir      # velocidad en esta GPU
    python entrenar_clase5.py gpt --epochs 2 --reanudar
    python entrenar_clase5.py bert --epochs 1 --reanudar

Datos: volcado de la Wikipedia en español del 1 de noviembre de 2023
(`wikimedia/wikipedia`, configuración `20231101.es`), licencia CC BY-SA 4.0.
"""
import argparse
import csv
import json
import math
import signal
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from tqdm import tqdm

WIKIPEDIA = Path.home() / "Documents/tfidf-demonstration/data/20231101.es"
DATOS = Path.home() / "Documents/nlp/datos_clase5"
SALIDA = Path(__file__).parent
FIN = "<fin>"
MASCARA = "<mascara>"


# ---------------------------------------------------------------------------
# El modelo (el tutorial muestra estas mismas clases)
# ---------------------------------------------------------------------------

@dataclass
class Configuracion:
    """Tamaños del modelo."""

    n_tokens: int = 16384
    dim: int = 384
    contexto: int = 256
    causal: bool = True


class Cabeza(nn.Module):
    """Una cabeza de atención."""

    def __init__(self, dim: int, dim_cabeza: int, causal: bool) -> None:
        super().__init__()
        self.query = nn.Linear(dim, dim_cabeza, bias=False)
        self.key = nn.Linear(dim, dim_cabeza, bias=False)
        self.value = nn.Linear(dim, dim_cabeza, bias=False)
        self.causal = causal

    def pesos(self, x: torch.Tensor) -> torch.Tensor:
        """x: (lote, largo, dim); devuelve (lote, largo, largo).

        Fila = token que pregunta; columna = token al que mira.
        """
        Q = self.query(x)
        K = self.key(x)
        puntajes = Q @ K.transpose(1, 2) / math.sqrt(K.shape[-1])
        if self.causal:
            # cada token solo mira los anteriores y a sí mismo
            largo = x.shape[1]
            futuro = torch.ones(largo, largo, dtype=torch.bool,
                                device=x.device).triu(diagonal=1)
            puntajes = puntajes.masked_fill(futuro, float("-inf"))
        return torch.softmax(puntajes, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (lote, largo, dim); devuelve (lote, largo, dim_cabeza)."""
        return self.pesos(x) @ self.value(x)


class VariasCabezas(nn.Module):
    """Varias cabezas de atención en paralelo."""

    def __init__(self, dim: int, causal: bool) -> None:
        super().__init__()
        # 6 cabezas de dimensión 64: 6 × 64 = 384
        self.cabezas = nn.ModuleList([
            Cabeza(dim, 64, causal),
            Cabeza(dim, 64, causal),
            Cabeza(dim, 64, causal),
            Cabeza(dim, 64, causal),
            Cabeza(dim, 64, causal),
            Cabeza(dim, 64, causal),
        ])
        self.salida = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (lote, largo, dim); devuelve (lote, largo, dim)."""
        # cada cabeza aporta su parte; se juntan y se mezclan
        juntas = torch.cat([
            self.cabezas[0](x),
            self.cabezas[1](x),
            self.cabezas[2](x),
            self.cabezas[3](x),
            self.cabezas[4](x),
            self.cabezas[5](x),
        ], dim=-1)
        return self.salida(juntas)


class FeedForward(nn.Module):
    """El bloque feed-forward: el mismo para cada token, por separado."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.up_proj = nn.Linear(dim, 4 * dim)
        self.down_proj = nn.Linear(4 * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (lote, largo, dim); devuelve (lote, largo, dim)."""
        # cada fila de `up_proj` es una pregunta; la ReLU deja pasar
        # solo las respuestas positivas; cada columna de `down_proj`
        # es una dirección que se suma si su pregunta respondió
        respuestas = torch.relu(self.up_proj(x))
        return self.down_proj(respuestas)


class Bloque(nn.Module):
    """Atención y feed-forward, cada uno sumado al vector que entra."""

    def __init__(self, dim: int, causal: bool) -> None:
        super().__init__()
        self.norma_atencion = nn.LayerNorm(dim)
        self.atencion = VariasCabezas(dim, causal)
        self.norma_feed_forward = nn.LayerNorm(dim)
        self.feed_forward = FeedForward(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (lote, largo, dim); devuelve (lote, largo, dim)."""
        # cada parte suma su aporte al vector, no lo reemplaza
        x = x + self.atencion(self.norma_atencion(x))
        x = x + self.feed_forward(self.norma_feed_forward(x))
        return x


class Transformer(nn.Module):
    """Tokens → vectores → bloques → un puntaje para cada token."""

    def __init__(self, config: Configuracion) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.n_tokens, config.dim)
        self.posicion = nn.Embedding(config.contexto, config.dim)
        self.bloques = nn.ModuleList([
            Bloque(config.dim, config.causal),
            Bloque(config.dim, config.causal),
            Bloque(config.dim, config.causal),
            Bloque(config.dim, config.causal),
            Bloque(config.dim, config.causal),
            Bloque(config.dim, config.causal),
        ])
        self.norma_final = nn.LayerNorm(config.dim)
        # los pesos parten al azar, con valores chicos
        for parametro in self.parameters():
            if parametro.dim() > 1:
                nn.init.normal_(parametro, std=0.02)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: (lote, largo); devuelve (lote, largo, n_tokens)."""
        posiciones = torch.arange(ids.shape[1], device=ids.device)
        x = self.embedding(ids) + self.posicion(posiciones)
        x = self.bloques[0](x)
        x = self.bloques[1](x)
        x = self.bloques[2](x)
        x = self.bloques[3](x)
        x = self.bloques[4](x)
        x = self.bloques[5](x)
        x = self.norma_final(x)
        # la salida usa la misma tabla de embeddings: el producto punto
        # del vector con la fila de cada token es su puntaje
        return x @ self.embedding.weight.T


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------

def textos_wikipedia(fraccion: float, semilla: int = 0):
    """Recorre una muestra de artículos de los 13 archivos del volcado.

    Parameters
    ----------
    fraccion : float
        Fracción de artículos que se toma de cada archivo.
    semilla : int
        Semilla del muestreo.

    Yields
    ------
    tuple[int, str]
        Id del artículo y su título más su texto.
    """
    generador = np.random.default_rng(semilla)
    for archivo in sorted(WIKIPEDIA.glob("train-*.parquet")):
        tabla = pq.read_table(archivo, columns=["id", "title", "text"])
        elegidos = generador.random(tabla.num_rows) < fraccion
        muestra = tabla.take(np.flatnonzero(elegidos))
        for id_articulo, titulo, texto in zip(muestra["id"].to_pylist(),
                                              muestra["title"].to_pylist(),
                                              muestra["text"].to_pylist()):
            yield int(id_articulo), titulo + "\n" + texto


def preparar_datos(n_tokens: int, fraccion: float,
                   tokens_entreno: int) -> None:
    """Entrena el tokenizador y guarda los tokens de entreno y validación.

    Parameters
    ----------
    n_tokens : int
        Tamaño del vocabulario del tokenizador.
    fraccion : float
        Fracción de artículos del volcado que se usan.
    tokens_entreno : int
        Tope de tokens de entreno; la validación son 2 millones aparte.
    """
    DATOS.mkdir(parents=True, exist_ok=True)
    articulos = list(textos_wikipedia(fraccion))
    print(f"{len(articulos)} artículos")

    # el 1 % de los artículos va a validación, separado por artículo
    generador = np.random.default_rng(1)
    es_validacion = generador.random(len(articulos)) < 0.01

    tokenizador = Tokenizer(models.BPE())
    tokenizador.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False)
    tokenizador.decoder = decoders.ByteLevel()
    entrenador = trainers.BpeTrainer(
        vocab_size=n_tokens, special_tokens=[FIN, MASCARA],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    muestra = [texto for _, texto in articulos[::10]]
    tokenizador.train_from_iterator(muestra, entrenador)
    tokenizador.save(str(DATOS / "tokenizador.json"))
    print("tokenizador listo:", tokenizador.get_vocab_size(), "tokens")

    id_fin = tokenizador.token_to_id(FIN)
    for nombre, cuales, tope in [("validacion", es_validacion, 2_000_000),
                                 ("entreno", ~es_validacion, tokens_entreno)]:
        textos = [t for (_, t), sirve in zip(articulos, cuales) if sirve]
        partes, total = [], 0
        for inicio in range(0, len(textos), 2000):
            lote = tokenizador.encode_batch(textos[inicio:inicio + 2000])
            for codificado in lote:
                partes.append(np.array(codificado.ids + [id_fin],
                                       dtype=np.uint16))
                total += len(codificado.ids) + 1
            if total >= tope:
                break
        tokens = np.concatenate(partes)[:tope]
        tokens.tofile(DATOS / f"{nombre}.bin")
        print(f"{nombre}: {len(tokens):,} tokens")


def ids_validacion(fraccion: float = 0.35) -> set[str]:
    """Los artículos que `preparar_datos` dejó para validación.

    Repite el mismo muestreo (mismas semillas) leyendo solo los ids.

    Parameters
    ----------
    fraccion : float
        La fracción que se usó en `preparar_datos`.

    Returns
    -------
    set[str]
        Ids de los artículos de validación.
    """
    generador = np.random.default_rng(0)
    ids = []
    for archivo in sorted(WIKIPEDIA.glob("train-*.parquet")):
        columna = pq.read_table(archivo, columns=["id"])["id"].to_pylist()
        elegidos = generador.random(len(columna)) < fraccion
        ids += [i for i, sirve in zip(columna, elegidos) if sirve]
    es_validacion = np.random.default_rng(1).random(len(ids)) < 0.01
    return {i for i, sirve in zip(ids, es_validacion) if sirve}


def tokenizar_todo() -> None:
    """Tokeniza todo el volcado, menos la validación, en `entreno.bin`.

    Usa el tokenizador de `preparar_datos` y la misma validación; el
    entreno anterior (el 35 %) queda como `entreno_35.bin`.
    """
    tokenizador = Tokenizer.from_file(str(DATOS / "tokenizador.json"))
    id_fin = tokenizador.token_to_id(FIN)
    fuera = ids_validacion()
    print(f"{len(fuera)} artículos de validación quedan fuera")
    anterior = DATOS / "entreno.bin"
    if anterior.exists() and not (DATOS / "entreno_35.bin").exists():
        anterior.rename(DATOS / "entreno_35.bin")
    total, articulos = 0, 0
    with open(DATOS / "entreno.bin", "wb") as salida:
        for archivo in sorted(WIKIPEDIA.glob("train-*.parquet")):
            tabla = pq.read_table(archivo, columns=["id", "title", "text"])
            textos = [titulo + "\n" + texto for i, titulo, texto
                      in zip(tabla["id"].to_pylist(),
                             tabla["title"].to_pylist(),
                             tabla["text"].to_pylist()) if i not in fuera]
            for inicio in range(0, len(textos), 2000):
                lote = tokenizador.encode_batch(textos[inicio:inicio + 2000])
                for codificado in lote:
                    np.array(codificado.ids + [id_fin],
                             dtype=np.uint16).tofile(salida)
                    total += len(codificado.ids) + 1
            articulos += len(textos)
            print(f"{archivo.name}: {articulos} artículos, "
                  f"{total:,} tokens", flush=True)


def armar_lote(tokens: np.ndarray, inicios: np.ndarray, contexto: int,
               objetivo: str, id_mascara: int, dispositivo: str,
               generador: torch.Generator | None = None
               ) -> tuple[torch.Tensor, torch.Tensor]:
    """Arma entradas y respuestas con las ventanas que empiezan en `inicios`.

    Parameters
    ----------
    tokens : np.ndarray
        Todos los tokens, uno detrás de otro.
    inicios : np.ndarray
        Dónde empieza cada ventana del lote.
    contexto : int
        Largo de cada ventana.
    objetivo : str
        "siguiente" (GPT: predecir el token siguiente) o "tapada"
        (BERT: predecir el 15 % de los tokens, tapados con <mascara>).
    id_mascara : int
        Id del token <mascara>.
    dispositivo : str
        "cuda" o "cpu".
    generador : torch.Generator | None
        Para tapar siempre los mismos tokens (validación).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Entradas (lote, contexto) y respuestas (lote, contexto); con
        "tapada", las respuestas valen -100 donde no hay nada que predecir.
    """
    ventanas = np.stack([tokens[i:i + contexto + 1] for i in inicios])
    ventanas = torch.from_numpy(ventanas.astype(np.int64))
    if objetivo == "siguiente":
        entradas, respuestas = ventanas[:, :-1], ventanas[:, 1:]
    else:
        entradas = ventanas[:, :-1].clone()
        tapar = torch.rand(entradas.shape, generator=generador) < 0.15
        respuestas = torch.where(tapar, entradas, -100)
        entradas[tapar] = id_mascara
    return entradas.to(dispositivo), respuestas.to(dispositivo)


# ---------------------------------------------------------------------------
# Diagnóstico: qué sabe el modelo en cada etapa del entrenamiento
# ---------------------------------------------------------------------------

FRASES_GPT = ["La capital de Chile es", "Pablo Neruda escribió",
              "Me senté a leer en un banco de la",
              "Pedí un crédito al banco y", "El perro muerde al",
              "La Segunda Guerra Mundial", "El río más largo de"]
FRASES_BERT = ["Me senté a leer en un <mascara> de la plaza.",
               "Pedí un crédito al <mascara> y me lo dieron.",
               "La capital de Chile es <mascara>.",
               "Pablo Neruda escribió <mascara> poemas de amor.",
               "El perro muerde al <mascara>."]
FRASE_ATENCION = "Pedí un crédito al banco y me lo dieron."
PALABRAS = [" banco", " Chile", " rey", " escribió", " azul", " río"]
PASOS_COPIA = (10, 30, 100, 300, 1000, 3000, 10000, 30000)


@torch.no_grad()
def diagnostico(modelo: Transformer, tokenizador: Tokenizer,
                paso: int, minutos: float) -> dict:
    """Mide qué sabe el modelo: predicciones, atención y vecinos.

    Parameters
    ----------
    modelo : Transformer
        El modelo, en la GPU.
    tokenizador : Tokenizer
        El tokenizador del modelo.
    paso : int
        Pasos de entrenamiento hechos.
    minutos : float
        Minutos desde el inicio del entrenamiento.

    Returns
    -------
    dict
        Una entrada del registro de evolución.
    """
    modelo.eval()
    texto_de = tokenizador.decode
    entrada = {"paso": paso, "minutos": round(minutos, 2)}

    def mejores(puntajes: torch.Tensor) -> list:
        """Los 5 tokens más probables con su probabilidad."""
        probabilidades = torch.softmax(puntajes.float(), dim=-1)
        top = torch.topk(probabilidades, 5)
        return [[texto_de([i]), round(p, 4)] for p, i
                in zip(top.values.tolist(), top.indices.tolist())]

    predicciones = []
    if modelo.config.causal:
        generador = torch.Generator(device="cuda").manual_seed(0)
        for frase in FRASES_GPT:
            ids = tokenizador.encode(frase).ids
            siguientes = mejores(modelo(torch.tensor([ids]).cuda())[0, -1])
            for _ in range(25):
                puntajes = modelo(torch.tensor([ids]).cuda())[0, -1]
                probabilidades = torch.softmax(puntajes.float(), dim=-1)
                ids.append(torch.multinomial(probabilidades, 1,
                                             generator=generador).item())
            predicciones.append({"frase": frase, "siguientes": siguientes,
                                 "continuacion": texto_de(ids)})
    else:
        id_mascara = tokenizador.token_to_id(MASCARA)
        for frase in FRASES_BERT:
            ids = tokenizador.encode(frase.replace(" " + MASCARA,
                                                   MASCARA)).ids
            puntajes = modelo(torch.tensor([ids]).cuda())[0]
            donde = ids.index(id_mascara)
            predicciones.append({"frase": frase,
                                 "tapada": mejores(puntajes[donde])})
    entrada["predicciones"] = predicciones

    # los pesos de todas las cabezas para una oración
    ids = tokenizador.encode(FRASE_ATENCION).ids
    x = torch.tensor([ids]).cuda()
    x = modelo.embedding(x) + modelo.posicion(
        torch.arange(len(ids), device="cuda"))
    pesos = []
    for bloque in modelo.bloques:
        normado = bloque.norma_atencion(x)
        pesos.append([cabeza.pesos(normado)[0].float().cpu().numpy()
                      .round(3).tolist()
                      for cabeza in bloque.atencion.cabezas])
        x = bloque(x)
    entrada["atencion"] = {"tokens": [texto_de([i]) for i in ids],
                           "pesos": pesos}

    # los vecinos de algunas palabras en la tabla de embeddings
    tabla = F.normalize(modelo.embedding.weight.float(), dim=-1)
    vecinos = {}
    for palabra in PALABRAS:
        id_palabra = tokenizador.encode(palabra).ids[0]
        cosenos = tabla @ tabla[id_palabra]
        cercanos = torch.topk(cosenos, 6).indices.tolist()[1:]
        vecinos[palabra.strip()] = [texto_de([i]).strip()
                                    for i in cercanos]
    entrada["vecinos"] = vecinos
    return entrada


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------

def entrenar(modelo: Transformer, objetivo: str, epochs: int, nombre: str,
             carpeta: Path = SALIDA, tamano_lote: int = 64,
             tasa_maxima: float = 1e-3, reanudar: bool = False,
             tope_pasos: int | None = None) -> None:
    """Entrena el modelo y deja un registro detallado del proceso.

    El texto de entreno se corta en ventanas seguidas, sin traslape; cada
    epoch pasa una vez por todas, en un orden distinto.

    Escribe en `carpeta` `<nombre>.csv` (pérdidas, tasa de aprendizaje,
    norma del gradiente, tiempo y tokens vistos), `<nombre>_evolucion.jsonl`
    (lo que predice el modelo en cada etapa) y `<nombre>_resumen.json`; en
    la carpeta de datos, `<nombre>.log` (el avance, para mirarlo con
    `tail -f`), una copia del modelo en cada paso de `PASOS_COPIA` y cada
    5.000 pasos, y el estado para seguir (cada 5.000 pasos, al final y al
    detenerlo con Ctrl+C).

    Parameters
    ----------
    modelo : Transformer
        El modelo, ya en la GPU.
    objetivo : str
        "siguiente" (GPT) o "tapada" (BERT).
    epochs : int
        Cuántas veces se recorre todo el texto de entreno.
    nombre : str
        Prefijo de los archivos de salida.
    carpeta : Path
        Dónde van el CSV, la evolución y el resumen.
    tamano_lote : int
        Ventanas por paso.
    tasa_maxima : float
        Tasa de aprendizaje después del calentamiento.
    reanudar : bool
        Si hay un estado guardado (`<nombre>_estado.pt`), sigue desde ahí
        en vez de empezar de cero.
    tope_pasos : int | None
        Para pruebas: corta el entrenamiento en este paso.
    """
    config = modelo.config
    entreno = np.fromfile(DATOS / "entreno.bin", dtype=np.uint16)
    validacion = np.fromfile(DATOS / "validacion.bin", dtype=np.uint16)
    tokenizador = Tokenizer.from_file(str(DATOS / "tokenizador.json"))
    id_mascara = tokenizador.token_to_id(MASCARA)
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=tasa_maxima,
                                    betas=(0.9, 0.95), weight_decay=0.1)
    calentamiento = 500

    # el texto cortado en ventanas seguidas de `contexto` tokens
    n_ventanas = (len(entreno) - 1) // config.contexto
    pasos_por_epoch = n_ventanas // tamano_lote
    pasos = epochs * pasos_por_epoch
    if tope_pasos is not None:
        pasos = min(pasos, tope_pasos)
    # la validación usa siempre las mismas 640 ventanas
    inicios_validacion = np.arange(10 * tamano_lote) * config.contexto
    orden = {}

    def inicios_del_paso(paso: int) -> np.ndarray:
        """Las ventanas del lote de este paso; cada epoch las baraja."""
        epoch, k = divmod(paso - 1, pasos_por_epoch)
        if epoch not in orden:
            orden.clear()
            orden[epoch] = np.random.default_rng(epoch).permutation(
                n_ventanas)
        elegidas = orden[epoch][k * tamano_lote:(k + 1) * tamano_lote]
        return elegidas * config.contexto

    def tasa(paso: int) -> float:
        """Sube durante el calentamiento y después baja como coseno."""
        if paso < calentamiento:
            return tasa_maxima * (paso + 1) / calentamiento
        avance = (paso - calentamiento) / (pasos - calentamiento)
        return tasa_maxima * (0.1 + 0.45 * (1 + math.cos(math.pi * avance)))

    def perdida(datos: np.ndarray, inicios: np.ndarray,
                generador: torch.Generator | None = None) -> torch.Tensor:
        """Pérdida (entropía cruzada) de un lote."""
        entradas, respuestas = armar_lote(
            datos, inicios, config.contexto, objetivo, id_mascara, "cuda",
            generador)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            puntajes = modelo(entradas)
        return F.cross_entropy(puntajes.float().flatten(0, 1),
                               respuestas.flatten(), ignore_index=-100)

    def anotar(paso: int, minutos: float) -> None:
        """Anota el diagnóstico y, en algunos pasos, guarda una copia."""
        if paso == 0 or paso in PASOS_COPIA or paso % 5000 == 0:
            guardar(modelo, DATOS / f"{nombre}_paso{paso}.pt")
        with open(carpeta / f"{nombre}_evolucion.jsonl", "a") as archivo:
            json.dump(diagnostico(modelo, tokenizador, paso, minutos),
                      archivo, ensure_ascii=False)
            archivo.write("\n")

    ruta_estado = DATOS / f"{nombre}_estado.pt"
    ruta_csv = carpeta / f"{nombre}.csv"
    ruta_evolucion = carpeta / f"{nombre}_evolucion.jsonl"
    columnas = ["paso", "epoch", "tokens_vistos", "minutos", "tasa",
                "norma_gradiente", "perdida_entreno", "perdida_validacion"]
    primer_paso, minutos_previos = 1, 0.0
    if reanudar and ruta_estado.exists():
        estado = torch.load(ruta_estado, weights_only=False)
        modelo.load_state_dict(estado["modelo"])
        optimizador.load_state_dict(estado["optimizador"])
        torch.set_rng_state(estado["azar_torch"])
        torch.cuda.set_rng_state(estado["azar_cuda"])
        np.random.set_state(estado["azar_numpy"])
        primer_paso = estado["paso"] + 1
        minutos_previos = estado["minutos"]
        # se borra lo anotado después del estado guardado
        with open(ruta_csv) as archivo:
            filas = [f for f in csv.reader(archivo)]
        filas = [filas[0]] + [f for f in filas[1:]
                              if int(f[0]) <= estado["paso"]]
        with open(ruta_csv, "w", newline="") as archivo:
            csv.writer(archivo).writerows(filas)
        with open(ruta_evolucion) as archivo:
            lineas = [l for l in archivo
                      if json.loads(l)["paso"] <= estado["paso"]]
        with open(ruta_evolucion, "w") as archivo:
            archivo.writelines(lineas)
        print(f"sigue desde el paso {estado['paso']}")
    else:
        ruta_evolucion.unlink(missing_ok=True)
        (DATOS / f"{nombre}.log").unlink(missing_ok=True)
        with open(ruta_csv, "w", newline="") as archivo:
            csv.writer(archivo).writerow(columnas)
        anotar(0, 0.0)

    # Ctrl+C no corta a mitad de un paso: termina el paso, guarda y sale.
    # Un segundo Ctrl+C sí corta de inmediato.
    parar = False

    def pedir_parada(senal: int, marco: object) -> None:
        """Marca que hay que parar al terminar el paso actual."""
        nonlocal parar
        parar = True
        signal.signal(signal.SIGINT, signal.default_int_handler)

    signal.signal(signal.SIGINT, pedir_parada)
    with open(ruta_csv, "a", newline="") as archivo:
        escritor = csv.writer(archivo)
        inicio = time.time() - minutos_previos * 60
        perdidas, normas = [], []
        barra = tqdm(range(primer_paso, pasos + 1), desc=nombre,
                     unit="paso", initial=primer_paso - 1, total=pasos,
                     dynamic_ncols=True)
        for paso in barra:
            for grupo in optimizador.param_groups:
                grupo["lr"] = tasa(paso - 1)
            modelo.train()
            perdida_entreno = perdida(entreno, inicios_del_paso(paso))
            optimizador.zero_grad()
            perdida_entreno.backward()
            norma = torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            optimizador.step()
            perdidas.append(perdida_entreno.item())
            normas.append(norma.item())

            # registro denso al principio, cuando todo cambia rápido
            cada = 10 if paso <= 1000 else 100
            if paso % cada == 0 or paso == pasos:
                modelo.eval()
                with torch.no_grad():
                    perdida_val = np.mean([
                        perdida(validacion,
                                inicios_validacion[j * tamano_lote:
                                                   (j + 1) * tamano_lote],
                                torch.Generator().manual_seed(j)).item()
                        for j in range(10)])
                minutos = (time.time() - inicio) / 60
                epoch = paso / pasos_por_epoch
                escritor.writerow([paso, round(epoch, 4),
                                   paso * tamano_lote * config.contexto,
                                   round(minutos, 3), f"{tasa(paso - 1):.2e}",
                                   round(float(np.mean(normas)), 4),
                                   round(float(np.mean(perdidas)), 4),
                                   round(float(perdida_val), 4)])
                archivo.flush()
                perdidas, normas = [], []
                barra.set_postfix(epoch=f"{epoch:.2f}",
                                  validacion=f"{perdida_val:.3f}")
                with open(DATOS / f"{nombre}.log", "a") as avance:
                    avance.write(f"paso {paso} de {pasos} (epoch "
                                 f"{epoch:.2f}): validación "
                                 f"{perdida_val:.3f}, {minutos:.1f} min\n")
            if (paso in PASOS_COPIA or paso % 1000 == 0
                    or paso == pasos):
                anotar(paso, (time.time() - inicio) / 60)
            if paso % 5000 == 0 or paso == pasos or parar:
                # todo lo necesario para seguir desde aquí
                temporal = ruta_estado.with_suffix(".tmp")
                torch.save({"paso": paso,
                            "minutos": (time.time() - inicio) / 60,
                            "modelo": modelo.state_dict(),
                            "optimizador": optimizador.state_dict(),
                            "azar_torch": torch.get_rng_state(),
                            "azar_cuda": torch.cuda.get_rng_state(),
                            "azar_numpy": np.random.get_state()}, temporal)
                temporal.replace(ruta_estado)
            if parar:
                barra.close()
                print(f"Detenido en el paso {paso}, con todo guardado. "
                      "Para seguir, correr de nuevo entrenar_clase5.sh.")
                sys.exit(130)

    signal.signal(signal.SIGINT, signal.default_int_handler)
    minutos = (time.time() - inicio) / 60
    resumen = {"epochs": epochs, "pasos": pasos,
               "pasos_por_epoch": pasos_por_epoch, "ventanas": n_ventanas,
               "minutos": round(minutos, 1),
               "tokens_vistos": pasos * tamano_lote * config.contexto,
               "tokens_por_segundo": round(
                   pasos * tamano_lote * config.contexto / minutos / 60),
               "tokens_de_entreno": len(entreno),
               "parametros": sum(p.numel() for p in modelo.parameters()),
               "gpu": torch.cuda.get_device_name(),
               "config": asdict(config), "objetivo": objetivo,
               "tamano_lote": tamano_lote, "tasa_maxima": tasa_maxima}
    with open(carpeta / f"{nombre}_resumen.json", "w") as archivo:
        json.dump(resumen, archivo, ensure_ascii=False, indent=1)


def guardar(modelo: nn.Module, ruta: Path) -> None:
    """Guarda la configuración y los pesos en float16."""
    pesos = {nombre: tensor.half().cpu()
             for nombre, tensor in modelo.state_dict().items()}
    torch.save({"config": asdict(modelo.config), "pesos": pesos}, ruta)


def principal() -> None:
    """Lee la orden de la línea de comandos y la ejecuta."""
    parser = argparse.ArgumentParser()
    parser.add_argument("orden",
                        choices=["datos", "todo", "medir", "gpt", "bert"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--fraccion", type=float, default=0.35)
    parser.add_argument("--tokens", type=int, default=600_000_000)
    parser.add_argument("--reanudar", action="store_true")
    parser.add_argument("--tope", type=int, default=300)
    argumentos = parser.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)

    if argumentos.orden == "datos":
        preparar_datos(Configuracion().n_tokens, argumentos.fraccion,
                       argumentos.tokens)
        return
    if argumentos.orden == "todo":
        tokenizar_todo()
        return

    causal = argumentos.orden != "bert"
    modelo = Transformer(Configuracion(causal=causal)).cuda()
    print(f"{sum(p.numel() for p in modelo.parameters()):,} parámetros")
    objetivo = "siguiente" if causal else "tapada"
    if argumentos.orden == "medir":
        entrenar(modelo, objetivo, 1, "medir", carpeta=DATOS,
                 tope_pasos=argumentos.tope, reanudar=argumentos.reanudar)
        return
    nombre = "5-gpt-wikipedia" if causal else "5-bert-wikipedia"
    entrenar(modelo, objetivo, argumentos.epochs, nombre,
             reanudar=argumentos.reanudar)
    guardar(modelo, SALIDA / f"{nombre}.pt")


if __name__ == "__main__":
    principal()
