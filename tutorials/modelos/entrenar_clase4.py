"""Entrena los tres modelos que carga el tutorial 4-rnn.ipynb.

- 4-lm-peliculas.pt: modelo de lenguaje con una GRU, sobre los
  argumentos de películas de la Wikipedia en español (los de la
  clase 2).
- 4-etiquetador-ud.pt: etiquetador gramatical con una GRU
  bidireccional, sobre UD Spanish-GSD (CC BY-SA 4.0).
- 4-traductor-es-en.pt: encoder–decoder con dos GRU, sin atención,
  sobre las frases español–inglés de Tatoeba (CC BY 2.0 FR, vía
  manythings.org/anki). Guarda solo pesos y vocabularios, no frases;
  las 3.000 de prueba se recuperan con la misma semilla.

Uso: python entrenar_clase4.py (con GPU tarda unos minutos).
"""
import io
import random
import re
import urllib.request
import zipfile
from collections import Counter

import pandas as pd
import torch
import torch.nn as nn

torch.manual_seed(0)
disp = "cuda" if torch.cuda.is_available() else "cpu"


def tokenizar_lm(texto: str) -> list[str]:
    """Devuelve palabras y signos de puntuación en minúsculas."""
    return re.findall(r"\w+|[.,;:¿?¡!()]", texto.lower())


class ModeloDeLenguaje(nn.Module):
    def __init__(self, n_palabras: int, d: int = 64, h: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(n_palabras, d, padding_idx=0)
        self.recurrente = nn.GRU(d, h, batch_first=True)
        self.salida = nn.Linear(h, n_palabras)

    def forward(self, X, h=None):
        H, h = self.recurrente(self.embedding(X), h)
        return self.salida(H), h


class Etiquetador(nn.Module):
    def __init__(self, n_palabras: int, n_categorias: int,
                 d: int = 64, h: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(n_palabras, d, padding_idx=0)
        self.recurrente = nn.GRU(d, h, batch_first=True,
                                 bidirectional=True)
        self.salida = nn.Linear(2 * h, n_categorias)

    def forward(self, X):
        H, _ = self.recurrente(self.embedding(X))
        return self.salida(H)


def guardar(modelo: nn.Module, ruta: str, **extra) -> None:
    """Guarda los pesos en media precisión junto con el vocabulario."""
    pesos = {k: v.half().cpu() for k, v in modelo.state_dict().items()}
    torch.save({"pesos": pesos, **extra}, ruta)


def entrenar_lm(d: int = 128, h: int = 384, largo: int = 64,
                epocas: int = 5) -> None:
    """Entrena el modelo de lenguaje sobre los argumentos."""
    df = pd.read_csv("../data/peliculas_wikipedia.csv.gz")
    textos = [tokenizar_lm(t) + ["<fin>"] for t in df["argumento"].dropna()]
    frec = Counter(w for t in textos for w in t)
    vocab = ["<pad>", "<unk>"] + [w for w, n in frec.most_common(20_000)
                                  if n >= 5 and w != "<fin>"] + ["<fin>"]
    ids = {w: i for i, w in enumerate(vocab)}
    flujo = torch.tensor([ids.get(w, 1) for t in textos for w in t])
    n = len(flujo) // (largo + 1) * (largo + 1)
    S = flujo[:n].view(-1, largo + 1)
    S = S[torch.randperm(len(S))]
    val, tr = S[:500].to(disp), S[500:].to(disp)
    print(f"vocabulario {len(vocab)}, tokens {len(flujo):,}")

    lm = ModeloDeLenguaje(len(vocab), d, h).to(disp)
    opt = torch.optim.Adam(lm.parameters(), lr=2e-3)
    perdida = nn.CrossEntropyLoss(ignore_index=0)
    for epoca in range(epocas):
        lm.train()
        for i in torch.randperm(len(tr)).split(128):
            opt.zero_grad()
            salida, _ = lm(tr[i, :-1])
            error = perdida(salida.flatten(0, 1), tr[i, 1:].flatten())
            error.backward()
            nn.utils.clip_grad_norm_(lm.parameters(), 1.0)
            opt.step()
        lm.eval()
        with torch.no_grad():
            salida, _ = lm(val[:, :-1])
            v = perdida(salida.flatten(0, 1), val[:, 1:].flatten())
        print(f"época {epoca + 1}: perplejidad de validación "
              f"{v.exp():.0f}")
    guardar(lm, "4-lm-peliculas.pt", vocab=vocab, d=d, h=h)


def leer_conllu(parte: str) -> list[list[tuple[str, str]]]:
    """Devuelve las oraciones como listas de pares (palabra, categoría)."""
    url = ("https://raw.githubusercontent.com/UniversalDependencies/"
           f"UD_Spanish-GSD/master/es_gsd-ud-{parte}.conllu")
    texto = urllib.request.urlopen(url).read().decode()
    oraciones = []
    for bloque in texto.strip().split("\n\n"):
        filas = [l.split("\t") for l in bloque.split("\n")
                 if not l.startswith("#")]
        oraciones.append([(f[1].lower(), f[3]) for f in filas
                          if f[0].isdigit()])
    return oraciones


def entrenar_etiquetador(d: int = 100, h: int = 128,
                         epocas: int = 10) -> None:
    """Entrena el etiquetador sobre UD Spanish-GSD."""
    train, test = leer_conllu("train"), leer_conllu("test")
    frec = Counter(w for o in train for w, _ in o)
    vocab = ["<pad>", "<unk>"] + [w for w, n in frec.most_common() if n >= 2]
    ids = {w: i for i, w in enumerate(vocab)}
    categorias = ["<pad>"] + sorted({c for o in train for _, c in o})

    def lote(oraciones):
        pad = nn.utils.rnn.pad_sequence
        X = pad([torch.tensor([ids.get(w, 1) for w, _ in o])
                 for o in oraciones], batch_first=True)
        Y = pad([torch.tensor([categorias.index(c) for _, c in o])
                 for o in oraciones], batch_first=True)
        return X.to(disp), Y.to(disp)

    (X, Y), (X_t, Y_t) = lote(train), lote(test)
    et = Etiquetador(len(vocab), len(categorias), d, h).to(disp)
    opt = torch.optim.Adam(et.parameters(), lr=2e-3)
    perdida = nn.CrossEntropyLoss(ignore_index=0)
    for epoca in range(epocas):
        et.train()
        for i in torch.randperm(len(X)).split(32):
            # palabra desconocida al azar: aprende a usar el contexto
            Xi = X[i].masked_fill((torch.rand_like(X[i], dtype=torch.float)
                                   < 0.1) & (X[i] > 0), 1)
            opt.zero_grad()
            perdida(et(Xi).flatten(0, 1), Y[i].flatten()).backward()
            opt.step()
        et.eval()
        with torch.no_grad():
            pred = et(X_t).argmax(-1)
        reales = Y_t != 0
        acc = (pred[reales] == Y_t[reales]).float().mean()
        print(f"época {epoca + 1}: accuracy de prueba {acc:.3f}")
    guardar(et, "4-etiquetador-ud.pt", vocab=vocab, categorias=categorias,
            d=d, h=h)


def tokenizar_tr(texto: str) -> list[str]:
    """Devuelve palabras y signos en minúsculas, para traducir."""
    return re.findall(r"\w+(?:'\w+)?|[^\w\s]", texto.lower())


class Traductor(nn.Module):
    def __init__(self, n_origen: int, n_destino: int,
                 d: int = 192, h: int = 384):
        super().__init__()
        self.emb_origen = nn.Embedding(n_origen, d, padding_idx=0)
        self.emb_destino = nn.Embedding(n_destino, d, padding_idx=0)
        self.encoder = nn.GRU(d, h, batch_first=True)
        self.decoder = nn.GRU(d, h, batch_first=True)
        self.salida = nn.Linear(h, n_destino)
        self.dropout = nn.Dropout(0.2)

    def codificar(self, X):
        largos = (X != 0).sum(1).cpu()
        E = nn.utils.rnn.pack_padded_sequence(
            self.dropout(self.emb_origen(X)), largos, batch_first=True,
            enforce_sorted=False)
        return self.encoder(E)[1]  # el único vector que pasa

    def forward(self, X, Y):
        H, _ = self.decoder(self.dropout(self.emb_destino(Y)),
                            self.codificar(X))
        return self.salida(self.dropout(H))


def entrenar_traductor(d: int = 192, h: int = 384, n_vocab: int = 10_000,
                       epocas: int = 14) -> None:
    """Entrena el traductor español → inglés sobre Tatoeba."""
    pedido = urllib.request.Request(
        "https://www.manythings.org/anki/spa-eng.zip",
        headers={"User-Agent": "curl/8.5.0", "Accept": "*/*"})
    archivo = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(pedido).read()))
    filas = [l.split("\t") for l in
             archivo.read("spa.txt").decode().splitlines()]
    vistos, pares = set(), []
    for en, es, *_ in filas:
        if es not in vistos:
            vistos.add(es)
            o, d_ = tokenizar_tr(es), tokenizar_tr(en)
            if 0 < len(o) <= 20 and 0 < len(d_) <= 20:
                pares.append((o, d_))
    random.seed(0)
    random.shuffle(pares)
    test, train = pares[:3000], pares[3000:]

    def vocab(textos):
        frec = Counter(w for t in textos for w in t)
        return (["<pad>", "<s>", "</s>", "<unk>"]
                + [w for w, _ in frec.most_common(n_vocab - 4)])

    origen, destino = vocab(o for o, _ in train), vocab(d_ for _, d_ in train)
    io_, id_ = ({w: i for i, w in enumerate(v)} for v in (origen, destino))

    def lote(pares_):
        pad = nn.utils.rnn.pad_sequence
        X = pad([torch.tensor([io_.get(w, 3) for w in o]) for o, _ in pares_],
                batch_first=True)
        Y = pad([torch.tensor([1] + [id_.get(w, 3) for w in d_] + [2])
                 for _, d_ in pares_], batch_first=True)
        return X.to(disp), Y.to(disp)

    tr = Traductor(len(origen), len(destino), d, h).to(disp)
    opt = torch.optim.Adam(tr.parameters(), lr=2e-3)
    pasos = epocas * (len(train) // 256 + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=pasos)
    perdida = nn.CrossEntropyLoss(ignore_index=0)
    for epoca in range(epocas):
        tr.train()
        random.shuffle(train)
        orden = sorted(range(len(train)), key=lambda i: len(train[i][0]))
        lotes = [orden[i:i + 256] for i in range(0, len(orden), 256)]
        random.shuffle(lotes)
        for ids in lotes:
            X, Y = lote([train[i] for i in ids])
            opt.zero_grad()
            error = perdida(tr(X, Y[:, :-1]).flatten(0, 1), Y[:, 1:].flatten())
            error.backward()
            nn.utils.clip_grad_norm_(tr.parameters(), 1.0)
            opt.step()
            sched.step()
        print(f"época {epoca + 1}: pérdida {error.item():.2f}")
    guardar(tr, "4-traductor-es-en.pt", origen=origen, destino=destino,
            d=d, h=h)


if __name__ == "__main__":
    entrenar_etiquetador()
    entrenar_lm()
    entrenar_traductor()
