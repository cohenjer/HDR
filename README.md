[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/cohenjer/HDR/HEAD)


# HDR manuscript of Jeremy E. Cohen, July 2026

This book is published using [Jupyter-book v1](https://jupyterbook.org/v1/intro.html) at this adress:

https://cohenjer.github.io/HDR/intro.html

It is meant to be fully reproducible: all the numerical experiments are embedded in the markdown files, most dependencies are widely available packages (see [installation](#installation) below), while some of the more specific modules are contained in `/tensorly_hdr`.

A presentation summarizing the book is available in `/presentation` and was produced using [quarto](https://quarto.org/).

To cite this book, use the following:

```
Cohen, Jérémy E., "Regularized Low-rank Approximations", Habilitation à Diriger les Recherches (HDR), Université Claude Bernard Lyon 1, July 2026. 
```

A pdf version is available [here](https://github.com/cohenjer/HDR/blob/pdf/manuscript_HDR_Cohen_v1.pdf).

## Installation

Assuming a working installation of mamba/conda/virtualenv and python, first create a virtual environment with python 3.11

```bash
mamba create -n HDR python=3.11
mamba activate HDR
```

then install the dependencies using either

```bash
pip install -r requirements.txt
```

or if you encounter any problems, using the full frozen dependencies

```bash
pip install -r requirements_auto.txt
```

Then you can render the book locally (which will trigger all the notebooks interpretation, this will take a couple minutes)

```bash
jupyter-book build .
```

To build the presentation as well, one may simply install the quarto-cli with `pip`:

```bash
pip install quarto-cli
```

then go in the presentation folder and run quarto

```bash
cd presentation
quarto render index.qmd
```