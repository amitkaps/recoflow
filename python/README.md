# Python version of RecoFlow

To install the development environment on OSX / linux / bashonwindows

1. First install `poetry` (See https://python-poetry.org/docs/) & confirm installation

```sh
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
poetry --version
```

2. You need a compatible python environment in your path. If you are using conda to manage your python, then create a compatible python version.

```sh
conda create --name recoflow python=3.8
conda activate recoflow
```

3. Install the dependencies for the project

```sh
cd recoflow/python
poetry install
```

4. To build the library

```sh
poetry build
```
