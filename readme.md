# Photomath Assignment-A

Photomath Assignment-A is a simple Python-based ML pipeline for converting an image of math expression to the solution.

## Requirements

* [Docker](https://www.docker.com/)

If you want to run locally
* Python 3.8
* tensorflow==2.7.0
* Flask==2.0.1
* gunicorn==20.1.0
* tensorflow===2.7.0
* numpy==1.19.3
* opencv-python==4.5.4.58

## Clone repo locally
* Clone the project to `/some_folder` directory
* `cd` into the project root directory

## Build Docker container

Use [docker](https://www.docker.com/) to create container locally.

```bash
docker build --tag photomath:python .
```

## Run Locally
```bash
docker run --rm -p 9090:8080 -e PORT=8080 photomath:python
```

## Usage

`path_to_img` should be absolute path of expression image.
```curl
curl --location --request POST "127.0.0.1:9090/" --form "file=@\"[path_to_img]\""
```

## Jupyter notebooks

* `Testing Notebook [Local].ipynb` -  If you want to test code locally in jupyter lab, it is recommended to install Conca env with all packages and run this notebook.
* `Train_Classifier [Google Colab].ipynb` - If you want to start model training, it is recommended to run this nodebook on Google Collaborate.

## Discussion

The table below shows the results of the experiments, you can see more details in `Train_Classifier [Google Colab].ipynb` notebook.

| Fold # | Accracy |
| --- | --- |
| 1 | 0.9938 |
| 2 | 0.9941  |
| 3 | 0.9941  |
| 4 | 0.9943  |
| 5 | 0.9944  |

This is a very simple project that transforms an image of a mathematical expression into a string and then calculates the value of that expression. The first assumption is that the input is a white paper with a written expression, which is far from the real situation. The first thing I would suggest is to create a segmentation network that would detect the text in the image (if there is any text at all :)). Then, the results of the classification network can be improved by augmenting the input dataset. After that, an algorithm for postprocessing data can be added because some of the classes are very similar (such as `1` and `/`) and it is difficult to do with the classifier only, but it is possible with certain algorithms such as (RNN, Rule Based, GNN, ensembles of previous algorithms of etc.).