# Implementação kNN

Atividade de implementação do algoritmo kNN para disciplina de Aprendizado de Máquina, segundo semestre de 2019.

## Início

Existem três arquivos neste repositório:

* knn.py: programa em python que implementa o algoritmo kNN.
* train.dat: dados de treinamento.
* test.dat: dados de teste.

### Pré-requisitos

É necessario ter python 3, numpy, matplotlib.

## Algoritmo

O algoritmo se desenvolve em 3 passos principais:

* Carrega os dados dos arquivos em listas, fazendo uma transformação necessária para ajudar nos cálculos posteriores.
* Faz as previsões de qual classe os dados de teste pertencem com base nos dados de treinamento, escolhendo a classe mais comum dos k vizinhos mas próximos.
* Calcula a taxa de acerto e forma a matriz de confusão.

### Como Usar

python knn.py <k> <train_size> <test_size> (maximum 10.000 for both, k must be odd)

### Resultados

Com 10.000 dados de treinamento e 100 dados de teste a taxa de acerto fica em 93%.

Reduzindo os dados de treino para 1.000 e mantendo os 100 dados de teste, a taxa de acerto reduz drásticamente para 76%.

Portanto podemos concluir que quanto maior o número de dados para treino, melhor será a taxa de acerto.

## Bibliotecas Utilizadas

* [python](https://www.python.org/downloads/) - Python is a programming language that lets you work quickly
and integrate systems more effectively.
* [numpy](https://numpy.org/) - NumPy is the fundamental package for scientific computing with Python.
* [matplotlib](https://matplotlib.org/) - Matplotlib is a Python 2D plotting library which produces publication quality figures in a variety of hardcopy formats and interactive environments across platforms.

## Autor

* **Giovanni Rosa** - [giovannirosa](https://github.com/giovannirosa)

## Licença

Código aberto, qualquer um pode usar para qualquer propósito.

## Reconhecimentos

* Python é simples e fácil de usar

## Bugs

Nenhum bug aparente encontrado em centenas de testes realizados.
