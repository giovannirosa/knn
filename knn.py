import math
import operator
import numpy
import pylab as pl
import sys


def transfomData(lines):
    transfLines = []
    for line in lines:
        data = line.split(' ')
        values = list(map(lambda d: float(d.split(':')[1]), data[1:133]))
        values.append(data[0])
        transfLines.append(values)
    return transfLines


def loadDatasets(trainFile, testFile, train_size, test_size):
    with open(trainFile, 'r') as f1:
        trainingSet = transfomData(f1.readlines()[:train_size])
    with open(testFile, 'r') as f2:
        testSet = transfomData(f2.readlines()[:test_size])
    return [trainingSet, testSet]


def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(),
                         key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def getAccuracy(testSet, predictions):
    correct = 0
    plot = numpy.full((10, 10), 0)
    for x in range(len(testSet)):
        plot[int(testSet[x][-1])][int(predictions[x])] += 1
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return [(correct/float(len(testSet))) * 100.0, plot]


def main(k, train_size, test_size):
    # prepare data
    print('Loading datasets...')
    sets = loadDatasets('train.dat', 'test.dat', train_size, test_size)
    trainingSet = sets[0]
    testSet = sets[1]
    print('Training model with ' + repr(len(trainingSet)))
    print('Testing model with ' + repr(len(testSet)))
    # generate predictions
    predictions = []
    print('Making predictions...')
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        # print(repr(x))
    [accuracy, plot] = getAccuracy(testSet, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')
    # cria a matriz de confusao
    print(plot)
    pl.matshow(plot)
    pl.colorbar()
    pl.show()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("Use: knn.py <k> <train_size> <test_size> (maximum 10,000 for both)")

    k = int(sys.argv[1])
    train_size = int(sys.argv[2])
    test_size = int(sys.argv[3])
    if train_size > 10000 or test_size > 10000:
        sys.exit("Maximum size allowed is 10,000 for both")
    if k % 2 == 0:
        sys.exit("k must be odd")

    main(k, train_size, test_size)
