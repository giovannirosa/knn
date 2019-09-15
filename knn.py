import math
import operator
import numpy


def transfomData(lines):
    transfLines = []
    for line in lines:
        data = line.split(' ')
        values = list(map(lambda d: float(d.split(':')[1]), data[1:133]))
        values.append(data[0])
        transfLines.append(values)
    return transfLines


def loadDatasets(trainFile, testFile):
    with open(trainFile, 'r') as f1:
        trainingSet = transfomData(f1.readlines()[:1000])
    with open(testFile, 'r') as f2:
        testSet = transfomData(f2.readlines()[:1000])
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
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0


def main():
    # prepare data
    print('Loading datasets...')
    sets = loadDatasets('train.dat', 'test.dat')
    trainingSet = sets[0]
    testSet = sets[1]
    print('Training model with ' + repr(len(trainingSet)))
    print('Testing model with ' + repr(len(testSet)))
    # generate predictions
    predictions = []
    k = 3
    print('Making predictions...')
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        # print(repr(x))
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')


if __name__ == "__main__":
    main()
