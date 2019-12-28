import random
import copy
import math

# Инициализируем данные массива нейросети нулями, иначе читаем из файла
"""
writeNewData = False

if writeNewData:
    mode = 'w'
else:
    mode = 'r'
f = open('1.txt', mode)
nLayers1 = 0
neuronCount1 = []
if writeNewData:
    neuronCount1.append(27)
    neuronCount1.append(9)
    neuronCount1.append(9)
    neuronCount1.append(9)
    nLayers1 = len(neuronCount1)
    f.write(str(nLayers1) + '\n')
    for n in neuronCount1:
        f.write(str(n) + '\n')
else:
    nLayers1=int(f.readline())
    for n in range(nLayers1):
        neuronCount1.append(int(f.readline()))

wNeuron1=[]
for i in range(len(neuronCount1) - 1):
    row=[]
    for j in range(neuronCount1[i]):
        col=[]
        for k in range(neuronCount1[i + 1]):
            if writeNewData:
                col.append(0.0)
            else:
                col.append(float(f.readline()))
        row.append(col)
    wNeuron1.append(row)
if writeNewData:
    for i in range(len(wNeuron1)):
        for j in range(len(wNeuron1[i])):
            for k in range(len(wNeuron1[i][j])):
                f.write(str(wNeuron1[i][j][k]) + '\n')

f = open('2.txt', mode)
nLayers2 = 0
neuronCount2 = []
if writeNewData:
    neuronCount2.append(27)
    neuronCount2.append(9)
    neuronCount2.append(9)
    neuronCount2.append(9)
    nLayers2 = len(neuronCount2)
    f.write(str(nLayers2) + '\n')
    for n in neuronCount2:
        f.write(str(n) + '\n')
else:
    nLayers2=int(f.readline())
    for n in range(nLayers2):
        neuronCount2.append(int(f.readline()))

wNeuron2=[]
for i in range(len(neuronCount2) - 1):
    row=[]
    for j in range(neuronCount2[i]):
        col=[]
        for k in range(neuronCount2[i + 1]):
            if writeNewData:
                col.append(0.0)
            else:
                col.append(float(f.readline()))
        row.append(col)
    wNeuron2.append(row)
if writeNewData:
    for i in range(len(wNeuron2)):
        for j in range(len(wNeuron2[i])):
            for k in range(len(wNeuron2[i][j])):
                f.write(str(wNeuron2[i][j][k]) + '\n')                
"""


class NeuroWeb:
    def __init__(self, neuronCount):
        self.neuronCount = neuronCount
        self.nLayers = len(self.neuronCount)
        self.wLink = []
        for i in range(len(self.neuronCount)-1):
            row = []
            for j in range(self.neuronCount[i]+1):
                col = []
                for k in range(self.neuronCount[i+1]):
                    #col.append(0.0)
                    col.append(0.5)
                row.append(col)
            self.wLink.append(row)
        self.wKoef = 0.5            
        self.WinRated = 0.0    
        self.savedOutput = []
        self.savedInput = []
        self.actConst = []
        self.actConst.append(0.0)
        self.actConst.append(1.0)
        self.actConst.append(self.actConst[1] + 1.0/2.0)
        self.actConst.append(self.actConst[2] + 1.0/3.0)
        self.actConst.append(self.actConst[3] + 1.0/4.0)
        self.actConst.append(self.actConst[4] + 1.0/5.0)
        self.actConst.append(self.actConst[5] + 1.0/6.0)
        self.actConst.append(self.actConst[6] + 1.0/7.0)
        self.actConst.append(self.actConst[7] + 1.0/8.0)
        self.actConst.append(self.actConst[8] + 1.0/9.0)
        self.actConst.append(self.actConst[9] + 1.0/10.0)
        self.actConst.append(self.actConst[10] + 1.0/11.0)
        self.Winner = 0
        

    def activateFunc(self, signal):
        value = abs(signal)
        sigInt = int(value)
        #sigInt = int(signal)
        baseInd = 11
        if sigInt < 11:
            baseInd = sigInt
        if signal < 0:
            return -(self.actConst[baseInd] + (value - baseInd) / (baseInd + 1))
        else:
           return self.actConst[baseInd] + (value - baseInd) / (baseInd + 1)
        #return self.actConst[baseInd] + (signal - baseInd) / (baseInd + 1)

    def loadFromFile(self, fName):
        f = open(fName, 'r')
        self.nLayers = int(f.readline())
        for n in range(self.nLayers):
            self.neuronCount.append(int(f.readline()))
        for i in range(len(self.neuronCount)-1):
            row = []
            for j in range(self.neuronCount[i]+1):
                col = []
                for k in range(self.neuronCount[i+1]):
                    col.append(float(f.readline()))
                row.append(col)
            self.wLink.append(row)

    def setRandomWeigth(self):
        self.wLink = []
        for i in range(len(self.neuronCount)-1):
            row = []
            for j in range(self.neuronCount[i]+1):
                col = []
                for k in range(self.neuronCount[i+1]):
                    #col.append(0.95 * (random.random() + random.random() - 1))
                    col.append(0.05 + 0.45 * (random.random() + random.random()))
                row.append(col)
            self.wLink.append(row)

    def setTopology(self, topology):
        self.wLink = []
        self.neuronCount = []
        self.nLayers = len(topology)
        for n in topology:
            self.neuronCount.append(n)
        for i in range(len(self.neuronCount)-1):
            row = []
            for j in range(self.neuronCount[i]+1):
                col = []
                for k in range(self.neuronCount[i+1]):
                    #col.append(0.0)
                    col.append(0.5)
                row.append(col)
            self.wLink.append(row)

    def saveToFile(self, fName):
        f = open(fName, 'w')
        f.write(str(self.nLayers) + '\n')
        for n in self.neuronCount:
            f.write(str(n) + '\n')
        for i in self.wLink:
            for j in i:
                for val in j:
                    f.write(str(val) + '\n')

    def chooseAndSaveMove(self, inputLayer, randFactor=0.0, difLimit=1.0, rand=False):
        #Случайность выбора ходов
        #randFactor = 0.0
        #Разница отличия от максимума, когда решение уже не принимается к рассмотрению (кроме случа фактора)
        #difLimit = 1.0
        difLimitFactor = difLimit - randFactor
        # Для крестиков-ноликов отсекаем варианты, где нет выбора по ходу
        sum = 0
        for val in inputLayer:
            sum += abs(val)
        if sum > 7:
            for i in range(len(inputLayer)):
                if inputLayer[i] == 0:
                    return i
            return -1
        while rand:
            i = random.randint(0, 8)
            if inputLayer[i] == 0:
                self.savedInput.append(inputLayer)
                self.savedOutput.append(i)
                return i
        # Инициализация матрицы сигналов в нейронах и задание первого слоя
        self.wSignalNeuron = []
        for i in range(self.nLayers):
            wSignalNeuronLayer = []
            for j in range(self.neuronCount[i]):
                wSignalNeuronLayer.append(0)
            if i < len(self.neuronCount) - 1:
                wSignalNeuronLayer.append(1)
            self.wSignalNeuron.append(wSignalNeuronLayer)
        if self.neuronCount[0] == 9:                   
            for j in range(len(inputLayer)):
                #self.wSignalNeuron[0][j] = inputLayer[j]
                self.wSignalNeuron[0][j] = 0.05 + 0.45 * (inputLayer[j] + 1)
        else:                
            for j in range(len(inputLayer)):
                if inputLayer[j] == -1:
                    self.wSignalNeuron[0][j] = 1
                elif inputLayer[j] == 0:
                    self.wSignalNeuron[0][j + 9] = 1
                else:
                    self.wSignalNeuron[0][j + 18] = 1

        # Вычисляем по слоям выходной сигнал
        for i in range(1, self.nLayers):
            # Для каждого нейрона очередного слоя
            for j in range(self.neuronCount[i]):
                # Вычисляем поступившую входную силу сигнала от предыдущего слоя, умноженного на вес связи
                signal = 0.0
                for k in range(self.neuronCount[i-1]+1):
                    if self.wSignalNeuron[i-1][k] != 0.0:
                        signal += self.wSignalNeuron[i-1][k] * self.wLink[i-1][k][j]
                if i < len(self.wSignalNeuron)-1:
                    signal = self.activateFunc(signal)
                self.wSignalNeuron[i][j] = signal
        # Обрабатываем последний слой, нормируем значения
        #min = 1000
        max = -1000
        for i in range(len(self.wSignalNeuron[-1])):
            if inputLayer[i] == 0:
                #if min > self.wSignalNeuron[-1][i]:
                #    min = self.wSignalNeuron[-1][i]
                if max < self.wSignalNeuron[-1][i]:
                    max = self.wSignalNeuron[-1][i]
        sum = 0
        i = 0
        #if max < min + 0.5:
        #    max = min + 0.5
        for i in range(len(self.wSignalNeuron[-1])):
            # Для игры в крестики-нолики сразу отсекаем варианты, где уже стоит чужой знак
            #if self.wSignalNeuron[0][i] + self.wSignalNeuron[0][i + 18] != 0:
            if inputLayer[i]!=0:
                self.wSignalNeuron[-1][i] = 0
            else:
                #self.wSignalNeuron[-1][i] = randFactor + (self.wSignalNeuron[-1][i] - min) / (max - min)
                self.wSignalNeuron[-1][i] += difLimitFactor - max
                if self.wSignalNeuron[-1][i] < 0.0:
                    self.wSignalNeuron[-1][i] = 0.0
                self.wSignalNeuron[-1][i] += randFactor    
            sum += self.wSignalNeuron[-1][i]
        rVal = sum * random.random()
        sum = 0
        i = 0
        for val in self.wSignalNeuron[-1]:
            sum += val
            if sum > rVal:
                self.savedInput.append(inputLayer)
                self.savedOutput.append(i)
                return i
            i += 1
        return -1

    def clearSave(self):
        self.savedInput = []
        self.savedOutput = []

    # Обучение после игры
    def education(self, difWinRate, step):
        if self.WinRated == 0 or difWinRate == 0.0:
            return False

        wLinkEd = []
        for i in range(len(self.neuronCount)-1):
            row = []
            for j in range(self.neuronCount[i]+1):
                col = []
                for k in range(self.neuronCount[i+1]):
                    col.append(0.0)
                row.append(col)
            wLinkEd.append(row)

        # Суммарный базовый рост/снижение силы всех связей, до которой нормируется сила обучения
        linkLayerNorm = 0.3
        # Отклонение, при соблюдении которого проводится обучение. 
        # Например, если наш вариант хороший, но лучший, то его улучшение будет идти на всю катушку 
        # Если же он итак лучший, но не больше, чем на эту величину, то будет проводится его частичное обучение
        limitDifForEd = 2.0
        # Сила замедления обучения связи близ пограничных значений -1 и 1
        # reductionEd = 3.0
        #WinRated = 0
        #if self.Winner > 0.0:
        #    WinRated = 1 - 0.3 * (len(self.savedInput) - 3)
        #else:
        #    WinRated = -1 + 0.4 * (len(self.savedInput) - 2)
        linkLayerNorm *= abs(difWinRate)
        for iter in range(len(self.savedInput)):
            if step > -1: #and difWinRate < 0:
                iter = step
                if iter >= len(self.savedInput):
                    return False
            inputLayer = self.savedInput[iter]
            outputNeuron = self.savedOutput[iter]
            # Инициализация матрицы сигналов в нейронах и задание первого слоя
            self.wSignalNeuron = []
            wReverseSignalNeuron = []
            for i in range(self.nLayers):
                wSignalNeuronLayer = []
                wReverseSignalNeuronLayer = []
                for j in range(self.neuronCount[i]):
                    wSignalNeuronLayer.append(0)
                    wReverseSignalNeuronLayer.append(0)
                if i < len(self.neuronCount) - 1:
                    wSignalNeuronLayer.append(1)
                    wReverseSignalNeuronLayer.append(0)
                self.wSignalNeuron.append(wSignalNeuronLayer)
                wReverseSignalNeuron.append(wReverseSignalNeuronLayer)

            if self.neuronCount[0] == 9:                   
                for j in range(len(inputLayer)):
                    #self.wSignalNeuron[0][j] = inputLayer[j]
                    self.wSignalNeuron[0][j] = 0.05 + 0.45 * (inputLayer[j] + 1)
            else:                
                for j in range(len(inputLayer)):
                    if inputLayer[j] == -1:
                        self.wSignalNeuron[0][j] = 1
                    elif inputLayer[j] == 0:
                        self.wSignalNeuron[0][j + 9] = 1
                    else:
                        self.wSignalNeuron[0][j + 18] = 1
            
            # Вычисляем по слоям выходной сигнал
            for i in range(1, self.nLayers):
                # Для каждого нейрона очередного слоя
                for j in range(self.neuronCount[i]):
                    # Вычисляем поступившую входную силу сигнала от предыдущего слоя, умноженного на вес связи
                    signal = 0.0
                    for k in range(self.neuronCount[i-1]+1):
                        signal += self.wSignalNeuron[i-1][k] * self.wLink[i-1][k][j]
                    if i < len(self.wSignalNeuron)-1:
                        signal = self.activateFunc(signal)
                    self.wSignalNeuron[i][j] = signal
            # Определяем максимальный выходной сигнал среди всех прочих выходных нейронов
            maxOutput = 0
            for j in range(self.neuronCount[-1]):
                if j != outputNeuron:
                    if maxOutput < self.wSignalNeuron[-1][j]:
                        maxOutput = self.wSignalNeuron[-1][j]
            koefEd = 1.0               
            if difWinRate > 0:
                koefEd = (maxOutput + limitDifForEd - self.wSignalNeuron[-1][outputNeuron]) / limitDifForEd
            elif difWinRate < 0:
                koefEd = (self.wSignalNeuron[-1][outputNeuron] + limitDifForEd - maxOutput) / limitDifForEd
            if koefEd > 0:
                if koefEd > 1:
                    koefEd = math.sqrt(koefEd)
                # А на последнем слое списываем записанный output
                #self.wSignalNeuron[-1][outputNeuron] = 1.0
                wReverseSignalNeuron[-1][outputNeuron] = difWinRate
                #Определяем слой для обучения
                nEdLayer = random.randint(0, self.nLayers-2)
                # Идем по слоям сети в обратном направлении, считая силу обратного импульса (wReverseSignalNeuron) и суммируя обучающий импульс по связям (wLinkEd)
                for i in range(self.nLayers - 2, nEdLayer - 1, -1):
                    # Для каждого нейрона предыдущего слоя
                    for j in range(self.neuronCount[i]+1):
                        # Вычисляем поступившую силу сигнала от следующего слоя по обратной связи
                        if i > nEdLayer:
                            signal = 0.0
                            for k in range(self.neuronCount[i+1]):
                                if wReverseSignalNeuron[i+1][k] != 0.0:
                                    signal += wReverseSignalNeuron[i+1][k] * self.wLink[i][j][k]
                            signal = self.activateFunc(signal)
                            wReverseSignalNeuron[i][j] = signal
                        # Запись обучающего импульса
                        if i == nEdLayer:
                            for k in range(self.neuronCount[i+1]):
                                if wReverseSignalNeuron[i+1][k] != 0.0 and self.wSignalNeuron[i][j] != 0.0:
                                    #С пропуском части связей
                                    if random.randint(0,2)==0:
                                        wLinkEd[i][j][k] += wReverseSignalNeuron[i+1][k] * self.wSignalNeuron[i][j]
            if step > -1 and difWinRate < 0:
                break                     
        # Нормируем обучение на каждом слое
        educated = False
        for i in range(len(wLinkEd)):
            sum = 0.0
            for j in range(len(wLinkEd[i])):
                for k in range(len(wLinkEd[i][j])):
                    sum += abs(wLinkEd[i][j][k])
            if sum > 0.0:        
                educated = True
                koef = koefEd*linkLayerNorm/sum
                for j in range(len(wLinkEd[i])):
                    for k in range(len(wLinkEd[i][j])):
                        #if wLinkEd[i][j][k] != 0.0 and (self.wLink[i][j][k] < 0.95 or wLinkEd[i][j][k] * koef < 0) and (self.wLink[i][j][k] > -0.95 or wLinkEd[i][j][k] * koef > 0):
                        if wLinkEd[i][j][k] != 0.0:
                            #self.wLink[i][j][k] += wLinkEd[i][j][k] * koef * (1.0 - abs(self.wLink[i][j][k]))
                            #self.wLink[i][j][k] += wLinkEd[i][j][k] * koef * (1.0 - 2*abs(0.5 - self.wLink[i][j][k]))
                            if wLinkEd[i][j][k] > 0:
                                self.wLink[i][j][k] += wLinkEd[i][j][k] * koef * (1 - self.wKoef) * (1.0 - self.wLink[i][j][k])
                            else:
                                self.wLink[i][j][k] += wLinkEd[i][j][k] * koef * self.wKoef * self.wLink[i][j][k]
                            if self.wLink[i][j][k] > 0.98:
                                self.wLink[i][j][k] = 0.98
                            if self.wLink[i][j][k] < 0.02:
                                self.wLink[i][j][k] = 0.02
                            #if wLinkEd[i][j][k] > 0:
                            #    self.wLink[i][j][k] += wLinkEd[i][j][k] * koef * ((1.0 - self.wLink[i][j][k]) / 2.0)
                            #elif wLinkEd[i][j][k] < 0:
                            #    self.wLink[i][j][k] += wLinkEd[i][j][k] * koef * ((1.0 + self.wLink[i][j][k]) / 2.0)
                            #else:
                            #    self.wLink[i][j][k] += wLinkEd[i][j][k] * koef
        return educated


def checkWin():
    if field[0][0] + field[1][1] + field[2][2] == 3:
        return 1
    if field[0][0] + field[1][1] + field[2][2] == -3:
        return -1
    if field[0][2] + field[1][1] + field[2][0] == 3:
        return 1
    if field[0][2] + field[1][1] + field[2][0] == -3:
        return -1
    for row in field:
        sum = 0
        for val in row:
            sum += val
        if sum == 3:
            return 1
        if sum == -3:
            return -1

    for j in range(fCols):
        sum = 0
        for i in range(fRows):
            sum += field[i][j]
        if sum == 3:
            return 1
        if sum == -3:
            return -1
    return 0


def printField():
    for row in field:
        st = ""
        for val in row:
            st += str(val) + " "
        print(row)

def initField():
    global field, emptyCells
    emptyCells = fRows * fCols
    field = []
    for i in range(fRows):
        row = []
        for j in range(fCols):
            row.append(0)
        field.append(row)

def game(startPlayer, randFactor, difLimit, randomizeStep):
    global neuro, field, emptyCells, winRate
    initField()
    neuro[0].clearSave()
    neuro[1].clearSave()
    #currentPlayer = random.randint(0, 1)
    currentPlayer = startPlayer
    Winner = 0
    while emptyCells > 0:
        field_1 = []
        for row in field:
            for val in row:
                if (currentPlayer == 0):
                    field_1.append(-val)
                else:
                    field_1.append(val)
        choiсe = neuro[currentPlayer].chooseAndSaveMove(field_1, randFactor[currentPlayer], difLimit[currentPlayer],randomizeStep==9-emptyCells)
        #field_1[choiсe] = 2 * currentPlayer - 1
        field[int(choiсe / 3)][choiсe % 3] = 2 * currentPlayer - 1
        emptyCells -= 1
        Winner = checkWin()
        #print("Player " + str(currentPlayer) +
        #      " Step " + str(fRows * fCols - emptyCells))
        #printField()
        if Winner != 0:
            neuro[0].Winner = -Winner
            neuro[1].Winner = Winner

            winRate[int(0.5*(1+Winner))] += 0.4 + 0.35 * int((emptyCells + 1) / 2)
            winRate[int(0.5*(1-Winner))] += -0.2 - 0.4 * int((emptyCells + 1) / 2)
            
            neuro[int(0.5*(1+Winner))].WinRated = 0.4 + 0.35 * int((emptyCells + 1) / 2)
            neuro[int(0.5*(1-Winner))].WinRated = -0.2 - 0.4 * int((emptyCells + 1) / 2)
            #print("Winner player " + str(int((Winner + 1) / 2)))
            #printField()
            break
        currentPlayer = 1 - currentPlayer
    if Winner == 0:
        winRate[currentPlayer] += 0.1
        neuro[currentPlayer].WinRated = 0.1
        neuro[1-currentPlayer].WinRated = 0.0
   

random.seed()
neuro = []
neuro1 = NeuroWeb([])
#neuro1.setTopology([27,27,9])
#neuro1.setRandomWeigth()
#neuro1.saveToFile("NeuroEd_27x27x9_1.txt")
neuro1.loadFromFile("NeuroEd_27x27x9_1.txt")
neuro.append(neuro1)
neuro2 = NeuroWeb([])
#neuro2.setTopology([27,27,9])
#neuro2.setRandomWeigth()
#neuro2.saveToFile("NeuroEd_27x27x9_2.txt")
neuro2.loadFromFile("NeuroEd_27x27x9_2.txt")
neuro.append(neuro2)
winRate=[]
winRate.append(0)
winRate.append(0)
oldWinRate=[]
oldWinRate.append(0)
oldWinRate.append(0)
#print("neuro2:")
#for i in neuro2.neuronCount:
#    print(str(i))
fRows = 3
fCols = 3
emptyCells = fRows * fCols
field = []


#Имитация отжига
"""edNeuro = 1
for i in range(iterCount + 1):
    edNeuro = 1 - edNeuro
    for j in range(100):
        winRate[0] = 0
        winRate[1] = 0
        #for k in range(10):
        game(0)
        #for k in range(10):
        game(1)
        ii = random.randint(0, neuro[edNeuro].nLayers - 2)
        jj = random.randint(0, neuro[edNeuro].neuronCount[ii])
        kk = random.randint(0, neuro[edNeuro].neuronCount[ii+1]-1)
        oldLink = neuro[edNeuro].wLink[ii][jj][kk]
        neuro[edNeuro].wLink[ii][jj][kk] = 2 * random.random() - 1
        oldWinRate=[]
        oldWinRate.append(winRate[0])
        oldWinRate.append(winRate[1])
        sumRate[0] += winRate[0]
        sumRate[1] += winRate[1]
        winRate[0] = 0
        winRate[1] = 0
        #for k in range(10):
        game(0)
        #for k in range(10):
        game(1)
        #Откат при неудаче   
        if winRate[edNeuro] < oldWinRate[edNeuro]:
            neuro[edNeuro].wLink[ii][jj][kk] = oldLink

    out = "WinRate p[0]=" + str(round(sumRate[0],1)) + ";\tWinRate p[1]=" + str(round(sumRate[1],1))
    sumRate[0] = 0
    sumRate[1] = 0
    out += "\t" + str(100 * i / iterCount) + "%"
    print(out)
"""    

iterCount = 10000
sumRate=[]
sumRate.append(0)
sumRate.append(0)
# Первый ходит игрок Х, обучается игрок У
# 1. Проверяем значение выигрыша игрока У при четкой игре, когда первым сходил игрок Х
# 2. Играем типовую игру с отклонением выбора на N - ходе, где N - случайный ход, кроме 9-го.
# 3. Если выигрыш не изменился, то пробуем изменить номер хода и смену ячейки ограниченное число раз (10)
# 3.1. В случае, если выигрыш У изменился - проводим обучение на модифицированном ходе
# 3.2. Увеличиваем веса на случайном слое только, если вероятность выбора хорошего варианта недостаточно велика
# 3.3. При увеличении весов "лучшего варианта" возможно следует уменьшить веса на основном выборе базового
# 3.4. Уменьшаем веса только, если вероятность выбора плохого выбора существенна

#Обратное распространение ошибки
linkCount = []
for n in neuro:
    count = 0
    for i in n.wLink:
        for j in i:
            for k in j:
                count += 1
    linkCount.append(count)            

edPlayer = 0
staticPlayer = 1 - edPlayer
for i in range(0, iterCount + 1):
    if i%100 == 0:
        out = "sumRate p[0]=" + str(round(sumRate[0],1)) + ";\tsumRate p[1]=" + str(round(sumRate[1],1))
        sumRate[0]=0
        sumRate[1]=0

        
        for n in range(len(neuro)):
            sum = 0.0
            for ii in neuro[n].wLink:
                for j in ii:
                    for k in j:
                        sum += k
            neuro[n].wKoef = sum / linkCount[n]          
        
        if i%1000==0:
            out += " " + str(100 * i / iterCount) + "%"
        
        print(out)    

    randFactor=[0.0, 0.0]
    difLimit=[0.01, 0.01]
    for firstPlayer in range(2):
        for edPlayer in range(2):
            #print(str(firstPlayer) + "-" + str(edPlayer))
            #игрок 1 ходит первым
            #сначала проводим четкую игру без особой вариантивности для определения - насколько хороша текущая сеть
            winRate[0] = 0
            winRate[1] = 0
            game(firstPlayer, randFactor, difLimit, -1)
            sumRate[0] += winRate[0]
            sumRate[1] += winRate[1]
            oldWinRate[0] = winRate[0]
            oldWinRate[1] = winRate[1]

            #Попробуем что-то изменить у игрока 1 (Тут он ходит первым)
            baseStepsCount = len(neuro[edPlayer].savedInput)
            stepModify = 0
            needEducate = True
            tryCount = 0
            #Пока не произошло обучения, пытаемся проверить несколько вариантов
            if oldWinRate[edPlayer] < 1.0:
                while needEducate and tryCount < 6:
                    #Пока функция выигрыша не изменилась
                    #winRate[0] = oldWinRate[0]
                    #winRate[1] = oldWinRate[1]
                    #while oldWinRate[edPlayer] == winRate[edPlayer] and tryCount < 10:
                    tryCount += 1
                    winRate[0] = 0
                    winRate[1] = 0
                    #Шаг игры, который будем пробовать модифицировать
                    stepModify = random.randint(0, baseStepsCount-1)
                    game(firstPlayer, randFactor, difLimit, 2*stepModify + (edPlayer - firstPlayer) % 2)
                    if winRate[edPlayer] > oldWinRate[edPlayer]:
                        needEducate = not neuro[edPlayer].education(winRate[edPlayer] - oldWinRate[edPlayer], stepModify)
            while needEducate and tryCount < 12:
                #Пока функция выигрыша не изменилась
                #while oldWinRate[edPlayer] == winRate[edPlayer] and tryCount < 10:
                tryCount += 1
                winRate[0] = 0
                winRate[1] = 0
                #Шаг игры, который будем пробовать модифицировать
                stepModify = random.randint(0, baseStepsCount-1)
                game(firstPlayer, randFactor, difLimit, 2*stepModify + (edPlayer - firstPlayer) % 2)
                if winRate[edPlayer] != oldWinRate[edPlayer]:
                    needEducate = not neuro[edPlayer].education(winRate[edPlayer] - oldWinRate[edPlayer], stepModify)
    


    
    
neuro1.saveToFile("NeuroEd_27x27x9_1.txt")
neuro2.saveToFile("NeuroEd_27x27x9_2.txt")
