import copy
import datetime

class Node(object):

    def __init__(self, table, father = None, g = 0):
        self.table = table;
        self.father = father;
        self.g = g;

    def __str__(self):
        string = '';
        for x in range(3):
            for y in range(3):
                string += str(self.table[x][y]) + '';
            string += " \n";
        string += "--------\n";
        return string;

    def __eq__(self, other):
        
        if type(other) != type(self):
            return False;
        else:
            return self.table == other.table;
        
def moveItem(table, srcX, srcY, drcX, drcY):

    tempTable = copy.deepcopy(table);
    tempTable[srcX][srcY], tempTable[drcX][drcY] = tempTable[drcX][drcY], tempTable[srcX][srcY];

    return tempTable;

def checkResolvable(node):
    
    table = node.table;
    y = 0;
    
    tempList = sum(table, []);
    tempList.remove(0);

    for i in range(8):
        for j in range(i):
            if tempList[i] < tempList[j]:
                y += 1;
    return y % 2 == 0;


def getNodeFromOpenList(node, openList):

    if node in openList:
        pos = openList.index(node);
        return openList[pos];
    return None;

def CandidateNodeIsNeedOrNot(node, closeList, openList):

    if node in closeList:
        return;

    if not node in openList:
        openList.append(node);

    else:
        tempNode = getNodeFromOpenList(node, openList);
        if node.g < tempNode.g:
            tempNode.g, tempNode.father = node.g, node.father;
             

def searchNeighbor(currentNode, endNode, closeList, openList):
    
    x = None;
    y = None;

    for row in range(3):
        for col in range(3):
            if currentNode.table[row][col] == 0:
                x = row;
                y = col;
                break;

    GCost = currentNode.g + 1;

    if x - 1 >= 0:
        tempTable = moveItem(currentNode.table , x, y, x - 1, y);
        tempNode = Node(tempTable, currentNode, GCost);

        CandidateNodeIsNeedOrNot(tempNode, closeList, openList);
    if x + 1 <= 2:
        tempTable = moveItem(currentNode.table, x, y, x + 1, y);
        tempNode = Node(tempTable, currentNode, GCost);

        CandidateNodeIsNeedOrNot(tempNode, closeList, openList);
    if y - 1 >= 0:
        tempTable = moveItem(currentNode.table, x, y, x, y - 1);
        tempNode = Node(tempTable, currentNode, GCost);

        CandidateNodeIsNeedOrNot(tempNode, closeList, openList);
    if y + 1 <= 2:
        tempTable = moveItem(currentNode.table, x, y, x, y + 1);
        tempNode = Node(tempTable, currentNode, GCost);

        CandidateNodeIsNeedOrNot(tempNode, closeList, openList);

def IDS_Search(startNode, endNode, interval):

    openList = [];
    closeList = [];
    pathList = [];  
    DepthList = [];     
    currentNode = startNode;

    step = 0;
    depth = interval;

    if checkResolvable(startNode) == False:
        print('Can not be solved !');
        return;

    startNode.g = step;
    openList.append(startNode);

    while True:

        DepthList = [];
        Flag = False;

        while True:

            currentNode = openList[-1];
            openList.pop();
            closeList.append(currentNode);

            if currentNode.table == endNode.table:

                while currentNode != None:
                    pathList.append(currentNode);
                    currentNode = currentNode.father;

                print('It solved!');
                Flag = True;
                break;

            if currentNode.g == depth:
                DepthList.append(currentNode);
            else:
                searchNeighbor(currentNode, endNode, closeList, openList);

            if len(openList) == 0:
                break;

        if Flag == True:
            break;
        else:
            openList = copy.deepcopy(DepthList);

        depth += interval;    

    for val in pathList[::-1]:
        print(val);

def main():

    end = Node([[1,2,3],[4,5,6],[7,8,0]]);

    listB = [[[6, 8, 0], [7, 4, 3], [2, 1, 5]], [[1, 5, 3], [4, 8, 2], [7, 0, 6]], [[2, 1, 7], [0, 6, 3], [8, 4, 5]], [[2, 3, 6], [4, 7, 5], [0, 8, 1]], [[2, 0, 4], [7, 6, 3], [1, 5, 8]], [[6, 7, 8], [2, 0, 3], [4, 5, 1]], [[0, 2, 5], [3, 4, 7], [1, 6, 8]], [[4, 1, 2], [5, 3, 7], [8, 0, 6]], [[1, 3, 6], [0, 8, 2], [4, 7, 5]], [[0, 6, 8], [4, 1, 3], [2, 7, 5]], [[3, 2, 6], [4, 7, 1], [8, 5, 0]], [[5, 4, 3], [6, 8, 0], [7, 2, 1]], [[3, 0, 4], [8, 5, 1], [7, 2, 6]], [[3, 5, 4], [1, 7, 6], [0, 2, 8]], [[8, 4, 0], [2, 5, 1], [7, 3, 6]], [[3, 6, 1], [5, 2, 4], [8, 0, 7]], [[5, 3, 8], [2, 7, 0], [1, 6, 4]], [[7, 1, 8], [2, 4, 0], [5, 6, 3]], [[7, 1, 2], [5, 4, 0], [6, 8, 3]], [[2, 0, 1], [4, 5, 8], [6, 3, 7]]];
    t = datetime.datetime.now()


    for val in listB:
        
        start = Node(val);
        IDS_Search(start, end, 5);

    s = datetime.datetime.now()

    print(str(s - t) + " s")

main(); 
