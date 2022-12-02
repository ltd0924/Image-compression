class Node(object):
    def __init__(self,name=None,value=None):
        self._name=name
        self._value=value
        self._left=None
        self._right=None

class HuffmanTree(object):
    def __init__(self,char_weights):
        self.Leav=[Node(key,value) for key,value in char_weights.items()]
        while len(self.Leav)!=1:
            self.Leav.sort(key=lambda node:node._value,reverse=True)
            c=Node(value=(self.Leav[-1]._value+self.Leav[-2]._value))
            c._left=self.Leav.pop(-1)
            c._right=self.Leav.pop(-1)
            self.Leav.append(c)
        self.root=self.Leav[0]
        self.Buffer=list(range(100))
        self.get_code()

    def pre(self, tree, length):
        node = tree
        if (not node):
            return
        elif node._name:

            print(node._name,end='')
            print(':\'',end='')
            for i in range(length):
                print(self.Buffer[i], end='')
            print('\',')
            return

        self.Buffer[length] = 0
        self.pre(node._left, length + 1)
        self.Buffer[length] = 1
        self.pre(node._right, length + 1)

    def get_code(self):
        print('{')
        self.pre(self.root, 0)
        print('}')


class rHuffmanTree(object):
    def __init__(self, char_weights):
        self.Leav = [Node(key, value) for key, value in char_weights.items()]
        while len(self.Leav) != 1:
            self.Leav.sort(key=lambda node: node._value, reverse=True)
            c = Node(value=(self.Leav[-1]._value + self.Leav[-2]._value))
            c._left = self.Leav.pop(-1)
            c._right = self.Leav.pop(-1)
            self.Leav.append(c)
        self.root = self.Leav[0]
        self.Buffer = list(range(100))
        self.get_code()

    def pre(self, tree, length):
        node = tree
        if (not node):
            return
        elif node._name:


            for i in range(length):
                print(self.Buffer[i], end='')
            print(':\'', end='')
            print(node._name, end='')

            print('\',')
            return

        self.Buffer[length] = 0
        self.pre(node._left, length + 1)
        self.Buffer[length] = 1
        self.pre(node._right, length + 1)

    def get_code(self):
        print('{')
        self.pre(self.root, 0)
        print('}')

