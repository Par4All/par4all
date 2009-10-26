import pyps
from pipscc import pipscc

class Pii(pipscc):

    def changes(self, ws):
        
        def filter(module):
            return len(module.code()) < 3

        ws.all(filter).inlining()


if __name__ == '__main__':

    thecompiler = Pii()
    thecompiler.run()
