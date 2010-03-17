import pyps
from pipscc import pipscc

class Pii(pipscc):

    def changes(self, ws):
        
        def thefilter(module):
            return len(module.code()) < 3

        ws.filter(thefilter).inlining()


if __name__ == '__main__':

    thecompiler = Pii()
    thecompiler.run()
