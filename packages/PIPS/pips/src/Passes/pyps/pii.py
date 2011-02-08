import pyps
from pipscc import pipscc

class Pii(pipscc):

    def changes(self, ws):
        
        def filter(module):
            return module.code.count("\n") < 3

        ws.filter(filter).inlining()


if __name__ == '__main__':

    thecompiler = Pii(sys.argv)
    thecompiler.run()
