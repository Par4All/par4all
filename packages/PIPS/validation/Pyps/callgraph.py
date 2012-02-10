from pyps import workspace

class CallgraphPrinter:
    def __init__(self):
        self.indent=0

    def visit(self, module):
        print " "*self.indent + module.name
        self.indent+=4
        [ self.visit(n) for n in module.callees ]
        self.indent-=4

with workspace("silber.c", verbose=False) as w :
    cp = CallgraphPrinter()
    cp.visit(w.fun.main)
