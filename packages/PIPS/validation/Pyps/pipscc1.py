import os
from pipscc import pipscc
pipscc(["pipscc","-c" , "basics0.c"]).run()
pipscc(["pipscc","basics0.o", "-o" , "b.out"]).run()
os.remove("b.out")
os.remove("basics0.o")
