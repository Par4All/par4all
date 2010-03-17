import os
from pipscc import pipscc
pipscc(["pipscc","basics0.c"]).run()
os.remove("a.out")
os.remove("basics0.o")
