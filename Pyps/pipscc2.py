import os
from pipscc import pipscc
pipscc(["pipscc","-c" , "basics0.c", "-o" , "/tmp/bb.o" ]).run()
pipscc(["pipscc","/tmp/bb.o", "-o" , "d.out"]).run()
os.remove("d.out")
os.remove("/tmp/bb.o")
