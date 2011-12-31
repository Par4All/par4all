import os
import pii

pii.Pii(["pii","basics0.c"]).run()
os.remove("a.out")
os.remove("basics0.o")
