from pyps import *
from subprocess import call
from os import remove

w = workspace(["cat.c"])
binary=w.compile(outfile="toto")
call("./"+binary)
remove(binary)

w["main"].run(["sed","-e",'s/cats/dogs/'])
w["main"].display()
binary=w.compile(outfile="toto2")
call("./"+binary)
remove(binary)



