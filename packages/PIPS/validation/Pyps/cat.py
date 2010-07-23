from pyps import *
from subprocess import call
from os import remove
from shutil import rmtree

w = workspace(["cat.c"])
w["main"].display()
binary=w.compile(outdir="toto")
call("./"+binary)
rmtree("toto")

w["main"].run(["sed","-e",'s/cats/dogs/'])
w["main"].display()
binary=w.compile(outdir="toto")
call("./"+binary)
rmtree("toto")



