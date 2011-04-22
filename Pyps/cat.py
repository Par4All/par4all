from __future__ import with_statement # this is to work with python2.5
from pyps import *
from subprocess import call
from os import remove
from shutil import rmtree

with workspace("cat.c",deleteOnClose=False) as w:
	w.fun.main.display()
	w.make()
	(binary,a,b,c,)=w.compile()
	call("./"+binary)
	
	w.fun.main.run(["sed","-e",'s/cats/dogs/'])
	w.fun.main.display()
	w.save()
	w.compile(rule="mrproper")
	(binary,rc,out,err) = w.compile()
	call("./"+binary)



