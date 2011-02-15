import subprocess
import re
import pyps
import new

def funcToMethod(func, clas):
	"""Internal helper function : Adds function func to class clas"""
	method = new.instancemethod(func, None, clas)
	clas.__dict__[func.__name__] = method

class workspace:
	"""Workspace extension to provide a binary_size function to each module.

	A call to the binary_size function of a given module attempts to compile the program and then to extract the compiled size and the instruction count of the given module.  The compiled size and the instruction count are returned by the functon in a tuple. A ValueError exception is thrown if the function is not found in the binary (see below).

	Be carefull. The symbol used by the compiler in the binary object for the given module must be guessed by the binary_size function. Given a function “foo”, the symbol can be “foo”, “foo.”, “_foo” or many others forms, thus we cannot ensure that this function will work in every situations. A wrong guess can lead to a ValueError exception or in a few cases to wrongs results.

	def __init__(self, ws, **args):
		self.ws = ws
		funcToMethod(binary_size, pyps.module)

def binary_size(module, func):
	ccexecp = pyps.ccexecParams()
	outfile = module._ws.compile(ccexecp)
	
	return __funcsize(func, outfile)


def __getLinePos(line):
	m = re.match(' *([0-9a-f]+):',line)
	return int(line[m.start(1):m.end(1)], 16)

def __matchSymbol(symbol, dump):
	return re.search("\n[0-9a-f]+ <" + symbol + ">:\n", dump)

def __funcsize(func, outfile = ""):
	args = ['objdump','-S',outfile]
	objdump = subprocess.Popen(args, stdout=subprocess.PIPE)
	dump = objdump.stdout.read()


	symbol = func + '\\.'
	m = __matchSymbol(symbol, dump)
	if m == None:
		m = __matchSymbol(func, dump)
		if m == None:
			raise ValueError("Function "+func+" not found in output file "+outfile) 
	

	fdump = dump[m.end():len(dump)-1]
	m = re.search('\n\n', fdump)
	fdump = fdump[0:m.start()]

	lines = fdump.split('\n')

	lineCount = len(lines)
	size = __getLinePos(lines[len(lines)-1]) - __getLinePos(lines[0])
	return (size, lineCount)

