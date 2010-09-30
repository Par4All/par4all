import pyps
import os
import sys
from subprocess import Popen, PIPE

hfile= r"""#ifndef PYPS_GETTIME_H
#define PYPS_GETTIME_H

#define main _pyps_main

#endif //PYPS_GETTIME_H
"""

timefile = "_pips_time.tmp"

cfile= r"""
#include <sys/time.h>
#include <stdio.h>

extern int _pyps_main(int argc, char **argv);

int main (int argc, char **argv)
{
	struct timeval time1, time2;
	int res;
	
	gettimeofday(&time1, NULL);

	res = _pyps_main(argc, argv);
	
	gettimeofday(&time2, NULL);
	
	long diff = (time2.tv_sec-time1.tv_sec)*1000000 + (time2.tv_usec-time1.tv_usec);
	
	FILE *out = fopen("%s", "w");
	if (out) {
		fprintf(out, "%%ld\n", diff);
		fclose(out);
	}

	return res;
}
""" % timefile

""" When going to compile, edit all the c files to add the macros
    allowing us to measure the time taken by the program"""
class workspace:
	def __init__(self, ws, source, *args, **kwargs):
		self.ws = ws

	def pre_goingToRunWith(self, files, outdir):
		"""Editing the source files to add the include in them"""

		""" Creating the file containing the new main """
		gettime_c = os.path.join(outdir, "_pyps_main_gettime.c")
		with open(gettime_c, 'w') as f:
			f.write(cfile)

		""" Creating the file containing the macro """
		gettime_h = "_pyps_main_gettime.h"
		with open(os.path.join(outdir, gettime_h), 'w') as f:
			f.write(hfile)

		for file in files:
			with open(file, 'r') as f:
				read_data = f.read()
			#Don't put the include more than once
			if read_data.find('\n#include "{0}"\n'.format(gettime_h)) != -1:
				continue
			with open(file, 'w') as f:
				f.write('/* Header automatically inserted by PYPS*/\n#include "{0}"\n\n'.format(gettime_h))
				f.write(read_data)
		files.append(gettime_c)

	def post_compile(self, *args, **kwargs):
		try:
			os.unlink("_pyps_main_gettime.c")
			os.unlink("_pyps_main_gettime.h")
		except OSError:
			pass
		outfile = kwargs.get("outfile", self.ws._name)

	def getLastTime(self):
		with open(timefile, "r") as f:
			time = int(f.readline())
		os.unlink(timefile)
		return time

	def benchmark(self, execname, compilemethod = None, CC = "gcc", CFLAGS = "", LDFLAGS = "", args = [], iterations = 1, reference = None):
		"""Compile and run the main() for this workspace.

		execname: basename for the executable file
		compilemethod: the method called for the compilation (self.compile
			by default)
		CC, CFLAGS: as expected
		args: list of strings, used as argv when running the executable
		iterations: average the running time over this number of iterations
		reference: a list of at most one string. When non empty, compare the
			output of the program with this. When empty, place the output of
			the program in reference[0].
		"""
		runtimes = []
		rep = self.ws.name +".database/Tmp"
		outfile = rep + "/" + execname
		if compilemethod is None:
			compilemethod = self.ws.compile

		compilemethod(outfile = outfile, rep = rep, CFLAGS = CFLAGS,
					  CC = CC, LDFLAGS = LDFLAGS)
		cmd = [outfile] + args
		for i in range(0, iterations):
			p = Popen(cmd, stdout = PIPE, stderr = PIPE)
			(out,err) = p.communicate()
			rc = p.returncode
			if rc != 0:
				message = "Program %s failed with return code %d" %(cmd, rc)
				#raise RuntimeError(message)
			time = 0
			try:
				time = self.getLastTime()
			except IOError:
				message  = "cmd: " + str(cmd) + "\n"
				message += "out: " + out + "\n"
				message += "err: " + err + "\n"
				message += "return code: " + str(rc) + "\n"
				raise RuntimeError(message)
			runtimes += [time]
			if reference:
				ref = reference[0]
				if out != ref:
					message  = "cmd: " + str(cmd) + "\n"
					message += "Output for "+str(cmd)+" changed from:\n"
					message += ref
					message += "\nTo:\n"
					message += out
					raise RuntimeError(message)
			else:
				if type(reference).__name__ == "list": reference.append(out)

		runtimes.sort()
		return runtimes[len(runtimes) / 2]
