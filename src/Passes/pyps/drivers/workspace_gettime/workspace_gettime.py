from __future__ import with_statement # to cope with python2.5
import pyps
import os
import sys
from subprocess import Popen, PIPE
from workspace_remote import workspace as workspace_rt
import random
import string
import re

hfile= r"""#ifndef PYPS_GETTIME_H
#define PYPS_GETTIME_H

#include <sys/time.h>

void __pyps_bench_start(struct timeval *timestart);
void __pyps_bench_stop(const char* module, const struct timeval *timestart);

#endif //PYPS_GETTIME_H
"""

cfile= r"""
#include <sys/time.h>
#include <stdio.h>

// Warning: these functions aren't thread-safe !!

static FILE *__pyps_timefile = 0;

void __pyps_bench_start(struct timeval *timestart)
{
	gettimeofday(timestart, NULL);
}

void __pyps_bench_stop(const char* module, const struct timeval *timestart)
{
	struct timeval timeend;
	gettimeofday(&timeend, NULL);
	
	long diff = (timeend.tv_sec-timestart->tv_sec)*1000000 + (timeend.tv_usec-timestart->tv_usec);
	
	if (__pyps_timefile == 0)
		__pyps_timefile = fopen("${timefile}", "w");
	if (__pyps_timefile)
	{
		fprintf(__pyps_timefile, "%s: %ld\n", module, diff);
		fflush(__pyps_timefile);
	}
}

void __pyps_bench_close()
{
	if (__pyps_timefile != 0)
		fclose(__pyps_timefile);
}
atexit(__pyps_bench_close);
"""

c_bench_start = r"""
struct timeval __pyps_time_start;
__pyps_bench_start(&__pyps_time_start);
{
"""

c_bench_stop = r"""
}
__pyps_bench_stop("${mn}", &__pyps_time_start);
"""

def benchmark_module(module, **kwargs):
	module.add_pragma(pragma_name='__pyps_benchmark_start', pragma_prepend=True)
	module.add_pragma("__pyps_benchmark_stop_%s" % module.name, pragma_prepend=False)


""" When going to compile, edit all the c files to add the macros
    allowing us to measure the time taken by the program"""
class workspace:
	def __init__(self, ws, source, *args, **kwargs):
		self.ws = ws
		self._timefile = self._gen_timefile_name()
		if "parents" in kwargs and workspace_rt in kwargs["parents"]:
			self.remote = kwargs.get("remoteExec", None)
		else:
			self.remote = None

	def _gen_timefile_name(self):
		bExists = True
		while bExists:
			randstr = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for x in range(10))
			path = os.path.join("/tmp", "pipstime"+randstr)
			bExists = os.path.islink(path) or os.path.exists(path)
		return path

	def post_init(self, sources, **args):
		"""Clean the temporary directory used for holding `SIMD.c'."""
		for m in self.ws:
			m.__class__.benchmark = benchmark_module

	def pre_goingToRunWith(self, files, outdir):
		"""Editing the source files to add the include in them"""

		""" Creating the file containing the functions __pyps_bench_(start|stop) """
		gettime_c = os.path.join(outdir, "_pyps_gettime.c")
		with open(gettime_c, 'w') as f:
			f.write(string.Template(cfile).substitute(timefile=self._timefile))

		""" Creating the file containing the macro """
		gettime_h = "_pyps_gettime.h"
		with open(os.path.join(outdir, gettime_h), 'w') as f:
			f.write(hfile)

		for file in files:
			with open(file, 'r') as f:
				read_data = f.read()
			# Change #pragma __pyps_bench_start by our source code
			read_data = re.sub(r"\#pragma __pyps_benchmark_start", c_bench_start, read_data)
			# Change #pragma __pyps_bench_end_$module by our source code
			read_data = re.sub(r"\#pragma __pyps_benchmark_stop_([a-zA-Z0-9_-]+)",
					lambda m: string.Template(c_bench_stop).substitute(mn=m.group(1)),
					read_data)
			#Don't put the include more than once
			add_include = read_data.find('\n#include "{0}"\n'.format(gettime_h)) == -1;
			with open(file, 'w') as f:
				if add_include:
						f.write('/* Header automatically inserted by PYPS*/\n#include "{0}"\n\n'.format(gettime_h))
				f.write(read_data)
		files.append(gettime_c)

	def post_compile(self, *args, **kwargs):
		try:
			os.unlink("_pyps_gettime.c")
			os.unlink("_pyps_gettime.h")
		except OSError:
			pass
		outfile = kwargs.get("outfile", self.ws._name)

	def _get_timefile_and_parse(self):
		if self.remote != None:
			self.remote.copyRemote(self._timefile,self._timefile)
		with open(self._timefile, "r") as f:
			rtimes = f.readlines()
		reTime = re.compile(r"^(.*): *([0-9]+)$")
		nmodule = dict()
		for l in rtimes:
			ma = reTime.match(l)
			if ma != None:
				mod = ma.group(1)
				if mod not in nmodule:
					nmodule[mod] = 0
				t = int(ma.group(2))
				if mod not in self._module_rtimes:
					self._module_rtimes[mod] = [list()]
				if nmodule[mod] >= len(self._module_rtimes[mod]):
					self._module_rtimes[mod].append(list())
				self._module_rtimes[mod][nmodule[mod]].append(t)
				nmodule[mod] = nmodule[mod]+1

	def getTimesModule(self, module):
		if not self._final_runtimes:
			raise RuntimeError("self.benchmark() must have been run !")
		return self._final_runtimes[module.name]

	def benchmark(self, execname, ccexecp, iterations = 1):
		ccexecp.rep = self.ws.name +".database/Tmp"
		ccexecp.outfile = ccexecp.rep + "/" + execname
		
		self.ws.compile_and_run(ccexecp)

		self._module_rtimes = dict()
		for i in range(0, iterations):
			print >>sys.stderr, "Launch execution of %s..." % execname
			rc,out,err = self.ws.run_output(ccexecp)
			print >>sys.stderr, "Program done."
			if rc != 0:
				message = "Program %s failed with return code %d.\nOutput:\n%s\nstderr:\n%s\n" %(str(ccexecp.cmd), rc, out,err)
				raise RuntimeError(message)
			time = 0
			try:
				self._get_timefile_and_parse()
			except IOError:
				message = "cmd: " + str(ccexecp.cmd) + "\n"
				message += "out: " + out + "\n"
				message += "err: " + err + "\n"
				message += "return code: " + str(rc) + "\n"
				raise RuntimeError(message)

		self._final_runtimes = dict()
		for module,execs in self._module_rtimes.iteritems():
			self._final_runtimes[module] = list()
			for times in execs:
				times.sort()
				self._final_runtimes[module].append(times[len(times)/2])
		return self._final_runtimes
