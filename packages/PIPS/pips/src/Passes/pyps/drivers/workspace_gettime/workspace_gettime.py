from __future__ import with_statement # to cope with python2.5
import pyps
import pypsutils
import os
import sys
from subprocess import Popen, PIPE
from workspace_remote import workspace as workspace_rt
import random
import string
import re
import tempfile
import shutil

pyps_gettime_c = "pyps_gettime.c"
pyps_gettime_h = "pyps_gettime.h"

c_bench_start = r"""
struct timeval __pyps_time_start;
__pyps_bench_start(&__pyps_time_start);
"""

c_bench_stop = r"""
__pyps_bench_stop("${mn}", &__pyps_time_start);
"""

def benchmark_module(module, **kwargs):
	module.add_pragma(pragma_name='__pyps_benchmark_start', pragma_prepend=True)
	module.add_pragma("__pyps_benchmark_stop_%s" % module.name, pragma_prepend=False)

pyps.module.benchmark_module=benchmark_module


""" When going to compile, edit all the c files to add the macros
    allowing us to measure the time taken by the program"""
class workspace(pyps.workspace):
	def __init__(self, *sources, **kwargs):
		self._timefile = tempfile.mkstemp()[1]
		
		#if workspace_rt in kwargs["parents"]:
		#	self.remote = kwargs.get("remoteExec", None)
		#else:
		self.remote = kwargs.get("remoteExec", None)
		kwargs['cppflags'] = kwargs.get('cppflags',"")+' -DPYPS_TIME_FILE=\\"'+self._timefile+'\\"'
		super(workspace,self).__init__(*sources, **kwargs)

	def save(self, rep=None):
		if rep == None:
			rep=self.tmpdirname
		
		(files,headers) = super(workspace,self).save(rep)
		shutil.copy(pypsutils.get_runtimefile(pyps_gettime_c,"pyps_gettime"),rep)
		shutil.copy(pypsutils.get_runtimefile(pyps_gettime_h,"pyps_gettime"),rep)

		
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
				add_include = read_data.find('\n#include "'+pyps_gettime_h+'"\n') == -1;
			with open(file, 'w') as f:
				if add_include:
						f.write('/* Header automatically inserted by PYPS*/\n#include "'+pyps_gettime_h+'"\n\n')
				f.write(read_data)
		files.append(os.path.join(rep,pyps_gettime_c))
		
		return files,headers+[os.path.join(rep,pyps_gettime_h)]

	def _get_timefile_and_parse(self):
		if self.remote:
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

	def benchmark(self, maker=pyps.Maker(), iterations = 1, args=[],**opt):
		rep = self.tmpdirname
		outfile = self.compile(rep=rep,maker=maker,rule="mrproper",**opt)
		outfile = self.compile(rep=rep,maker=maker,**opt)
		
		self._module_rtimes = dict()
		for i in range(0, iterations):
			print >>sys.stderr, "Launch execution of %s %s..." % (outfile," ".join(map(str,args)))
			rc,out,err = self.run(outfile,args)
			if rc != 0:
				message = "Program %s failed with return code %d.\nOutput:\n%s\nstderr:\n%s\n" %(outfile+" ".join(args), rc, out,err)
				raise RuntimeError(message)
			try:
				self._get_timefile_and_parse()
			except IOError:
				message = "command: " + outfile + " ".join(args)+"\n"
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
