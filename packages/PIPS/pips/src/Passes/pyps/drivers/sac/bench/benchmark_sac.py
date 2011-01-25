#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import with_statement # to cope with python2.5

#import pyrops
import pyps
import workspace_gettime as gt
import workspace_remote as rt
import workspace_check as ck
import memalign
import sac
import sys
import os
import time
import shutil
import bench_cfg
import shlex

from subprocess import *
from optparse import OptionParser
import re

def benchrun(s):
	def do_benchmark(ws, wcfg, cc_cfg, compile_f, args, n_iter, name_custom):
		times = {wcfg.module: [0]}
		benchname = cc_cfg.name() + "+" + name_custom
		ccp = pyps.ccexecParams(compilemethod=compile_f,CC=cc_cfg.cc,CFLAGS=cc_cfg.cflags,args=args)
		try:
			if doBench:
				times = ws.benchmark(execname=benchname,ccexecp=ccp,iterations=n_iter)
				benchtimes[benchname] = {'time': times[wcfg.module][0], 'cc_cmd': ccp.cc_cmd, 'cc_stderr': ccp.cc_stderr}
			else:
				good,out = ws.check_output(ccexecp=ccp)
				if not good:
					msg = "Validation case %s-%s failed !" % (wcfg.name(),benchname)
					errors.append(msg)
					print >>sys.stderr, msg
					if opts.strict: raise RuntimeError(msg)
		except RuntimeError, e:
			errors.append("Benchmark: %s\n%s" % (benchname, str(e)))
			print >> sys.stderr, e
			if opts.strict: raise

	doBench = s.default_mode=="benchmark"
	wk_parents = [sac.workspace,memalign.workspace]
	if doBench:
		wk_parents.append(gt.workspace)
	else:
		wk_parents.append(ck.workspace)
	if s.remoteExec:
		wk_parents.append(rt.workspace)
	for wcfg in s.workspaces:
		wcfg.load()
		benchtimes = {}
		srcs = map(lambda s: str(s), wcfg.files)
		wcfg.module = str(wcfg.module)
		if doBench:
			cflags = "-D__PYPS_SAC_BENCHMARK "
		else:
			cflags = "-D__PYPS_SAC_VALIDATE "
		if "include_dirs" in wcfg:
			cflags += "-I" +  str(" -I".join(wcfg.include_dirs))
		s.cc_reference.load()
		ccp_ref=None
		if not doBench:
			args = shlex.split(str(wcfg.args_validate))
			ccp_ref = pyps.ccexecParams(CC=s.cc_reference.cc,CFLAGS=s.cc_reference.cflags,args=args)
		with pyps.workspace(*srcs,
				       parents = wk_parents,
				       driver = s.default_driver,
				       remoteExec = s.remoteExec,
				       cppflags = cflags,
				       deleteOnClose=False,
				       recoverIncludes=False,
				       ccexecp_ref=ccp_ref) as ws:
			m = ws[wcfg.module]
			if doBench:
				args = wcfg.args_benchmark
				n_iter = wcfg.iterations_bench
				m.benchmark()
			else:
				args = wcfg.args_validate
				n_iter = 1
			args = shlex.split(str(args))

			if wcfg.memalign:
				ws.memalign()
			if doBench:
				do_benchmark(ws, wcfg, s.cc_reference, ws.compile, args, n_iter, "ref")
				for cc in s.ccs_nosac:
					cc.load()
					do_benchmark(ws, wcfg, cc, ws.compile, args, n_iter, "nosac")

			if "ccs_sac" in s:
				m.sac()
				for cc in s.ccs_sac:
					cc.load()
					do_benchmark(ws, wcfg, cc, ws.simd_compile, args, n_iter, "sac")

			if not doBench:
				if "ccs_sac" not in s:
					m.sac()
				# If we are in validation mode, validate the generic SIMD implementation thanks
				# to s.cc_reference
				do_benchmark(ws, wcfg, s.cc_reference, ws.compile, args, n_iter, "ref+sac")

		wstimes[wcfg.name()] = benchtimes


parser = OptionParser(usage = "%prog")
parser.add_option("-m", "--mode", dest = "mode",
				  help = "benchmark mode: validation or benchmark")
parser.add_option("-s", "--session", dest = "session_name",
				  help = "session to use (defined in sessions.cfg")
parser.add_option("-d", "--driver", dest = "driver",
				  help = "driver to use (avx|sac|3dnow|neon)")
parser.add_option("--cflags", "--CFLAGS", dest = "cflags",
				  help = "additionnal CFLAGS for all compilations")
parser.add_option("--normalize", dest = "normalize", action = "store_true",
				  default = False, help = "normalize timing results")
parser.add_option("--remote-host", dest = "remoteHost",
				  help = "compile and execute sources on a remote machine (using SSH)")
parser.add_option("--control-master-path", dest = "controlMasterPath",
				  help = "path to the SSH control master (if wanted) [optional]")
parser.add_option("--remote-working-directory", dest = "remoteDir",
				  help = "path to the remote directory that will be used")
parser.add_option("--outfile", dest="outfile",
				  help = "write the results into a file [default=stdout]")
parser.add_option("--strict", dest="strict", action="store_true",
		help = "stop the program as soon as an exception occurs.")
(opts, _) = parser.parse_args()

wstimes = {}
errors = []

session = bench_cfg.sessions.get(opts.session_name)
session.load()

if opts.remoteHost:
	if opts.remoteDir == None:
		raise RuntimeError("--remote-working-directory option is required !")
	session.remoteExec = rt.remoteExec(host=opts.remoteHost, controlMasterPath=opts.controlMasterPath, remoteDir=opts.remoteDir)
elif "default_remote" in session:
	session.default_remote.load()
	session.remoteExec = rt.remoteExec(host=session.default_remote.host, controlMasterPath=session.default_remote.control_path, remoteDir=session.default_remote.remote_working_dir)
else:
	session.remoteExec = False

if opts.driver:
	session.default_driver = opts.driver

if opts.mode:
	session.default_mode = opts.mode

benchrun(session)

if opts.outfile:
	outfile = open(opts.outfile, "w")
else:
	outfile = sys.stdout

if session.default_mode == "benchmark":
	outfile.write("\t")
	cc_ref_name = session.cc_reference.name()+"+ref"
	columns = [cc_ref_name]
	for cc in session.ccs_nosac: columns.append(cc.name()+"+nosac")
	if "ccs_sac" in session:
		for cc in session.ccs_sac: columns.append(cc.name()+"+sac")
	for c in columns:
		outfile.write(c+"\t")
	outfile.write("\n")
	for wsname, benchtimes in wstimes.iteritems():
		outfile.write(wsname + "\t")
		if cc_ref_name not in benchtimes:
			print >>sys.stderr, "Warning: reference compilater %s not computed. Normalisation disabled." % cc_ref_name
			opts.normalize = False

		for benchname in columns:
			if benchname not in benchtimes:
				outfile.write("NA\t")
				continue
			res = benchtimes[benchname]
			bencht = res['time']
			if opts.normalize:
				bencht = float(benchtimes[cc_ref_name]['time']) / float(bencht)

			outfile.write(str(bencht))
			outfile.write("\t")
		if opts.normalize:
			outfile.write("("+str(float(benchtimes[cc_ref_name]['time']))+")")
		outfile.write("\n")
	outfile.write("\n")
	
	for wsname, benchtimes in wstimes.iteritems():
		outfile.write("Details for workspace %s:\n" % wsname)
		outfile.write("---------------------\n")
		for benchname, res in benchtimes.iteritems():
			outfile.write("Compiler configuration: "+benchname+"\n")
			outfile.write("CC command: "+res['cc_cmd']+"\n")
			outfile.write("CC stderr output: "+res['cc_stderr']+"\n\n")
		outfile.write("\n")



if len(errors) > 0:
	outfile.write("\nErrors:\n")
	outfile.write("-------\n\n")
	outfile.write("\n".join(errors))
	exit(1)
