#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import with_statement # to cope with python2.5

#import pyrops
import pyps, pyrops
import workspace_gettime as gt
import workspace_remote as rt
import workspace_check as ck
#import binary_size as binsize
import memalign
import sac
import sys
import os
import time
import shutil
import bench_cfg
import shlex
import copy

from subprocess import *
from optparse import OptionParser
import re

def dummy_get_maker(c):return pyps.Maker
def sac_get_maker(c): return sac.sacMaker(pyps.Maker,c)

def benchrun(s,calms=None,calibrate_out=None):
	def do_benchmark(ws, wcfg, get_maker, cc_cfg,  args, n_iter, name_custom):
		times = {wcfg.module: [0]}
		benchname = cc_cfg.name() + "+" + name_custom
		ccp = get_maker(cc_cfg.maker)()
		try:
			if doBench:
				times = ws.benchmark(maker=maker,iterations=n_iter,args=args,CC=cc_cfg.cc, CFLAGS=cc_cfg.cflags)
				benchtimes[benchname] = {'time': times[wcfg.module][0], 'makefile': makefile, 'cc_stderr': "ccp.cc_stderr"}
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
	
	def do_calibrate(ws, cc_cfg_ref, args, arg_n, calms, module_name):
		''' Calibrate the args of a workspace using the reference compiler, such as it takes a given running time. '''
		ccp = pyps.ccexecParams(compilemethod=ws.compile,CC=cc_cfg_ref.cc,CFLAGS=cc_cfg_ref.cflags,args=args,outfile=cc_cfg_ref.name()+"+calibrate")
		size_kernel = int(args[arg_n])

		# First, it looks at the time taken by the workspace with the default argument.
		# Then, it computes an approximate value (supposing that the running time varies linearely
		# with the kernel size).
		# And, finally, it makes a dichotomy in order to find a good kernel size approximation.

		# Define a tolerance for the running time (in ms)
		TOLERANCE_MS = 5
		def make_args(size_kernel):
			args_ar_tmp = copy.deepcopy(args)
			args_ar_tmp[arg_n] = str(size_kernel)
			return args_ar_tmp

		def get_run_time(size_kernel):
			''' Return the running time with the current kernel size in ms '''
			ccp.args = make_args(size_kernel)
			times = ws.benchmark(ccexecp=ccp,iterations=5) # Return time in us
			return float(times[module_name][0])/1000.0

		cur_runtime = get_run_time(size_kernel)
		print "Org runtime:",cur_runtime
		# Compute the first approximation of size_kernel
		size_kernel = int(size_kernel * calms/cur_runtime)
		cur_runtime = get_run_time(size_kernel)
		print "Approx runtime:",cur_runtime

		if cur_runtime < calms:
			low_size = size_kernel
			high_size = size_kernel*10
		else:
			high_size = size_kernel
			low_size = min(size_kernel/10,100)
		# Do the dichotomy if useful
		while abs(cur_runtime-calms) > TOLERANCE_MS:
			new_size = (high_size+low_size)/2
			cur_runtime = get_run_time(new_size)
			if cur_runtime < calms:
				if low_size == new_size:
					low_size = new_size/5
				else:
					low_size = new_size
			else:
				if high_size == new_size:
					high_size = new_size*5
				else:
					high_size = new_size
			print "New runtime:",cur_runtime,new_size

		return make_args(size_kernel)

	def binary_size(module, cc_cfg, compile_f):
		ccp = pyps.backendCompiler(compilemethod=compile_f,CC=cc_cfg.cc,CFLAGS=cc_cfg.cflags,args="",outfile=module._name+cc.name())
		return m.binary_size(ccp)

	doBench = s.default_mode=="benchmark"
	doCalibrate = calms != None
	doSize = s.default_mode=="size"
	doValidate = s.default_mode=="validation"
	onlySac = s.default_mode=="sac"
	
	class myworkspace():
		pass

	if s.remoteExec:
		class myworkspace(rt.workspace):
			pass	
	
	class myworkspace(myworkspace,sac.workspace):
		pass
	
	if doBench:
		class myworkspace(myworkspace,gt.workspace):
			pass
	if doValidate:
		class myworkspace(myworkspace,ck.workspace):
			pass
	
	for wcfg in s.workspaces:
		wcfg.load()
		benchtimes = {}
		wssizes[wcfg.name()] = dict()
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
		pyps.workspace.delete(wcfg.name())
		ws=myworkspace(*srcs,
				       driver = s.default_driver,
				       remoteExec = s.remoteExec,
				       cppflags = cflags,
				       deleteOnClose=False,
				       recoverIncludes=True,
				       ccexecp_ref=ccp_ref)
		m = ws[wcfg.module]
		if not onlySac:
			if doBench:
				args = wcfg.args_benchmark
				n_iter = wcfg.iterations_bench
				m.benchmark_module()
			else:
				args = wcfg.args_validate
				n_iter = 1
			args = shlex.split(str(args))

#			if wcfg.memalign:
#				ws.memalign()

			if doCalibrate:
				new_args = do_calibrate(ws, s.cc_reference, args, wcfg.arg_kernel_size, calms, wcfg.module)
				wcfg.args_benchmark = " ".join(new_args)
				continue 

			if doBench:
				do_benchmark(ws, wcfg, dummy_get_maker, s.cc_reference, args, n_iter, "ref")
				for cc in s.ccs_nosac:
					cc.load()
					do_benchmark(ws, wcfg, dummy_get_maker, cc, args, n_iter, "nosac")

			if doSize:
				for cc in s.ccs_nosac:
					wssizes[wcfg.name()][cc.name()+"+nosac"] = binary_size(m, cc, ws.compile)[0]

		sac_gen_path = wcfg.name()+"-sac-generated"
#		if not onlySac and os.path.exists(sac_gen_path):
#			ws.close()
#			files=srcs
#			files=map(lambda s: os.path.join(sac_gen_path,os.path.basename(s)),files)
#			ws=pyps.workspace(*files, parents=wk_parents,
#					driver = s.default_driver,
#					remoteExec = s.remoteExec,
#					cppflags = cflags+" -I"+sac_gen_path,
#					deleteOnClose=True,
#					recoverIncludes=True,
#					ccexecp_ref=ccp_ref)
#			ws[wcfg.name()].benchmark_module()
#		else:
		m.sac()
		if onlySac:
			rep=wcfg.name()+"-sac-generated"
			ws.goingToRunWith(ws.save(rep=rep),rep)
		else:
			if doValidate:
				# If we are in validation mode, validate the generic SIMD implementation thanks
				# to s.cc_reference
				do_benchmark(ws, wcfg, s.cc_reference, args, n_iter, "ref+sac")

			if "ccs_sac" in s:
				for cc in s.ccs_sac:
					cc.load()
					if doSize:
						wssizes[wcfg.name()][cc.name()+"+sac"] = binary_size(m, cc, ws.compile)[0]
					else:
						do_benchmark(ws, wcfg, sac_get_maker, cc, args, n_iter, "sac")
		ws.close()

		wstimes[wcfg.name()] = benchtimes
	
	if doCalibrate:
		bench_cfg.workspaces.save(calibrate_out)
		return


parser = OptionParser(usage = "%prog")
parser.add_option("-m", "--mode", dest = "mode",
				  help = "benchmark mode: validation, benchmark, sac or size")
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
parser.add_option("--calibrate", dest="calms",
				  help = "Calibrate the argument of the workspaces so that they take a given running time (in ms). For benchmark mode only")
parser.add_option("--calibrate-etc-out", dest="calout",
				  help = "Output workspaces configuration file for --calibrate")
parser.add_option("--with-etc-dir", dest="etc_dir",
				  help = "Use this directory for configuration files")
parser.add_option("--with-etc-wk", dest="etc_wk_file",
				  help = "Use this configuration file for workspaces (override the one that will be used by default, or with --with-etc-dir)")
(opts, _) = parser.parse_args()

wstimes = {}
wssizes = {}
errors = []

etc_dir = "etc/"
if opts.etc_dir != None:
	etc_dir = opts.etc_dir
wk_file = None
if opts.etc_wk_file != None:
	wk_file = opts.etc_wk_file

bench_cfg.init(etc_dir=etc_dir,wk_file=wk_file)

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


calms = None
calout = None
if opts.calms:
	calms = int(opts.calms)
	if opts.calout == None:
		raise RuntimeError("--calibrate requires --calibrate-etc-out !")
	calout = opts.calout

benchrun(session,calms,calout)

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
			outfile.write("used makefile: "+res['makefile']+"\n")
			outfile.write("CC stderr output: "+res['cc_stderr']+"\n\n")
		outfile.write("\n")

if session.default_mode == "size":
	outfile.write("\t")
	columns = []
	for cc in session.ccs_nosac: columns.append(cc.name()+"+nosac")
	if "ccs_sac" in session:
		for cc in session.ccs_sac: columns.append(cc.name()+"+sac")
	for c in columns:
		outfile.write(c+"\t")
	outfile.write("\n")

	for wsname, sizetimes in wssizes.iteritems():
		outfile.write(wsname + "\t")
		for benchname in columns:
			if benchname not in sizetimes:
				outfile.write("NA\t")
				continue
			res = sizetimes[benchname]
			outfile.write(str(res))
			outfile.write("\t")
	outfile.write("\n")

if len(errors) > 0:
	outfile.write("\nErrors:\n")
	outfile.write("-------\n\n")
	outfile.write("\n".join(errors))
	exit(1)
