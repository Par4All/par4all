# -*- coding: utf-8 -*-
import os,sys
import re
import tempfile
import shutil
import pypsutils
import fileinput
import subprocess
import pyps
from subprocess import Popen, PIPE

simd_c = "SIMD.c"
simd_h = "SIMD_types.h"
sse_h = "sse.h"
threednow_h = "3dnow.h"
avx_h = "avx.h"
neon_h = "neon.h"
patterns_c = "patterns.c"
patterns_h = "patterns.h"
curr_sse_h=sse_h

def gen_simd_zeros(code):
	""" This function will match the pattern SIMD_ZERO*_* + SIMD_LOAD_*
	and replaces it by the real corresponding SIMD_ZERO function. """
	pattern=r'(SIMD_LOAD_V([4 8])SF\(vec(.*), &(RED[0-9]+)\[0\]\);)'
	compiled_pattern = re.compile(pattern)
	occurences = re.findall(compiled_pattern,code)
	if occurences != []: 
		for item in occurences:
			code = re.sub(item[3]+"\[[0-"+item[1]+"]\] = (.*);\n","",code)
			code = re.sub(re.escape(item[0]),"SIMD_ZERO_V"+item[1]+"SF(vec"+item[2]+");",code)
	return code

def autotile(m,verb):
	''' Function that autotile a module's loops '''
	#m.rice_all_dependence()
	#m.internalize_parallel_code()
	#m.nest_parallelization()
	#m.internalize_parallel_code()
	m.split_update_operator()
	def tile_or_dive(m,loops):
		kernels=list()
		for l in loops:
			if l.loops():
				try:
					l.simdizer_auto_tile()
					kernels.append(l)
				except:
					kernels+=tile_or_dive(m,l.loops())
			else:
				kernels.append(l)
		return kernels
	kernels=tile_or_dive(m,m.loops())
	extram=list()
	for l in kernels:
		mn=m.name+"_"+l.label
		m.outline(module_name=mn,label=l.label)
		lm=m._ws[mn]
		extram.append(lm)
		if lm.loops() and lm.loops()[0].loops():
			lm.loop_nest_unswitching()
			if verb:
				lm.display()
			lm.suppress_dead_code()
			if verb:
				lm.display()
			lm.loop_normalize(one_increment=True,skip_index_side_effect=True)
			lm.partial_eval()
			lm.partial_eval()
			lm.partial_eval()
			lm.flatten_code()
			if verb:
				lm.display()
		else:
			lm.loops()[0].loop_auto_unroll()

	if verb:
		m.display()
	extram.append(m)
	return extram

class sacbase(object):
	@staticmethod
	def sac(module, **cond):		
		ws = module._ws
		if not cond.has_key("verbose"):
			cond["verbose"] = ws.verbose
		# Here are the transformations made by benchmark.tpips.h, blindy
		# translated in pyps.

		ws.activate("preconditions_intra")
		ws.activate("transformers_intra_fast")

		ws.props.loop_unroll_with_prologue = False
		ws.props.constant_path_effects = False
		#ws.props.ricedg_statistics_all_arrays = True
		ws.props.c89_code_generation = True


		ws.props.simd_fortran_mem_organisation = False
		ws.props.sac_simd_register_width = cond["register_width"]
		ws.props.prettyprint_all_declarations = True
		ws.props.compute_all_dependences = True
		module.forward_substitute()

		if cond.get("verbose"):
			module.display()
		module.split_update_operator()

		if cond.get("if_conversion", False):
			if cond.get("verbose"):
				module.display()
			module.if_conversion_init()
			module.if_conversion()
			module.if_conversion_compact()
			if cond.get("verbose"):
				module.display()


		ws.activate("MUST_REGIONS")
		ws.activate("REGION_CHAINS")
		ws.activate("RICE_REGIONS_DEPENDENCE_GRAPH")
		ws.activate("PRECONDITIONS_INTER_FULL")
		ws.activate("TRANSFORMERS_INTER_FULL")

		# Perform auto-loop tiling
		allm=autotile(module,cond.get("verbose"))
		for module in allm:
		
			module.simd_remove_reductions()
			if cond.get("verbose"):
				module.display()
			module.flatten_code(unroll = False)
			module.partial_eval()
		if cond.get("verbose"):
			module.display()
		module.clean_declarations()
		if cond.get("suppress_dead_code", True):
			module.suppress_dead_code()
		# module.print_dot_dependence_graph()
		if cond.get("enhanced_reduction",False):
			module.simd_remove_reductions(prelude="SIMD_ZERO_V4SF")
		else:
			module.simd_remove_reductions()
		if cond.get("verbose"):
			module.display()

		for p in ( "__PIPS_SAC_MULADD" , ):
			module.expression_substitution(pattern=p)
			if cond.get("verbose"):
				module.display()

			module.scalar_renaming()

			try:
				module.simdizer(generate_data_transfers=True)
			except Exception,e:
				print >>sys.stderr, "Module %s simdizer exeception:",str(e)

			if cond.get("verbose"):
				#module.print_dot_dependence_graph()
				module.display()

			module.redundant_load_store_elimination()

		if cond.get("verbose"):
			module.display()
		try:
			module.print_dot_dependence_graph()
			module.delay_communications()
			module.flatten_code(unroll = False)
		except RuntimeError: pass

			if cond.get("verbose"):
				module.display()


	@staticmethod
	def addintrinsics(fname, header, replacements):
		finput = fileinput.FileInput([fname], inplace = True)
		for line in finput:
			if finput.isfirstline():
				print header
			for pattern, repl in replacements:
				line = re.sub(pattern, repl, line)
			print line,

class sacsse(sacbase):
	register_width = 128
	hfile = sse_h
	@staticmethod
	def sac(module, **kwargs):
		global curr_sse_h
		kwargs["register_width"] = sacsse.register_width
		sacbase.sac(module, **kwargs)
		curr_sse_h=sse_h

	@staticmethod
	def addintrinsics(fname):
		global curr_sse_h
		replacements = [
				# drop the alignement attribute on the __m128 registers
				(r"(v4s[if]_[^[]*\[[^]]*?\]) __attribute__ \(\(aligned \(\d+\)\)\)", r"\1"),
				# drop the v4s[if] prefix from the declaration; use the type __m128
				# instead of float[4]
				(r"float (v4sf_[^[]+)", r"__m128 \1"),
				(r"int (v4si_[^[]+)", r"__m128i \1"),
				(r"double (v2df_[^[]+)", r"__m128d \1"),
				(r"double (v2di_[^[]+)", r"__m128i \1"),
				# drop the v4s[if] prefix for the usage
				(r"v4s[if]_([^,[]+)\[[^]]*\]", r"\1"),
				(r"v4s[if]_([^ ,[]+)", r"\1"),
				(r"v2d[if]_([^,[]+)\[[^]]*\]", r"\1"),
				(r"v2d[if]_([^ ,[]+)", r"\1"),
				]
		sacbase.addintrinsics(fname, curr_sse_h, replacements)

	@staticmethod
	def post_memalign(*args, **kwargs):
		# XXX Adrien : looks like a hack to see the benefits of alignement
		# (SAC does not seem to use aligned store/load instruction after the
		# use of memalign, and even if posix_memalign is already used)
		global curr_sse_h
		curr_sse_h = re.sub("_mm_loadu", "_mm_load", sse_h)
		curr_sse_h = re.sub("_mm_storeu", "_mm_store", sse_h)

	CFLAGS = "-msse4.2 -march=native -O3"

class sac3dnow(sacbase):
	register_width = 64
	hfile = threednow_h
	@staticmethod
	def sac(module, *args, **kwargs):
		kwargs["register_width"] = sac3dnow.register_width
		# 3dnow supports only floats
		for line in module.code():
			if re.search("double", line) or re.search(r"\b(cos|sin)\b", line):
				raise RuntimeError("Can't vectorize double operations with 3DNow!")
		sacbase.sac(module, *args, **kwargs)

	@staticmethod
	def addintrinsics(fname):
		replacements = [
				# drop the alignement attribute on the __m64 registers
				(r"(v2sf_[^[]*\[[^]]*?\]) __attribute__ \(\(aligned \(\d+\)\)\)", r"\1"),
				# drop the v2sf prefix from the declaration; use the type __m64
				# instead of float[2]
				(r"float (v2sf_[^[]+)", r"__m64 \1"),
				# drop the v2sf prefix for the usage
				(r"v2sf_([^,[]+)\[[^]]*\]", r"\1"),
				(r"v2sf_([^ ,[]+)", r"\1"),
				]
		sacbase.addintrinsics(fname, Threednow_h, replacements)

	@staticmethod
	def post_memalign(*args, **kwargs):
		pass

	CFLAGS = "-m3dnow -march=opteron -O3"

class sacavx(sacbase):
	register_width = 256
	hfile = avx_h
	@staticmethod
	def sac(module, *args, **kwargs):
		kwargs["register_width"] = sacavx.register_width
		sacbase.sac(module, *args, **kwargs)

	@staticmethod
	def addintrinsics(fname):
		replacements = [
			]
		sacbase.addintrinsics(fname, avx_h, replacements)

	@staticmethod
	def post_memalign(*args, **kwargs):
		pass

	CFLAGS = "-mavx -O3"

class sacneon(sacbase):
	register_width = 128
	hfile = neon_h
	@staticmethod
	def sac(module, *args, **kwargs):
		kwargs["register_width"] = sacneon.register_width
		sacbase.sac(module, *args, **kwargs)

	@staticmethod
	def addintrinsics(fname):
		replacements = [
			]
		sacbase.addintrinsics(fname, neon_h, replacements)

	@staticmethod
	def post_memalign(*args, **kwargs):
		pass

	CFLAGS = "-mfpu=neon -mfloat-abi=softfp -O3"

class workspace(pyps.workspace):
	"""The SAC subsystem, in Python.

	Add a new transformation, for adapting code to SIMD instruction
	sets (SSE, 3Dnow, AVX and ARM NEON)"""
	def __init__(self, *sources, **kwargs):
		drivers = {"sse": sacsse, "3dnow": sac3dnow, "avx": sacavx, "neon": sacneon}
		self.driver = drivers[kwargs.get("driver", "sse")]
		#Warning: this patches every modules, not only those of this worspace 
		pyps.module.sac=self.driver.sac
		# Add -DRWBITS=self.driver.register_width to the cppflags of the workspace
		kwargs['cppflags'] = kwargs.get('cppflags',"")+" -DRWBITS=%d " % (self.driver.register_width)
		self.use_generic_simd = True
		super(workspace,self).__init__(pypsutils.get_runtimefile(simd_c,"sac"), pypsutils.get_runtimefile(patterns_c,"sac"), *sources, **kwargs)

	def post_init(self, sources, **args):
		"""Clean the temporary directory used for holding 'SIMD.c' and 'patterns.c'."""
		shutil.rmtree(self.tmpdir)
		for m in self.ws:
			m.__class__.sac = self.driver.sac

	def save(self, rep=None):
		"""Add $driver.h, which replaces general purpose SIMD instructions
		with machine-specific ones."""
		files = super(workspace,self).save(rep)
		

		#run gen_simd_zeros on every file
		for file in files:
			with open(file, 'r') as f:
				read_data = f.read()
			read_data = gen_simd_zeros(read_data)
			with open(file, 'w') as f:
			    f.write(read_data)
		
		if not self.use_generic_simd:
			# We need to add $driver.h to the compiled files, and to remove SIMD.c
			# (ICC breaks quite badly when using the sequential versions mixed
			# with SSE intrinsics).
			newfiles = []
			for fname in files:
				if not re.search(r"SIMD\.c$", fname):
					self.driver.addintrinsics(fname)
					newfiles.append(fname)
			files[:] = []
			files.extend(newfiles)
		else:
			# Generate SIMD.h according to the register width
			# thanks to gcc -E and cproto (ugly, need something
			# better)
			simd_h_fname = os.path.abspath(rep + "/SIMD.h")
			simd_c_fname = os.path.abspath(rep + "/SIMD.c")
			p = subprocess.Popen("gcc -DRWBITS=%d -E %s |cproto" % (self.driver.register_width, simd_c_fname), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
			(simd_cus_header,serr) = p.communicate()
			if p.returncode != 0:
				raise RuntimeError("Error while creating SIMD.h: command returned %d.\nstdout:\n%s\nstderr:\n%s\n" % (p.returncode, simd_cus_header, serr))

			p = subprocess.Popen("gcc -DRWBITS=%d -E %s |cproto" % (self.driver.register_width, simd_c), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
			(simdz_cus_header,serr) = p.communicate()
			if p.returncode != 0:
				raise RuntimeError("Error while creating SIMD.h: command returned %d.\nstdout:\n%s\nstderr:\n%s\n" % (p.returncode, simd_cus_header, serr))
			
			pypsutils.string2file('#include "'+simd_h+'"\n'+simd_cus_header, simd_h_fname)
			pypsutils.string2file(simd_h+"\n"+simdz_cus_header, simd_h_fname)

			for fname in files:
				if not fname.endswith("SIMD.c"):
					pypsutils.addBeginnning(fname, '#include "'+simd_h+'"')

		# Add the contents of patterns.h
		for fname in files:
			if not fname.endswith("patterns.c"):
				pypsutils.addBeginnning(fname, '#include "'+patterns_h+'"\n')
		# Add header to the save rep
		shutil.copy(pypsutils.get_runtimefile(simd_h,"sac"),rep)
		shutil.copy(pypsutils.get_runtimefile(patterns_h,"sac"),rep)
		return files

	def post_memalign(self, *args, **kwargs):
		self.driver.post_memalign(self, *args, **kwargs)

	def get_sacCompiler(self,backendCompiler):
		"""Calls sacCompiler to return a compiler class using the driver set in the workspace"""
		return sacCompiler(backendCompiler,self.driver)

def sacCompiler(backendCompiler,driver):
	"""Returns a compiler class inheriting from the backendCompiler class given in the arguments and using the driver given in the arguments"""
	class C(backendCompiler):
		"""compiler class inheriting from backendCompiler and using its own compile method to comply with the sac driver"""
		def __init__(self,CC="cc", CFLAGS="", LDFLAGS="", compilemethod=None, rep=None, outfile="", args=[], extrafiles=[]):
			super(C,self).__init__(CC, " ".join([CFLAGS,driver.CFLAGS]),LDFLAGS,compilemethod,rep,outfile,args,extrafiles)

		def compile(self, filename, extraCFLAGS="", verbose=False):
			filepath = os.path.dirname(filename)
			filetruename = os.path.basename(filename)
			#change the includes
			filestring = pypsutils.file2string(filename)
			filestring= re.sub('#include "SIMD.h"','#include "'+driver.hfile+'"',filestring)
			newcfile = os.path.join(filepath,"sac_"+filetruename)
			pypsutils.string2file(filestring,newcfile)
			#create symlink .h file
			linkpath = os.path.join(filepath,driver.hfile)
			if not os.path.exists(linkpath):
				os.symlink(pypsutils.get_runtimefile(driver.hfile,"sac"),linkpath)
			
			#start the fun
			outfilename = os.path.splitext(filename)[0]+".o"
			command = [self.CC, extraCFLAGS, self.CFLAGS, "-c", newcfile, "-o", outfilename]
			commandline = " ".join(command)
			if verbose:
				print >> sys.stderr , "Compiling a workspace file with", commandline
			p = Popen(commandline, shell=True, stdout = PIPE, stderr = PIPE)
			(out,err) = p.communicate()
			self.cc_stderr = err
			ret = p.returncode
			if ret != 0:
				os.remove(filename)
				print >> sys.stderr, err
				raise RuntimeError("%s failed with return code %d" % (commandline, ret))
			self.cc_cmd = commandline
			return [outfilename]

	return C

