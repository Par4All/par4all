import os,sys
import re
import tempfile
import shutil
import pypsutils
import fileinput
import subprocess

class sacbase(object):
	@staticmethod
	def sac(module, **cond):
		ws = module._ws
		
		# Here are the transformations made by benchmark.tpips.h, blindy
		# translated in pyps.

		ws.activate("MUST_REGIONS")
		ws.activate("REGION_CHAINS")
		ws.activate("RICE_REGIONS_DEPENDENCE_GRAPH")
		ws.activate("PRECONDITIONS_INTER_FULL")
		ws.activate("TRANSFORMERS_INTER_FULL")

		ws.props.loop_unroll_with_prologue = False
		ws.props.constant_path_effects = False
		ws.props.ricedg_statistics_all_arrays = True
		ws.props.c89_code_generation = True
		ws.props.delay_communications_interprocedural = False


		ws.props.simd_fortran_mem_organisation = False
		ws.props.sac_simd_register_width = cond["register_width"]
		ws.props.prettyprint_all_declarations = True
		ws.props.compute_all_dependences = True

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


		#if cond.get("reduction_detection", False):
		#	try:
		#		ws.set_property(KEEP_READ_READ_DEPENDENCE = True)
		#		while True:
		#			module.reduction_detection()
		#			if cond.get("verbose"):
		#				module.display()
		#	except:
		#		pass

		#	if cond.get("verbose"):
		#		module.display()

		if cond.get("auto_unroll", True):
			module.simdizer_auto_unroll(minimize_unroll=False,simple_calculation=True)
		module.partial_eval()
		if cond.get("full_unroll", False):
			for l in module.inner_loops():
				try:l.full_unroll()
				except RuntimeError:pass
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

		module.reduction_atomization()
		if cond.get("verbose"):
			module.display()

		for p in ( "__PIPS_SAC_MULADD" , ):
			module.expression_substitution(pattern=p)
			if cond.get("verbose"):
				module.display()

		module.simd_atomizer()
		module.partial_eval()
		if cond.get("verbose"):
			module.display()
			module.print_dot_dependence_graph()


		# module.deatomizer()
		# module.partial_eval()
		# module.use_def_elimination()
		# module.display()

		# module.print_dot_dependence_graph()
		module.scalar_renaming()

		if cond.get("verbose"):
			module.display()

		module.simdizer(allow_padding = cond.get("simdizer_allow_padding", False))
		if cond.get("verbose"):
			module.display()
		try:
			module.print_dot_dependence_graph()
			module.delay_communications()
			module.flatten_code(unroll = False)
		except RuntimeError: pass

		# module.use_def_elimination()

		# module.use_def_elimination()
		if cond.get("suppress_dead_code", True):
			module.suppress_dead_code()
		module.flatten_code(unroll = False)
		if cond.get("enhanced_reduction",False):
			module.redundant_load_store_elimination(SIMD_REMOVE_REDUCTIONS_PRELUDE="SIMD_ZERO_V4SF")
		else:
			module.redundant_load_store_elimination()
		if cond.get("suppress_dead_code", True):
			module.suppress_dead_code()
		module.clean_declarations()

		try:
			module.print_dot_dependence_graph()
			module.delay_communications()
			module.flatten_code(unroll = False)
		except RuntimeError: pass
		# ws.set_property(EOLE_OPTIMIZATION_STRATEGY = "ICM")
		# module.optimize_expressions()
		# module.partial_redundancy_elimination()
		# module.common_subexpression_elimination()
		if cond.get("verbose"):
			module.display()

		# module.use_def_elimination()
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

class workspace:
	"""The SAC subsystem, in Python.

	Add a new transformation, for adapting code to SIMD instruction
	sets (SSE, 3Dnow, AVX and ARM NEON)"""
	def __init__(self, ws, **args):
		# Add SIMD.c and patterns.c to the project
		self.tmpdir = tempfile.mkdtemp()
		tmpSIMD = self.tmpdir + "/SIMD.c"
		tmpPatterns = self.tmpdir + "/patterns.c"
		pypsutils.string2file(simd_c, tmpSIMD)
		pypsutils.string2file(patterns_c, tmpPatterns)
		ws._sources.append(tmpSIMD)
		ws._sources.append(tmpPatterns)
		self.ws = ws
		# Do we want not to compile with $driver.h by default?
		self.use_generic_simd = True
		drivers = {"sse": sacsse, "3dnow": sac3dnow, "avx": sacavx, "neon": sacneon}
		self.driver = drivers[args.get("driver", "sse")]
		# Add -DRWBITS=self.driver.register_width to the cppflags of the workspace
		self.ws.cppflags += " -DRWBITS=%d " % (self.driver.register_width)

	def post_init(self, sources, **args):
		"""Clean the temporary directory used for holding 'SIMD.c' and 'patterns.c'."""
		shutil.rmtree(self.tmpdir)
		for m in self.ws:
			m.__class__.sac = self.driver.sac

	def pre_goingToRunWith(self, files, outdir):
		"""Add $driver.h, which replaces general purpose SIMD instructions
		with machine-specific ones."""
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
			simd_h_fname = os.path.abspath(outdir + "/SIMD.h")
			simd_c_fname = os.path.abspath(outdir + "/SIMD.c")
			p = subprocess.Popen("gcc -DRWBITS=%d -E %s |cproto" % (self.driver.register_width, simd_c_fname), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
			(simd_cus_header,serr) = p.communicate()
			if p.returncode != 0:
				raise RuntimeError("Error while creating SIMD.h: command returned %d.\nstdout:\n%s\nstderr:\n%s\n" % (p.returncode, simd_cus_header, serr))

			pypsutils.string2file(simd_h+"\n"+simd_cus_header, simd_h_fname)
			for fname in files:
				if not fname.endswith("SIMD.c"):
					pypsutils.addBeginnning(fname, '#include "SIMD.h"')

		# Add the contents of patterns.h
		for fname in files:
			pypsutils.addBeginnning(fname, patterns_h)

	def simd_compile(self, ccexecp, *args, **kwargs):
		"""Compile the workspace with sse.h."""
		self.use_generic_simd = False
		CFLAGS = self.driver.CFLAGS
		ccexecp.CFLAGS += " " + CFLAGS
		r = self.ws.compile(ccexecp, *args, **kwargs)
		self.use_generic_simd = True
		return r

	def post_memalign(self, *args, **kwargs):
		self.driver.post_memalign(self, *args, **kwargs)

simd_c = """
%%include impl/SIMD.c%%
"""

simd_h = """
#include <stdarg.h>
#include <stdint.h>

%%include impl/SIMD_types.h%%
"""

sse_h = """
%%include impl/sse.h%%
"""

Threednow_h = """
%%include impl/3dnow.h%%
"""

avx_h = """
%%include impl/avx.h%%
"""

neon_h = """
%%include impl/neon.h%%
"""

patterns_c = """
%%include impl/patterns.c%%
"""

patterns_h = """
%%gen_header impl/patterns.c%%
"""
