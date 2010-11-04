import os
import re
import tempfile
import shutil
import pypsutils
import fileinput

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

		ws.props.constant_path_effects = False
		ws.props.ricedg_statistics_all_arrays = True
		ws.props.c89_code_generation = True

		ws.props.simd_fortran_mem_organisation = False
		ws.props.sac_simd_register_width = cond["register_width"]
		ws.props.prettyprint_all_declarations = True

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

		module.simd_atomizer()
		module.partial_eval()
		if cond.get("verbose"):
			module.display()
			module.print_dot_dependence_graph()

		if cond.get("reduction_detection", False):
			try:
				ws.set_property(KEEP_READ_READ_DEPENDENCE = True)
				while True:
					module.reduction_detection()
					if cond.get("verbose"):
						module.display()
			except:
				pass

			if cond.get("verbose"):
				module.display()

		if cond.get("auto_unroll", True):
			module.simdizer_auto_unroll(minimize_unroll=False,simple_calculation=False)
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
			for i in xrange(3):
				module.simd_loop_const_elim()
				module.redundant_load_store_elimination(SIMD_REMOVE_REDUCTIONS_PRELUDE="SIMD_ZERO_V4SF")
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
	@staticmethod
	def sac(module, **kwargs):
		global curr_sse_h
		kwargs["register_width"] = 128
		sacbase.sac(module, **kwargs)
		curr_sse_h=sse_h

	@staticmethod
	def addintrinsics(fname):
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
		global curr_sse_h
		curr_sse_h = re.sub("_mm_loadu", "_mm_load", sse_h)
		curr_sse_h = re.sub("_mm_storeu", "_mm_store", sse_h)

	CFLAGS = "-msse4.2 -march=native -O3 "

class sac3dnow(sacbase):
	@staticmethod
	def sac(module, *args, **kwargs):
		kwargs["register_width"] = 64
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

	CFLAGS = "-m3dnow -march=opteron -O2 "

class workspace:
	"""The SAC subsystem, in Python.

	Add a new transformation, for adapting code to SIMD instruction
	sets (SSE for now, 3Dnow coming soon)."""
	def __init__(self, ws, sources, **args):
		# add SIMD.c to the project
		self.tmpdir = tempfile.mkdtemp()
		tmpSIMD = self.tmpdir + "/SIMD.c"
		pypsutils.string2file(simd_c, tmpSIMD)
		ws._sources.append(tmpSIMD)
		self.ws = ws
		# Do we want to compile with SSE.h by default? NB: if changing
		# this, invert also the logic in compile() and compile_sse().
		self.compile_sse = False
		drivers = {"sse": sacsse, "3dnow": sac3dnow}
		self.driver = drivers[args.get("driver", "sse")]

	def post_init(self, sources, **args):
		"""Clean the temporary directory used for holding `SIMD.c'."""
		shutil.rmtree(self.tmpdir)
		for m in self.ws:
			m.__class__.sac = self.driver.sac

	def pre_goingToRunWith(self, files, outdir):
		"""Add sse.h, which replaces general purpose SIMD instructions
		with machine-specific ones."""
		if self.compile_sse:
			# We need to add SSE.h to the compiled files, and to remove SIMD.c
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
			simd_h_fname = os.path.abspath(outdir + "/SIMD.h")
			pypsutils.string2file(simd_h, simd_h_fname)
			for fname in files:
				pypsutils.addBeginnning(fname, '#include "'+simd_h_fname+'"')

	def simd_compile(self, **args):
		"""Compile the workspace with sse.h."""
		self.compile_sse = True
		CFLAGS = self.driver.CFLAGS
		if args.has_key("CFLAGS"):
			args["CFLAGS"] = CFLAGS + args["CFLAGS"]
		else:
			args["CFLAGS"] = CFLAGS
		r = self.ws.compile(**args)
		self.compile_sse = False
		return r

	def post_memalign(self, *args, **kwargs):
		self.driver.post_memalign(self, *args, **kwargs)

# SIMD.c from validation/SAC/include/SIMD.c r2257
simd_c = """
#ifdef WITH_TRIGO
#  include <math.h>
#  ifndef COS
#	define COS cos
#  endif
#  ifndef SIN
#	define SIN sin
#  endif
#endif

#define LOGICAL int
#define DMAX(A,B) (A)>(B)?(A):(B)
void SIMD_LOAD_V4SI_TO_V4SF(float a[4], int b[4])
{
	a[0]=b[0];
	a[1]=b[1];
	a[2]=b[2];
	a[3]=b[3];
}
void SIMD_STORE_V4SF_TO_V4SI(float a[4], int b[4])
{
	b[0]=a[0];
	b[1]=a[1];
	b[2]=a[2];
	b[3]=a[3];
}
void SIMD_STORE_V2DF_TO_V2SF(double a[2],float b[2])
{
	b[0]=a[0];
	b[1]=a[1];
}
void SIMD_LOAD_V2SF_TO_V2DF(double a[2],float b[2])
{
	a[0]=b[0];
	a[1]=b[1];
}

int
PHI (LOGICAL L, int X1, int X2)
{
	return L ? X1 : X2;
}

void
SIMD_PHIW(int R[4], LOGICAL L[4], int X1[4], int X2[4])
{
	int i;
	for (i=0;i<2;i++)
		R[i]=L[i]?X1[i]:X2[i];
}

void
SIMD_GTD(int R[4], int X1[4], int X2[4])
{
	int i;
	for (i=0;i<4;i++)
		R[i]=X1[i]>X2[i];
}
void
SIMD_LOAD_V4SI (int VEC[4], int BASE[4])
{
	VEC[0] = BASE[0];
	VEC[1] = BASE[1];
	VEC[2] = BASE[2];
	VEC[3] = BASE[3];
}

void
SIMD_LOAD_V4SF (float VEC[4], float BASE[4])
{
	VEC[0] = BASE[0];
	VEC[1] = BASE[1];
	VEC[2] = BASE[2];
	VEC[3] = BASE[3];
}

void
SIMD_LOAD_V2DF (double VEC[2], double BASE[2])
{
	VEC[0] = BASE[0];
	VEC[1] = BASE[1];
}

void
SIMD_LOAD_GENERIC_V2DF (double VEC[2], double X0, double X1)
{
	VEC[0] = X0;
	VEC[1] = X1;
}
void
SIMD_LOAD_GENERIC_V4SI (int VEC[4], int X0, int X1, int X2, int X3)
{
	VEC[0] = X0;
	VEC[1] = X1;
	VEC[2] = X2;
	VEC[3] = X3;
}

void
SIMD_LOAD_GENERIC_V4SF (float VEC[4], float X0, float X1, float X2, float X3)
{
	VEC[0] = X0;
	VEC[1] = X1;
	VEC[2] = X2;
	VEC[3] = X3;
}

void
SIMD_LOAD_CONSTANT_V4SF (float VEC[4], float X0, float X1, float X2, float X3)
{

	VEC[0] = X0;
	VEC[1] = X1;
	VEC[2] = X2;
	VEC[3] = X3;
}

void
SIMD_LOAD_CONSTANT_V4SI (int VEC[4], int X0, int X1, int X2, int X3)
{

	VEC[0] = X0;
	VEC[1] = X1;
	VEC[2] = X2;
	VEC[3] = X3;
}

void
SIMD_STORE_V4SI (int VEC[4], int BASE[4])
{  
	BASE[0] = VEC[0];
	BASE[1] = VEC[1];
	BASE[2] = VEC[2];
	BASE[3] = VEC[3];
}
void
SIMD_STORE_V4SF (float VEC[4], float BASE[4])
{  
	BASE[0] = VEC[0];
	BASE[1] = VEC[1];
	BASE[2] = VEC[2];
	BASE[3] = VEC[3];
}
void
SIMD_STORE_V2DF (double VEC[2], double BASE[2])
{  
	BASE[0] = VEC[0];
	BASE[1] = VEC[1];
}

void
SIMD_STORE_MASKED_V4SF(float VEC[4], float BASE[3])
{  
	BASE[0] = VEC[0];
	BASE[1] = VEC[1];
	BASE[2] = VEC[2];
}


void
SIMD_STORE_GENERIC_V2DF (double VEC[2], double X1[1], double X2[1])
{

	X1 [0]= VEC[0];
	X2 [0]= VEC[1];
}
void
SIMD_STORE_GENERIC_V4SI (int VEC[4], int X1[1], int X2[1],
		int X3[1], int X4[1])
{

	X1 [0]= VEC[0];
	X2 [0]= VEC[1];
	X3 [0]= VEC[2];
	X4 [0]= VEC[3];
}
void
SIMD_STORE_GENERIC_V4SF (float VEC[4], float X1[1], float X2[1],
		float X3[1], float X4[1])
{

	X1 [0]= VEC[0];
	X2 [0]= VEC[1];
	X3 [0]= VEC[2];
	X4 [0]= VEC[3];
}

void
SIMD_ZERO_V4SF(float VEC[4])
{
	VEC[0] = 0.0f;
	VEC[1] = 0.0f;
	VEC[2] = 0.0f;
	VEC[3] = 0.0f;
}

void
SIMD_GTPS (LOGICAL DEST[4], float SRC1[4], float SRC2[4])
{
	DEST[0] = SRC1[0] > SRC2[0];
	DEST[1] = SRC1[1] > SRC2[1];
	DEST[2] = SRC1[2] > SRC2[2];
	DEST[3] = SRC1[3] > SRC2[3];
}
void
SIMD_GTPD (LOGICAL DEST[2], double SRC1[2], double SRC2[2])
{
	DEST[0] = SRC1[0] > SRC2[0];
	DEST[1] = SRC1[1] > SRC2[1];
}

void
SIMD_PHIPS (float DEST[4], LOGICAL COND[4], float SRC1[4], float SRC2[4])
{

	if (COND[0])
	{
		DEST[0] = SRC1[0];
	}
	else
	{
		DEST[0] = SRC2[0];
	}
	if (COND[1])
	{
		DEST[1] = SRC1[1];
	}
	else
	{
		DEST[1] = SRC2[1];
	}
	if (COND[2])
	{
		DEST[2] = SRC1[2];
	}
	else
	{
		DEST[2] = SRC2[2];
	}
	if (COND[3])
	{
		DEST[3] = SRC1[3];
	}
	else
	{
		DEST[3] = SRC2[3];
	}
}

void
SIMD_ADDPS (float DEST[4], float SRC1[4], float SRC2[4])
{
	DEST[0] = SRC1[0] + SRC2[0];
	DEST[1] = SRC1[1] + SRC2[1];
	DEST[2] = SRC1[2] + SRC2[2];
	DEST[3] = SRC1[3] + SRC2[3];
}

void
SIMD_SUBPS (float DEST[4], float SRC1[4], float SRC2[4])
{
	DEST[0] = SRC1[0] - SRC2[0];
	DEST[1] = SRC1[1] - SRC2[1];
	DEST[2] = SRC1[2] - SRC2[2];
	DEST[3] = SRC1[3] - SRC2[3];
}

void
SIMD_UMINPS (float DEST[4], float SRC1[4])
{
	DEST[0] =  - SRC1[0];
	DEST[1] =  - SRC1[1];
	DEST[2] =  - SRC1[2];
	DEST[3] =  - SRC1[3];
}

void
SIMD_MULPS (float DEST[4], float SRC1[4], float SRC2[4])
{
	DEST[0] = SRC1[0] * SRC2[0];
	DEST[1] = SRC1[1] * SRC2[1];
	DEST[2] = SRC1[2] * SRC2[2];
	DEST[3] = SRC1[3] * SRC2[3];
}
void
SIMD_DIVPD (double DEST[2], double SRC1[2], double SRC2[2])
{
	DEST[0] = SRC1[0] / SRC2[0];
	DEST[1] = SRC1[1] / SRC2[1];
}
void
SIMD_MULPD (double DEST[2], double SRC1[2], double SRC2[2])
{
	DEST[0] = SRC1[0] * SRC2[0];
	DEST[1] = SRC1[1] * SRC2[1];
}

#ifdef WITH_TRIGO

void
SIMD_SINPD (double DEST[2], double SRC1[2])
{
	DEST[0] = SIN(SRC1[0]);
	DEST[1] = SIN(SRC1[1]);
}
void
SIMD_COSPD (double DEST[2], double SRC1[2])
{
	DEST[0] = COS(SRC1[0]);
	DEST[1] = COS(SRC1[1]);
}
#endif
void
SIMD_ADDPD (double DEST[2], double SRC1[2], double SRC2[2])
{
	DEST[0] = SRC1[0] + SRC2[0];
	DEST[1] = SRC1[1] + SRC2[1];
}
void
SIMD_SUBPD (double DEST[2], double SRC1[2], double SRC2[2])
{
	DEST[0] = SRC1[0] - SRC2[0];
	DEST[1] = SRC1[1] - SRC2[1];
}

void
SIMD_UMINPD (double DEST[2], double SRC1[2])
{
	DEST[0] =  - SRC1[0];
	DEST[1] =  - SRC1[1];
}

void
SIMD_DIVPS (float DEST[4], float SRC1[4], float SRC2[4])
{
	DEST[0] = SRC1[0] / SRC2[0];
	DEST[1] = SRC1[1] / SRC2[1];
	DEST[2] = SRC1[2] / SRC2[2];
	DEST[3] = SRC1[3] / SRC2[3];
}

void
SIMD_MAXPS (float DEST[4], float SRC1[4], float SRC2[4])
{
	DEST[0] = DMAX (SRC1[0], SRC2[0]);
	DEST[1] = DMAX (SRC1[1], SRC2[1]);
	DEST[2] = DMAX (SRC1[2], SRC2[2]);
	DEST[3] = DMAX (SRC1[3], SRC2[3]);
}

/* Integer manipulation */
/* V4SI: vector of 4 32-bit integers (standard integers ?)
	V4HI: vector of 4 16-bit integers (half integers ?) */

void
SIMD_LOAD_V2SI_TO_V2SF(float TO[2], int VEC[2])
{
	TO[0]=VEC[0];
	TO[1]=VEC[1];
}
void
SIMD_STORE_V2SI_TO_V2SF(int TO[2], float VEC[2])
{
	TO[0]=VEC[0];
	TO[1]=VEC[1];
}

void
SIMD_STORE_V2SF(float VEC[2], float BASE[2])
{
	BASE[0] = VEC[0];
	BASE[1] = VEC[1];
}

void
SIMD_LOAD_V2SF(float VEC[2], float BASE[2])
{
	VEC[0] = BASE[0];
	VEC[1] = BASE[1];
}

void
SIMD_LOAD_GENERIC_V2SF(float VEC[2], float BASE0, float BASE1)
{
	VEC[0] = BASE0;
	VEC[1] = BASE1;
}

void
SIMD_LOAD_CONSTANT_V2SF (float VEC[2], float HIGH, float LOW)
{

	VEC[0] = LOW;
	VEC[1] = HIGH;
}
void
SIMD_LOAD_CONSTANT_V2SI (int VEC[2], int HIGH, int LOW)
{

	VEC[0] = LOW;
	VEC[1] = HIGH;
}

void
SIMD_LOAD_V2SI (int VEC[2], int BASE[2])
{  
	VEC[0] = BASE[0];
	VEC[1] = BASE[1];
}

void
SIMD_LOAD_GENERIC_V2SI (int VEC[2], int X1, int X2)
{

	VEC[0] = X1;
	VEC[1] = X2;
}

void
SIMD_STORE_V2SI (int VEC[2], int BASE[2])
{  BASE[0] = VEC[0];
	BASE[1] = VEC[1];
}

void
SIMD_STORE_GENERIC_V2SI (int VEC[2], int X1[1], int X2[1])
{

	X1 [0]= VEC[0];
	X2 [0]= VEC[1];
}


void
SIMD_STORE_V2DI (int VEC[2], int BASE[2])
{  BASE[0] = VEC[0];
	BASE[1] = VEC[1];
}

void
SIMD_ADDW (short DEST[8], short SRC1[8], short SRC2[8])
{
	DEST[0] = SRC1[0] + SRC2[0];
	DEST[1] = SRC1[1] + SRC2[1];
	DEST[2] = SRC1[2] + SRC2[2];
	DEST[3] = SRC1[3] + SRC2[3];
	DEST[4] = SRC1[4] + SRC2[4];
	DEST[5] = SRC1[5] + SRC2[5];
	DEST[6] = SRC1[6] + SRC2[6];
	DEST[7] = SRC1[7] + SRC2[7];
}

void
SIMD_SUBW (short DEST[8], short SRC1[8], short SRC2[8])
{
	DEST[0] = SRC1[0] - SRC2[0];
	DEST[1] = SRC1[1] - SRC2[1];
	DEST[2] = SRC1[2] - SRC2[2];
	DEST[3] = SRC1[3] - SRC2[3];
	DEST[4] = SRC1[4] - SRC2[4];
	DEST[5] = SRC1[5] - SRC2[5];
	DEST[6] = SRC1[6] - SRC2[6];
	DEST[7] = SRC1[7] - SRC2[7];
}

void
SIMD_MULW (short DEST[8], short SRC1[8], short SRC2[8])
{
	DEST[0] = SRC1[0] * SRC2[0];
	DEST[1] = SRC1[1] * SRC2[1];
	DEST[2] = SRC1[2] * SRC2[2];
	DEST[3] = SRC1[3] * SRC2[3];
	DEST[4] = SRC1[4] * SRC2[4];
	DEST[5] = SRC1[5] * SRC2[5];
	DEST[6] = SRC1[6] * SRC2[6];
	DEST[7] = SRC1[7] * SRC2[7];
}

void
SIMD_DIVW (short DEST[8], short SRC1[8], short SRC2[8])
{
	DEST[0] = SRC1[0] / SRC2[0];
	DEST[1] = SRC1[1] / SRC2[1];
	DEST[2] = SRC1[2] / SRC2[2];
	DEST[3] = SRC1[3] / SRC2[3];
	DEST[4] = SRC1[4] / SRC2[4];
	DEST[5] = SRC1[5] / SRC2[5];
	DEST[6] = SRC1[6] / SRC2[6];
	DEST[7] = SRC1[7] / SRC2[7];
}

void
SIMD_LOAD_GENERIC_V8HI(short VEC[8], short BASE0, short BASE1, short BASE2, short BASE3, short BASE4, short BASE5, short BASE6, short BASE7)
{  
	VEC[0] = BASE0;
	VEC[1] = BASE1;
	VEC[2] = BASE2;
	VEC[3] = BASE3;
	VEC[4] = BASE4;
	VEC[5] = BASE5;
	VEC[6] = BASE6;
	VEC[7] = BASE7;
}

void
SIMD_LOAD_V8HI (short VEC[8], short BASE[8])
{  
	VEC[0] = BASE[0];
	VEC[1] = BASE[1];
	VEC[2] = BASE[2];
	VEC[3] = BASE[3];
	VEC[4] = BASE[4];
	VEC[5] = BASE[5];
	VEC[6] = BASE[6];
	VEC[7] = BASE[7];
}

void
SIMD_LOAD_V4QI_TO_V4HI (short VEC[4], char BASE[4])
{  VEC[0] = BASE[0];
	VEC[1] = BASE[1];
	VEC[2] = BASE[2];
	VEC[3] = BASE[3];
}

void
SIMD_LOAD_V4HI_TO_V4SI(int DEST[4], short SRC[4]) {
	DEST[0] = SRC[0];
	DEST[1] = SRC[1];
	DEST[2] = SRC[2];
	DEST[3] = SRC[3];
}

void
SIMD_STORE_V8HI (short VEC[8], short BASE[8])
{  
	BASE[0] = VEC[0];
	BASE[1] = VEC[1];
	BASE[2] = VEC[2];
	BASE[3] = VEC[3];
	BASE[4] = VEC[4];
	BASE[5] = VEC[5];
	BASE[6] = VEC[6];
	BASE[7] = VEC[7];
}



void
SIMD_PHID (int DEST[4], LOGICAL COND[4], int SRC1[4], int SRC2[4])
{

	if (COND[0])
	{
		DEST[0] = SRC1[0];
	}
	else
	{
		DEST[0] = SRC2[0];
	}
	if (COND[1])
	{
		DEST[1] = SRC1[1];
	}
	else
	{
		DEST[1] = SRC2[1];
	}
	if (COND[2])
	{
		DEST[2] = SRC1[2];
	}
	else
	{
		DEST[2] = SRC2[2];
	}
	if (COND[3])
	{
		DEST[3] = SRC1[3];
	}
	else
	{
		DEST[3] = SRC2[3];
	}
}

void
SIMD_ADDD (int DEST[4], int SRC1[4], int SRC2[4])
{
	DEST[0] = SRC1[0] + SRC2[0];
	DEST[1] = SRC1[1] + SRC2[1];
	DEST[2] = SRC1[2] + SRC2[2];
	DEST[3] = SRC1[3] + SRC2[3];
}

void
SIMD_SUBD (int DEST[4], int SRC1[4], int SRC2[4])
{
	DEST[0] = SRC1[0] - SRC2[0];
	DEST[1] = SRC1[1] - SRC2[1];
	DEST[2] = SRC1[2] - SRC2[2];
	DEST[3] = SRC1[3] - SRC2[3];
}

void
SIMD_MULD (int DEST[4], int SRC1[4], int SRC2[4])
{
	DEST[0] = SRC1[0] * SRC2[0];
	DEST[1] = SRC1[1] * SRC2[1];
	DEST[2] = SRC1[2] * SRC2[2];
	DEST[3] = SRC1[3] * SRC2[3];
}
void
SIMD_DIVD (int DEST[4], int SRC1[4], int SRC2[4])
{
	DEST[0] = SRC1[0] / SRC2[0];
	DEST[1] = SRC1[1] / SRC2[1];
	DEST[2] = SRC1[2] / SRC2[2];
	DEST[3] = SRC1[3] / SRC2[3];
}

void
SIMD_LOAD_CONSTANT_V8QI (char VEC[8], int HIGH, int LOW)
{
	VEC[0] = (char) LOW;
	VEC[1] = (char) (LOW >> 1);
	VEC[2] = (char) (LOW >> 2);
	VEC[3] = (char) (LOW >> 3);
	VEC[4] = (char) HIGH;
	VEC[5] = (char) (HIGH >> 1);
	VEC[6] = (char) (HIGH >> 2);
	VEC[7] = (char) (HIGH >> 3);
}

void
SIMD_LOAD_V8QI (char VEC[8], char BASE[8])
{  VEC[0] = BASE[0];
	VEC[1] = BASE[1];
	VEC[2] = BASE[2];
	VEC[3] = BASE[3];
	VEC[4] = BASE[4];
	VEC[5] = BASE[5];
	VEC[6] = BASE[6];
	VEC[7] = BASE[7];
}

void
SIMD_LOAD_GENERIC_V8QI (char VEC[8], char X1,
		char X2, char X3, char X4, char X5, char X6,
		char X7, char X8)
{
	VEC[0] = X1;
	VEC[1] = X2;
	VEC[2] = X3;
	VEC[3] = X4;
	VEC[4] = X5;
	VEC[5] = X6;
	VEC[6] = X7;
	VEC[7] = X8;
}

void
SIMD_STORE_V8QI (char VEC[8], char BASE[8])
{  BASE[0] = VEC[0];
	BASE[1] = VEC[1];
	BASE[2] = VEC[2];
	BASE[3] = VEC[3];
	BASE[4] = VEC[4];
	BASE[5] = VEC[5];
	BASE[6] = VEC[6];
	BASE[7] = VEC[7];
}

void
SIMD_STORE_GENERIC_V8QI (char VEC[8], char *X0,
		char X1[1], char X2[1], char X3[1], char X4[1], char X5[1],
		char X6[1], char X7[1])
{

	X0[0] = VEC[0];
	X1[0] = VEC[1];
	X2[0] = VEC[2];
	X3[0] = VEC[3];
	X4[0] = VEC[4];
	X5[0] = VEC[5];
	X6[0] = VEC[6];
	X7[0] = VEC[7];
}

void
SIMD_ADDB (char DEST[8], char SRC1[8], char SRC2[8])
{
	DEST[0] = SRC1[0] + SRC2[0];
	DEST[1] = SRC1[1] + SRC2[1];
	DEST[2] = SRC1[2] + SRC2[2];
	DEST[3] = SRC1[3] + SRC2[3];
	DEST[4] = SRC1[4] + SRC2[4];
	DEST[5] = SRC1[5] + SRC2[5];
	DEST[6] = SRC1[6] + SRC2[6];
	DEST[7] = SRC1[7] + SRC2[7];
}

void
SIMD_SUBB (char DEST[8], char SRC1[8], char SRC2[8])
{
	DEST[0] = SRC1[0] - SRC2[0];
	DEST[1] = SRC1[1] - SRC2[1];
	DEST[2] = SRC1[2] - SRC2[2];
	DEST[3] = SRC1[3] - SRC2[3];
	DEST[4] = SRC1[4] - SRC2[4];
	DEST[5] = SRC1[5] - SRC2[5];
	DEST[6] = SRC1[6] - SRC2[6];
	DEST[7] = SRC1[7] - SRC2[7];
}

void
SIMD_MULB (char DEST[8], char SRC1[8], char SRC2[8])
{

	DEST[0] = SRC1[0] * SRC2[0];
	DEST[1] = SRC1[1] * SRC2[1];
	DEST[2] = SRC1[2] * SRC2[2];
	DEST[3] = SRC1[3] * SRC2[3];
	DEST[4] = SRC1[4] * SRC2[4];
	DEST[5] = SRC1[5] * SRC2[5];
	DEST[6] = SRC1[6] * SRC2[6];
	DEST[7] = SRC1[7] * SRC2[7];
}

void
SIMD_MOVPS (float DEST[2], float SRC[2])
{
	DEST[0] = SRC[0];
	DEST[1] = SRC[1];
}

void
SIMD_MOVD (int DEST[2], int SRC[2])
{
	DEST[0] = SRC[0];
	DEST[1] = SRC[1];
}

void
SIMD_MOVW (short DEST[4], short SRC[4])
{
	DEST[0] = SRC[0];
	DEST[1] = SRC[1];
	DEST[2] = SRC[2];
	DEST[3] = SRC[3];
}

void
SIMD_MOVB (char DEST[8], char SRC[8])
{

	DEST[0] = SRC[0];
	DEST[1] = SRC[1];
	DEST[2] = SRC[2];
	DEST[3] = SRC[3];
	DEST[4] = SRC[4];
	DEST[5] = SRC[5];
	DEST[6] = SRC[6];
	DEST[7] = SRC[7];
}

void
SIMD_OPPPS (float DEST[2], float SRC[2])
{
	DEST[0] = -SRC[0];
	DEST[1] = -SRC[1];
}

void
SIMD_OPPD (int DEST[2], int SRC[2])
{
	DEST[0] = -SRC[0];
	DEST[1] = -SRC[1];
}

void
SIMD_OPPW (short DEST[4], short SRC[4])
{
	DEST[0] = -SRC[0];
	DEST[1] = -SRC[1];
	DEST[2] = -SRC[2];
	DEST[3] = -SRC[3];
}

void
SIMD_OPPB (char DEST[8], char SRC[8])
{
	DEST[0] = -SRC[0];
	DEST[1] = -SRC[1];
	DEST[2] = -SRC[2];
	DEST[3] = -SRC[3];
	DEST[4] = -SRC[4];
	DEST[5] = -SRC[5];
	DEST[6] = -SRC[6];
	DEST[7] = -SRC[7];
}

void
SIMD_SETPS (float DEST[2], float SRC[2])
{
	DEST[0] = SRC[0];
	DEST[1] = SRC[1];
	DEST[2] = SRC[2];
	DEST[3] = SRC[3];
}
void
SIMD_SETPD (double DEST[2], double SRC[2])
{
	DEST[0] = SRC[0];
	DEST[1] = SRC[1];
}

void
SIMD_SETD (int DEST[2], int SRC[2])
{
	DEST[0] = SRC[0];
	DEST[1] = SRC[1];
	DEST[2] = SRC[2];
	DEST[3] = SRC[3];
}

void
SIMD_SETW (short DEST[4], short SRC[4])
{
	DEST[0] = SRC[0];
	DEST[1] = SRC[1];
}

void
SIMD_SETB (char DEST[8], char SRC[8])
{
	DEST[0] = SRC[0];
	DEST[1] = SRC[1];
	DEST[2] = SRC[2];
	DEST[3] = SRC[3];
	DEST[4] = SRC[4];
	DEST[5] = SRC[5];
	DEST[6] = SRC[6];
	DEST[7] = SRC[7];
}

void
SIMD_LOAD_CONSTANT_V2DF(double vec[2],double v0,double v1)
{
	vec[0]=v0;
	vec[1]=v1;
}

/* char */
void SIMD_STORE_V8HI_TO_V8SI(char vec[8],char arr[8])
{
	vec[0] = arr[0];
	vec[1] = arr[1];
	vec[2] = arr[2];
	vec[3] = arr[3];
	vec[4] = arr[4];
	vec[5] = arr[5];
	vec[6] = arr[6];
	vec[7] = arr[7];
}
void SIMD_LOAD_V8SI_TO_V8HI(char vec[8],char arr[8])
{
	vec[0] = arr[0];
	vec[1] = arr[1];
	vec[2] = arr[2];
	vec[3] = arr[3];
	vec[4] = arr[4];
	vec[5] = arr[5];
	vec[6] = arr[6];
	vec[7] = arr[7];
}


#undef LOGICAL
#undef DMAX
"""

simd_h = """
typedef float  a2sf[2];
typedef float  a4sf[4];
typedef double a2df[2];
typedef int	a2si[2];
typedef int	a4si[4];

typedef float  v2sf[2];
typedef float  v4sf[4];
typedef double v2df[2];
typedef int	v2si[2];
typedef int	v4si[4];
typedef short	v8si[8];

void SIMD_LOAD_V4SI_TO_V4SF(float a[4], int b[4]);
void SIMD_STORE_V4SF_TO_V4SI(float a[4], int b[4]);
void SIMD_STORE_V2DF_TO_V2SF(double a[2], float b[2]);
void SIMD_LOAD_V2SF_TO_V2DF(double a[2], float b[2]);
int PHI(int L, int X1, int X2);
void SIMD_PHIW(int R[4], int L[4], int X1[4], int X2[4]);
void SIMD_GTD(int R[4], int X1[4], int X2[4]);
void SIMD_LOAD_V4SI(int VEC[4], int BASE[4]);
void SIMD_LOAD_V4SF(float VEC[4], float BASE[4]);
void SIMD_LOAD_V2DF(double VEC[2], double BASE[2]);
void SIMD_LOAD_GENERIC_V2DF(double VEC[2], double X0, double X1);
void SIMD_LOAD_GENERIC_V4SI(int VEC[4], int X0, int X1, int X2, int X3);
void SIMD_LOAD_GENERIC_V4SF(float VEC[4], float X0, float X1, float X2, float X3);
void SIMD_LOAD_CONSTANT_V4SF(float VEC[4], float X0, float X1, float X2, float X3);
void SIMD_LOAD_CONSTANT_V4SI(int VEC[4], int X0, int X1, int X2, int X3);
void SIMD_STORE_V4SI(int VEC[4], int BASE[4]);
void SIMD_STORE_V4SF(float VEC[4], float BASE[4]);
void SIMD_STORE_V2DF(double VEC[2], double BASE[2]);
void SIMD_STORE_MASKED_V4SF(float VEC[4], float BASE[3]);
void SIMD_STORE_GENERIC_V2DF(double VEC[2], double X1[1], double X2[1]);
void SIMD_STORE_GENERIC_V4SI(int VEC[4], int X1[1], int X2[1], int X3[1], int X4[1]);
void SIMD_STORE_GENERIC_V4SF(float VEC[4], float X1[1], float X2[1], float X3[1], float X4[1]);
void SIMD_ZERO_V4SF(float VEC[4]);
void SIMD_STORE_V2SF(float VEC[2], float BASE[2]);
void SIMD_LOAD_V2SF(float VEC[2], float BASE[2]);
void SIMD_LOAD_GENERIC_V2SF(float VEC[2], float BASE0, float BASE1);
void SIMD_GTPS(int DEST[4], float SRC1[4], float SRC2[4]);
void SIMD_GTPD(int DEST[2], double SRC1[2], double SRC2[2]);
void SIMD_PHIPS(float DEST[4], int COND[4], float SRC1[4], float SRC2[4]);
void SIMD_ADDPS(float DEST[4], float SRC1[4], float SRC2[4]);
void SIMD_SUBPS(float DEST[4], float SRC1[4], float SRC2[4]);
void SIMD_UMINPS(float DEST[4], float SRC1[4]);
void SIMD_MULPS(float DEST[4], float SRC1[4], float SRC2[4]);
void SIMD_DIVPD(double DEST[2], double SRC1[2], double SRC2[2]);
void SIMD_MULPD(double DEST[2], double SRC1[2], double SRC2[2]);
#ifdef WITH_TRIGO
void SIMD_SINPD(double DEST[2], double SRC1[2]);
void SIMD_COSPD(double DEST[2], double SRC1[2]);
#endif
void SIMD_ADDPD(double DEST[2], double SRC1[2], double SRC2[2]);
void SIMD_SUBPD(double DEST[2], double SRC1[2], double SRC2[2]);
void SIMD_UMINPD(double DEST[2], double SRC1[2]);
void SIMD_DIVPS(float DEST[4], float SRC1[4], float SRC2[4]);
void SIMD_MAXPS(float DEST[4], float SRC1[4], float SRC2[4]);
void SIMD_LOAD_V2SI_TO_V2SF(float TO[2], int VEC[2]);
void SIMD_STORE_V2SI_TO_V2SF(int TO[2], float VEC[2]);
void SIMD_LOAD_CONSTANT_V2SF(float VEC[2], float HIGH, float LOW);
void SIMD_LOAD_CONSTANT_V2SI(int VEC[2], int HIGH, int LOW);
void SIMD_LOAD_V2SI(int VEC[2], int BASE[2]);
void SIMD_LOAD_GENERIC_V2SI(int VEC[2], int X1, int X2);
void SIMD_STORE_V2SI(int VEC[2], int BASE[2]);
void SIMD_STORE_GENERIC_V2SI(int VEC[2], int X1[1], int X2[1]);
void SIMD_STORE_V2DI(int VEC[2], int BASE[2]);
void SIMD_ADDW(short DEST[8], short SRC1[8], short SRC2[8]);
void SIMD_SUBW(short DEST[8], short SRC1[8], short SRC2[8]);
void SIMD_MULW(short DEST[8], short SRC1[8], short SRC2[8]);
void SIMD_DIVW(short DEST[8], short SRC1[8], short SRC2[8]);
void SIMD_LOAD_GENERIC_V8HI(short VEC[8], short BASE0, short BASE1, short BASE2, short BASE3, short BASE4, short BASE5, short BASE6, short BASE7);
void SIMD_LOAD_V8HI(short VEC[8], short BASE[8]);
void SIMD_LOAD_V4QI_TO_V4HI(short VEC[4], char BASE[4]);
void SIMD_LOAD_V4HI_TO_V4SI(int DEST[4], short SRC[4]);
void SIMD_STORE_V8HI(short VEC[8], short BASE[8]);
void SIMD_STORE_V8HI_TO_V8SI(char vec[8],char arr[8]);
void SIMD_LOAD_V8SI_TO_V8HI(char vec[8],char arr[8]);
void SIMD_PHID(int DEST[4], int COND[4], int SRC1[4], int SRC2[4]);
void SIMD_ADDD(int DEST[4], int SRC1[4], int SRC2[4]);
void SIMD_SUBD(int DEST[4], int SRC1[4], int SRC2[4]);
void SIMD_MULD(int DEST[4], int SRC1[4], int SRC2[4]);
void SIMD_DIVD(int DEST[4], int SRC1[4], int SRC2[4]);
void SIMD_LOAD_CONSTANT_V8QI(char VEC[8], int HIGH, int LOW);
void SIMD_LOAD_V8QI(char VEC[8], char BASE[8]);
void SIMD_LOAD_GENERIC_V8QI(char VEC[8], char X1, char X2, char X3, char X4, char X5, char X6, char X7, char X8);
void SIMD_STORE_V8QI(char VEC[8], char BASE[8]);
void SIMD_STORE_GENERIC_V8QI(char VEC[8], char *X0, char X1[1], char X2[1], char X3[1], char X4[1], char X5[1], char X6[1], char X7[1]);
void SIMD_ADDB(char DEST[8], char SRC1[8], char SRC2[8]);
void SIMD_SUBB(char DEST[8], char SRC1[8], char SRC2[8]);
void SIMD_MULB(char DEST[8], char SRC1[8], char SRC2[8]);
void SIMD_MOVPS(float DEST[2], float SRC[2]);
void SIMD_MOVD(int DEST[2], int SRC[2]);
void SIMD_MOVW(short DEST[4], short SRC[4]);
void SIMD_MOVB(char DEST[8], char SRC[8]);
void SIMD_OPPPS(float DEST[2], float SRC[2]);
void SIMD_OPPD(int DEST[2], int SRC[2]);
void SIMD_OPPW(short DEST[4], short SRC[4]);
void SIMD_OPPB(char DEST[8], char SRC[8]);
void SIMD_SETPS(float DEST[4], float SRC[4]);
void SIMD_SETPD(double DEST[2], double SRC[2]);
void SIMD_SETD(int DEST[2], int SRC[2]);
void SIMD_SETW(short DEST[4], short SRC[4]);
void SIMD_SETB(char DEST[8], char SRC[8]);
void SIMD_LOAD_CONSTANT_V2DF(double vec[2], double v0, double v1);
"""

# taken from validation/SAC/include/sse.h r2291
sse_h = """
#include <xmmintrin.h>

typedef float  a2sf[2] __attribute__ ((aligned (16)));
typedef float  a4sf[4] __attribute__ ((aligned (16)));
typedef double a2df[2] __attribute__ ((aligned (16)));
typedef int	a4si[4] __attribute__ ((aligned (16)));

typedef __m128  v4sf;
typedef __m128d v2df;
typedef __m128i v4si;
typedef __m128i v8hi;

/* float */
#define SIMD_LOAD_V4SF(vec,arr) vec=_mm_loadu_ps(arr)
#define SIMD_LOADA_V4SF(vec,arr) vec=_mm_load_ps(arr)
#define SIMD_MULPS(vec1,vec2,vec3) vec1=_mm_mul_ps(vec2,vec3)
#define SIMD_DIVPS(vec1,vec2,vec3) vec1=_mm_div_ps(vec2,vec3)
#define SIMD_ADDPS(vec1,vec2,vec3) vec1=_mm_add_ps(vec2,vec3)
#define SIMD_SUBPS(vec1, vec2, vec3) vec1 = _mm_sub_ps(vec2, vec3)
/* umin as in unary minus */
#define SIMD_UMINPS(vec1, vec2)				\
		do {						\
		__m128 __pips_tmp;			\
		__pips_tmp = _mm_setzero_ps();		\
		vec1 = _mm_sub_ps(__pips_tmp, vec2);	\
		} while(0)

#define SIMD_STORE_V4SF(vec,arr) _mm_storeu_ps(arr,vec)
#define SIMD_STOREA_V4SF(vec,arr) _mm_store_ps(arr,vec)
#define SIMD_STORE_GENERIC_V4SF(vec,v0,v1,v2,v3)			\
		do {								\
		float __pips_tmp[4] __attribute__ ((aligned (16)));	\
		SIMD_STOREA_V4SF(vec,&__pips_tmp[0]);			\
		*(v0)=__pips_tmp[0];					\
		*(v1)=__pips_tmp[1];					\
		*(v2)=__pips_tmp[2];					\
		*(v3)=__pips_tmp[3];					\
		} while (0)

#define SIMD_ZERO_V4SF(vec) vec = _mm_setzero_ps()

#define SIMD_LOAD_GENERIC_V4SF(vec,v0,v1,v2,v3)				\
		do {								\
		float __pips_v[4] __attribute ((aligned (16)));\
		__pips_v[0]=v0;\
		__pips_v[1]=v1;\
		__pips_v[2]=v2;\
		__pips_v[3]=v3;\
		SIMD_LOADA_V4SF(vec,&__pips_v[0]);			\
		} while(0)

/* handle padded value, this is a very bad implementation ... */
#define SIMD_STORE_MASKED_V4SF(vec,arr)					\
		do {								\
		float __pips_tmp[4] __attribute__ ((aligned (16)));					\
		SIMD_STOREA_V4SF(vec,&__pips_tmp[0]);			\
		(arr)[0] = __pips_tmp[0];				\
		(arr)[1] = __pips_tmp[1];				\
		(arr)[2] = __pips_tmp[2];				\
		} while(0)

#define SIMD_LOAD_V4SI_TO_V4SF(v, f)		\
		do {					\
		float __pips_tmp[4];		\
		__pips_tmp[0] = (f)[0];		\
		__pips_tmp[1] = (f)[1];		\
		__pips_tmp[2] = (f)[2];		\
		__pips_tmp[3] = (f)[3];		\
		SIMD_LOAD_V4SF(v, __pips_tmp);	\
		} while(0)

/* double */
#define SIMD_LOAD_V2DF(vec,arr) vec=_mm_loadu_pd(arr)
#define SIMD_MULPD(vec1,vec2,vec3) vec1=_mm_mul_pd(vec2,vec3)
#define SIMD_ADDPD(vec1,vec2,vec3) vec1=_mm_add_pd(vec2,vec3)
#define SIMD_UMINPD(vec1, vec2)				\
		do {						\
		__m128d __pips_tmp;			\
		__pips_tmp = _mm_setzero_pd();		\
		vec1 = _mm_sub_pd(__pips_tmp, vec2);	\
		} while(0)

#define SIMD_COSPD(vec1, vec2)						\
		do {								\
		double __pips_tmp[2] __attribute__ ((aligned (16)));	\
		SIMD_STORE_V2DF(vec2, __pips_tmp);			\
		__pips_tmp[0] = cos(__pips_tmp[0]);			\
		__pips_tmp[1] = cos(__pips_tmp[1]);			\
		SIMD_LOAD_V2DF(vec2, __pips_tmp);			\
		} while(0)

#define SIMD_SINPD(vec1, vec2)						\
		do {								\
		double __pips_tmp[2] __attribute__ ((aligned (16)));	\
		SIMD_STORE_V2DF(vec2, __pips_tmp);			\
		__pips_tmp[0] = sin(__pips_tmp[0]);			\
		__pips_tmp[1] = sin(__pips_tmp[1]);			\
		} while(0)

#define SIMD_STORE_V2DF(vec,arr) _mm_storeu_pd(arr,vec)
#define SIMD_STORE_GENERIC_V2DF(vec, v0, v1)	\
		do {					\
		double __pips_tmp[2];			\
		SIMD_STORE_V2DF(vec,&__pips_tmp[0]);	\
		*(v0)=__pips_tmp[0];			\
		*(v1)=__pips_tmp[1];			\
		} while (0)
#define SIMD_LOAD_GENERIC_V2DF(vec,v0,v1)	\
		do {					\
		double v[2] = { v0,v1};		\
		SIMD_LOAD_V2DF(vec,&v[0]);	\
		} while(0)

/* conversions */
#define SIMD_STORE_V2DF_TO_V2SF(vec,f)			\
		do {						\
		double __pips_tmp[2];			\
		SIMD_STORE_V2DF(vec, __pips_tmp);	\
		(f)[0] = __pips_tmp[0];			\
		(f)[1] = __pips_tmp[1];			\
		} while(0)

#define SIMD_LOAD_V2SF_TO_V2DF(vec,f)		\
		SIMD_LOAD_GENERIC_V2DF(vec,(f)[0],(f)[1])

/* char */
#define SIMD_LOAD_V8HI(vec,arr) \
		vec = (__m128i*)(arr)

#define SIMD_STORE_V8HI(vec,arr)\
		*(__m128i *)(&(arr)[0]) = vec

#define SIMD_STORE_V8HI_TO_V8SI(vec,arr)\
	SIMD_STORE_V8HI(vec,arr)
#define SIMD_LOAD_V8SI_TO_V8HI(vec,arr)\
	SIMD_LOAD_V8HI(vec,arr)


"""

Threednow_h = """
#include <mm3dnow.h>

typedef float a2sf[2] __attribute__ ((aligned (16)));
typedef __m64 v2sf;

typedef int	a2si[2] __attribute__ ((aligned (16)));
typedef __m64 v2si;

#define SIMD_LOAD_V2SF(vec, f)			\
		vec = *(const __m64 *) &(f)[0]
#define SIMD_LOAD_GENERIC_V2SF(vec, f0,f1)			\
		do {\
			a2sf __tmp;\
			__tmp[0]=f0;\
			__tmp[1]=f1;\
			vec = *(const __m64 *) &(__tmp)[0];\
			} while(0)

#define SIMD_STORE_V2SF(vec, f)			\
		*(__m64 *)(&(f)[0]) = vec

#define SIMD_MULPS(vec1, vec2, vec3)		\
		vec1 = _m_pfmul(vec2, vec3)

#define SIMD_ADDPS(vec1, vec2, vec3)		\
		vec1 = _m_pfadd(vec2, vec3)

#define SIMD_SUBPS(vec1, vec2, vec3)		\
		vec1 = _m_pfsub(vec2, vec3)

/* should not be there :$ */
#define SIMD_ZERO_V4SF(vec) \
		SIMD_SUBPS(vec,vec,vec)

#define SIMD_UMINPS(vec1, vec2)					\
		do {							\
		__m64 __pips_tmp;				\
		__pips_tmp = _m_pxor(__pips_tmp, __pips_tmp);	\
		SIMD_SUBPS(vec1, __pips_tmp, vec2);		\
		} while(0)

#define SIMD_LOAD_V2SI_TO_V2SF(vec, f)		\
		do {					\
		float __pips_f[2];		\
		__pips_f[0] = (f)[0];		\
		__pips_f[1] = (f)[1];		\
		SIMD_LOAD_V2SF(vec, __pips_f);	\
		} while (0)

"""
