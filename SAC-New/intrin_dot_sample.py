import re
import tempfile
import shutil
from sys import stdin
from os import fdopen
from pyps import *

BEGIN_RECOVERY="/* begin recovery {*/\n"
END_RECOVERY="/*} end recovery */\n"

def header_generator(fh):
	fh.write(BEGIN_RECOVERY)
	fh.write("""
typedef struct { short array[4]; } __m64;
extern __m64  _mm_setzero_si64();
extern __m64 _m_pmaddwd(__m64,__m64);
extern __m64 _m_paddw(__m64,__m64);
extern short _m_to_int(__m64);
extern __m64 _m_psrlqi(__m64,unsigned int);
#define _m_empty() // cleared
""")
	fh.write(END_RECOVERY)

def stub_generator():
	(fd,path)=tempfile.mkstemp(suffix=".c",text=True)
	stub=fdopen(fd,'w')
	header_generator(stub)
	stub.write("""
	__m64 simd_mm_set_zero(__m64 a) {
		return a = _mm_setzero_si64();
	}
	__m64 simd_m_pmaddwd(__m64 a, __m64 *b, __m64 *c) {
		return a = _m_pmaddwd(*b,*c);
	}
	__m64 simd_m_paddw(__m64 a,__m64 b, __m64 c) {
		return a = _m_paddw(b,c);
	}
	__m64 simd_load_w(__m64* a, short b) {
		return a = &b;
	}
	__m64 simd_m_to_int(__m64 a, short b) {
		return b = _m_to_int(a);
	}
	__m64 simd_m_psrlqi(__m64 a,__m64 b) {
		return a = _m_psrlqi(b,32);
	}

	""")
	stub.close()
	#print "generating stub in " , path
	return (path,["simd_load_w","simd_mm_set_zero","simd_m_pmaddwd","simd_m_paddw","simd_m_to_int","simd_m_psrlqi"])

def detect_intrinsic(input,output):
	# rework input files
	(stub,funcs)=stub_generator()
	ninput="/tmp/"+input
	src=open(input,"r") ; nsrc=open(ninput,"w")
	header_generator(nsrc)
	for line in src:
		nsrc.write(line if not re.match("\s*#\s*include\s*<pmmintrin\.h>\s*",line) else "// " + line)
	src.close() ; nsrc.close()
	# use the ability (!) of pips to bypass dir name
	# so that we get the same file name with a different source
	w=workspace([ninput,stub])
	m=w['MMX_dot_product']
	for func in funcs:
		m.expression_substitution(pattern=func)
	m.display()
	w.save(indir=output)
	w.close()

def add_intrinsic_header(input, output):
	fdi=open(input,"r")
	fdo=open(output,"w")
	fdo.write("""
	#define simd_mm_set_zero(a) SIMD_LOAD_GENERIC_V4SI(a,0,0,0,0)
	#define simd_load_w(a,b)  a=&b //SIMD_LOAD_V4SI(a,&b)
	#define simd_m_pmaddwd(a,b,c) SIMD_MULW(a,b,c)
	#define simd_m_paddw(a,b,c) SIMD_ADDW(a,b,c)
	#define simd_m_to_int(a,b) b=a[3];
	#define simd_m_psrlqi(a,b) b[3]=a[2];
	typedef short __m64[4];
	""")
	for line in fdi:
		if line == END_RECOVERY:break
	for line in fdi:
		fdo.write(re.sub("\*\s*ptr"," ptr",line))
	fdi.close() ; fdo.close()


def run_sac(input,outdir):
	ws=workspace([input,"include/SIMD_64.c","include/SIMD_128.c"],
			activates=["RICE_SEMANTICS_DEPENDENCE_GRAPH", "MUST_REGIONS", "PRECONDITIONS_INTER_FULL", "TRANSFORMERS_INTER_FULL"]
			)
	# some more restructuring properties
	ws.set_property(
			UNSPAGHETTIFY_WHILE_RECOVER=True,
			FOR_TO_WHILE_LOOP_IN_CONTROLIZER=False,
			RICEDG_STATISTICS_ALL_ARRAYS=True)
	
	# sac global properties
	ws.set_property(SIMD_FORTRAN_MEM_ORGANISATION=False,SAC_SIMD_REGISTER_WIDTH=128)
	
	# first try to recover as much do loops as possible
	# sac if-conversion fails in presence of while loops :'(
	ws.all.restructure_control()
	ws.all.recover_for_loop()
	m=ws["MMX_dot_product"]
	m.display()
	m.unfolding()
	m.display()
	#m.flatten_code(unroll=False)
	#m.common_subexpression_elimination(skip_added_constant=True,skip_lhs=False)
	#m.display()
	#m.display()
	#m.display(With='PRINT_CODE_PROPER_REDUCTIONS')
	#m.display(With='PRINT_CODE_CUMULATED_REDUCTIONS')
	#m.display()
	#m.partial_eval(always_simplify=True)
	#m.if_conversion_init()
	#m.display()
	#m.if_conversion()
	#m.display()
	#m.if_conversion_compact()
	#m.display()
	#m.localize_declaration()
	#m.display()
	#m.simdizer_auto_unroll(simple_calculation=False,minimize_unroll=True,loop_unroll_merge=True)
	for l in m.loops():
		l.unroll(rate=2,loop_unroll_merge=False)
	#	m.suppress_dead_code()
	m.recompile_module()
	m.display()
	#for l in m.loops():
	#	l.reduction_variable_expansion()
	#m.suppress_dead_code()
	m.partial_eval(always_simplify=True)
	m.display()
	m.flatten_code(unroll=False)
	m.recompile_module()
	m.display()
	m.simd_remove_reductions()
	m.recompile_module()
	m.display()
	#m.display()
	#m.forward_substitute(optimistic_clean=True)
	#m.simd_atomizer()
	#m.single_assignment()
	#m.display()
	#m.flatten_code(unroll=False)
	#m.suppress_dead_code()
#	m.clean_declarations()
	m.simdizer()
	m.display()
	m.simd_loop_const_elim()
	m.display()
	#m.clean_declarations()
	#m.display()
	#ws.save(indir=outdir)



if __name__ == "__main__":
	input="intrin_dot_sample.c"
	tmpdir="detect"
	tmpdir2="with_header"
	outdir="intrin_dot_sample"
	detect_intrinsic(input,tmpdir)
	try:
		os.mkdir(tmpdir2)
	except os.error:
		pass
	add_intrinsic_header(tmpdir+"/"+input,tmpdir2+"/"+input)
	run_sac(tmpdir2+"/"+input,outdir)
	




