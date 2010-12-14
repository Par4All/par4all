from pyps import module, workspace
import terapyps_asm
import pypsutils
from subprocess import Popen, PIPE
import os,sys,shutil,tempfile

def generate_check_ref(self):
	"""Generate a reference run for workspace"""
	a_out=self.compile()
	pid=Popen("./"+a_out,stdout=PIPE)
	if pid.wait() == 0:
		self.ref=pid.stdout.readlines()
	else :
		exit(1)
workspace.generate_check_ref=generate_check_ref

def check(self,debug):
	if debug:
		a_out=self.compile()
		pid=Popen("./"+a_out,stdout=PIPE)
		if pid.wait() == 0:
			if self.ref!=pid.stdout.readlines():
				print "**** check failed *****"
				exit(1)
			else:
				print "**** check ok ******"
		else :
			exit(1)
workspace.check=check

dma="terapix"
assembly="terasm"
runtime={}

class workspace:
	"""A Terapix workspace"""
	def __init__(self, ws, sources, **args):
		"""Add terapix runtime to the workspace"""
		self.tmpdir = tempfile.mkdtemp()
		for src in [ dma, assembly ]:
			tmpC=self.tmpdir + "/" + src + ".c"
			pypsutils.string2file(runtime[src], tmpC)
			ws._sources.append(tmpC)

	def post_init(self,sources, **args):
		"""Clean tmp files"""
		shutil.rmtree(self.tmpdir)

		
		

def smart_loop_expansion(m,l,sz,debug,center=False):
	""" smart loop expansion, has a combinaison of loop_expansion_init, statement_insertion and loop_expansion """
	l.loop_expansion_init(loop_expansion_size=sz)
	if debug:m.display()
	m.statement_insertion()
	if debug:m.display()
	l.loop_expansion(size=sz,center=center)
	if debug:m.display()
	m.partial_eval()
	m.invariant_code_motion()
	m.redundant_load_store_elimination()
	if debug:m.display()




module.smart_loop_expansion=smart_loop_expansion

def vconv(tiling_vector):
	return ",".join(tiling_vector)

def all_callers(m):
	callers=[]
	for i in m.callers:
		if i not in callers:
			callers.append(i)
			for j in all_callers(i):
				callers.append(j)
	return callers


def terapix_code_generation(m,nbPE=128,memoryPE=512,debug=False):
	"""Generate terapix code for m if it's not part of the runtime """
	if m.cu in runtime.keys():return
	w=m._ws
	w.generate_check_ref()
	# choose the proper analyses and properties
	w.props.constant_path_effects=False
	w.activate(module.must_regions)
	w.activate(module.transformers_inter_full)
	w.activate(module.interprocedural_summary_precondition)
	w.activate(module.preconditions_inter_full)
	w.activate(module.region_chains)
	w.props.semantics_trust_array_declarations=True

	if debug:print "tidy the code just in case of"
	m.partial_eval()
	 
	#print "I have to do this early"
	if debug:m.display()
	m.recover_for_loop()
	m.for_loop_to_do_loop()
	if debug:m.display()
	unknown_width="__TERAPYPS_HEIGHT"
	unknown_height="__TERAPYPS_WIDTH"
	m.run(["sed","-e","3 i    unsigned int "
		+ unknown_height + ", " + unknown_width + ";"])
	if debug:m.display()
	tiling_vector=[unknown_height, unknown_width]

	
	print "tiling"
	for l in m.loops():
		if l.loops():
				# this take care of expanding the loop in order to match number of processor constraint
				m.smart_loop_expansion(l,tiling_vector[0],debug)
				# this take care of expanding the loop in order to match memory size constraint
				m.smart_loop_expansion(l.loops()[0],tiling_vector[1],debug)
				l.symbolic_tiling(force=True,vector=vconv(tiling_vector))
				if debug:m.display()
				#m.icm()
				if debug:m.display()

	print "group constants and isolate"
	kernels=[]
	for l0 in m.loops():
		for l1 in l0.loops():
			for l2 in l1.loops():
				if debug:m.display(activate="PRINT_CODE_REGIONS")
				m.solve_hardware_constraints(label=l2.label,unknown=tiling_vector[0],limit=nbPE,type="NB_PROC")
				if debug:m.display()
				m.partial_eval()
				if debug:m.display()
				m.solve_hardware_constraints(label=l2.label,unknown=tiling_vector[1],limit=memoryPE*nbPE,type="VOLUME")
				if debug:m.display()
				if debug:m.display(activate="PRINT_CODE_REGIONS")
				m.partial_eval()
				if debug:m.display()
				m.forward_substitute()
				if debug:m.display()
				m.redundant_load_store_elimination()
				m.clean_declarations()
	#			m.group_constants(layout="terapix",statement_label=l2.label,skip_loop_range=True)
				if debug:m.display(activate="PRINT_CODE_REGIONS")
				m.smart_loop_expansion(l2,str(nbPE),debug,center=False)
				if debug:m.display(activate="PRINT_CODE_REGIONS")
				m.array_expansion()
				if debug:m.display()
				for k in all_callers(m):
					if debug:k.display(activate="PRINT_CODE_REGIONS")
					k.array_expansion()
					if debug:k.display()
				w.check(debug)

	for l0 in m.loops():
		for l1 in l0.loops():
			for l2 in l1.loops():
				if debug:m.display(activate="PRINT_CODE_REGIONS")
				kernels+=[l2]
				m.group_constants(layout="terapix",statement_label=l2.label,skip_loop_range=True)
				if debug:m.display()
				m.isolate_statement(label=l2.label)
	if debug:m.display()
	m.loop_normalize(one_increment=True,skip_index_side_effect=True,lower_bound=0)
	if debug:m.display()
	m.partial_eval()
	if debug:m.display()
	#m.iterator_detection()
	#m.array_to_pointer(convert_parameters="POINTER",flatten_only=False)
	#m.display(activate="PRINT_CODE_PROPER_EFFECTS")
	#m.common_subexpression_elimination(skip_lhs=False)
	#m.simd_atomizer(atomize_reference=True,atomize_lhs=True)
	#m.invariant_code_motion(CONSTANT_PATH_EFFECTS=False)
	#m.icm(CONSTANT_PATH_EFFECTS=False)
	#m.display()
	
	print "outlining to launcher"
	seed,nb="launcher_",0
	launchers=[]
	for k in kernels:
		name=seed+str(nb)
		nb+=1
		m.privatize_module()
		m.outline(module_name=name,label=k.label,smart_reference_computation=True,loop_bound_as_parameter=k.loops()[0].label)
		launchers+=[w[name]]
	if debug:m.display()
	if debug:
		for l in launchers:l.display(activate='PRINT_CODE_REGIONS')
	
	print "outlining to microcode"
	microcodes=[]
	for l in launchers:
		if debug:l.display()
		theloop=l.loops()[0]
		#l.loop_normalize(one_increment=True,lower_bound=0)
		#l.redundant_load_store_elimination()
		if debug:l.display()
		name=l.name+"_microcode"
		loop_to_outline=theloop.loops()[0]
		print "label:" , loop_to_outline.label
		l.outline(module_name=name,label=loop_to_outline.label,smart_reference_computation=True)
		if debug:m.display()
		mc=w[name]
		if debug:l.display()
		if debug:mc.display()
		microcodes+=[mc]
	w.check(debug)
	print "refining microcode"
	for m in microcodes:
		m.loop_normalize(one_increment=True,lower_bound=0)
		m.redundant_load_store_elimination()
		if debug:m.display()
		m.flatten_code(flatten_code_unroll=False)
		if debug:m.display()
		m.linearize_array(use_pointers=True)
		if debug:m.display()
		m.strength_reduction()
		m.forward_substitute()
		m.redundant_load_store_elimination()
		if debug:m.display()
		m.split_update_operator()
		m.simd_atomizer(atomize_reference=True,atomize_lhs=True)
		m.generate_two_addresses_code()
		w.check(debug)
		if debug:m.display()
		m.flatten_code(flatten_code_unroll=False)
		m.clean_declarations()
		if debug:m.display()
		w.check(debug)
		m.normalize_microcode()
		m.clean_declarations()
		if debug:
			m.display()
			m.callers.display()
	w.check(debug)

	# generate assembly
	for m in microcodes:
		for asm in w.fun:
			if asm.cu == assembly:
				m.expression_substitution(asm.name)
		if debug:m.display()
		terapyps_asm.conv(w.dirname()+m.show("printed_file"),sys.stdout)

module.terapix_code_generation=terapix_code_generation

# the dma runtime
runtime[dma]="""
#include <stdlib.h>

/* A small implementation of the runtime used by the code generated by the
   kernel_load_store and isolate_statement
*/


/* To copy scalars */
void P4A_copy_from_accel(size_t element_size,
			 void *host_address,
			 void *accel_address) {
  size_t i;
  char * cdest = host_address;
  char * csrc = accel_address;
  for(i = 0; i < element_size; i++)
    cdest[i] = csrc[i];
}


void P4A_copy_to_accel(size_t element_size,
		       void *host_address,
		       void *accel_address) {
  size_t i;
  char * cdest = accel_address;
  char * csrc = host_address;
  for(i = 0; i < element_size; i++)
    cdest[i] = csrc[i];
}


/* To copy parts of 1D arrays */
void P4A_copy_from_accel_1d(size_t element_size,
			    size_t d1_size,
			    size_t d1_block_size,
			    size_t d1_offset,
			    void *host_address,
			    void *accel_address) {
  size_t i;
  char * cdest = d1_offset*element_size + (char *)host_address;
  char * csrc = accel_address;
  for(i = 0; i < d1_block_size*element_size; i++)
    cdest[i] = csrc[i];
}


void P4A_copy_to_accel_1d(size_t element_size,
			  size_t d1_size,
			  size_t d1_block_size,
			  size_t d1_offset,
			  void *host_address,
			  void *accel_address) {
  size_t i;
  char * cdest = accel_address;
  char * csrc = d1_offset*element_size + (char *)host_address;
  for(i = 0; i < d1_block_size*element_size; i++)
    cdest[i] = csrc[i];
}


/* To copy parts of 2D arrays */
void P4A_copy_from_accel_2d(size_t element_size,
			    size_t d1_size, size_t d2_size,
			    size_t d1_block_size, size_t d2_block_size,
			    size_t d1_offset, size_t d2_offset,
			    void *host_address,
			    void *accel_address) {
  size_t i, j;
  char * cdest = d2_offset*element_size + (char*)host_address;
  char * csrc = (char*)accel_address;
  for(i = 0; i < d1_block_size; i++)
    for(j = 0; j < d2_block_size*element_size; j++)
      cdest[(i + d1_offset)*element_size*d2_size + j] =
        csrc[i*element_size*d2_block_size + j];
}


void P4A_copy_to_accel_2d(size_t element_size,
			  size_t d1_size, size_t d2_size,
			  size_t d1_block_size, size_t d2_block_size,
			  size_t d1_offset,   size_t d2_offset,
			  void *host_address,
			  void *accel_address) {
  size_t i, j;
  char * cdest = (char *)accel_address;
  char * csrc = d2_offset*element_size + (char *)host_address;
  for(i = 0; i < d1_block_size; i++)
    for(j = 0; j < d2_block_size*element_size; j++)
      cdest[i*element_size*d2_block_size + j] =
        csrc[(i + d1_offset)*element_size*d2_size + j];
}

/* Allocate memory on the accelerator */
void P4A_accel_malloc(void **ptr, size_t n) {
    if(n) *ptr=malloc(n);
    else *ptr=NULL;
}

/* Deallocate memory on the accelerator */
void P4A_accel_free(void *ptr) {
    free(ptr);
}
"""

# the assembly patterns
runtime[assembly]="""
#define _OP(a,b) a##b
#define OP(a,b) _OP(a,b)
#define TYPE int
#define SUFF i

TYPE OP(add,SUFF)(TYPE lhs, TYPE rhs)
{
    return lhs=lhs+rhs;
}
TYPE OP(addr,SUFF)(TYPE lhs, TYPE *rhs)
{
    return lhs=lhs+*rhs;
}
TYPE OP(sub,SUFF)(TYPE lhs, TYPE rhs)
{
    return lhs=lhs-rhs;
}
TYPE OP(subr,SUFF)(TYPE lhs, TYPE *rhs)
{
    return lhs=lhs-*rhs;
}
TYPE OP(lshift,SUFF)(TYPE lhs, TYPE rhs)
{
    return lhs=lhs<<rhs;
}
TYPE OP(rshift,SUFF)(TYPE lhs, TYPE rhs)
{
    return lhs=lhs>>rhs;
}
TYPE OP(prshift,SUFF)(TYPE *lhs, TYPE rhs)
{
    return *lhs=*lhs>>rhs;
}
TYPE OP(mul,SUFF)(TYPE lhs, TYPE rhs)
{
    return lhs=lhs*rhs;
}
TYPE OP(div,SUFF)(TYPE lhs, TYPE rhs)
{
    return lhs=lhs/rhs;
}
TYPE OP(mulr,SUFF)(TYPE lhs, TYPE *rhs)
{
    return lhs=lhs**rhs;
}
TYPE OP(set,SUFF)(TYPE lhs, TYPE rhs)
{
    return lhs=rhs;
}
TYPE OP(pset,SUFF)(TYPE *lhs, TYPE rhs)
{
    return *lhs=rhs;
}
TYPE OP(setp,SUFF)(TYPE lhs, TYPE *rhs)
{
    return lhs=*rhs;
}
TYPE OP(psetp,SUFF)(TYPE *lhs, TYPE *rhs)
{
    return *lhs=*rhs;
}
TYPE* OP(padd,SUFF)(TYPE *lhs, int rhs)
{
    return lhs=lhs+rhs;
}
TYPE* OP(psub,SUFF)(TYPE *lhs, int rhs)
{
    return lhs=lhs-rhs;
}
"""
