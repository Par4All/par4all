# -*- coding: utf-8 -*-
from __future__ import with_statement 
import os,sys
import re
import shutil
import pypsutils
import subprocess
import pyps

simd_h = "SIMD_types.h"

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
    m.partial_eval()
    extram=list()
    for l in kernels:
        mn=m.name+"_"+l.label
        m.outline(module_name=mn,label=l.label)
        lm=m.workspace[mn]
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
        ws = module.workspace
        if not cond.has_key("verbose"):
            cond["verbose"] = ws.verbose
        # Here are the transformations made by benchmark.tpips.h, blindy
        # translated in pyps.

        ws.activate("preconditions_intra")
        ws.activate("transformers_intra_full")

        ws.props.loop_unroll_with_prologue = False
        ws.props.constant_path_effects = False
        #ws.props.ricedg_statistics_all_arrays = True
        ws.props.c89_code_generation = True


        ws.props.simd_fortran_mem_organisation = False
        ws.props.sac_simd_register_width = cond["register_width"]
        ws.props.prettyprint_all_declarations = True
        ws.props.compute_all_dependences = True
        module.recover_for_loop()
        module.for_loop_to_do_loop()
        module.split_initializations()

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
            module.partial_eval()
        
            module.simd_remove_reductions()
            if cond.get("verbose"):
                module.display()

            for p in ( "__PIPS_SAC_MULADD" , ):
                module.expression_substitution(pattern=p)

            module.simd_atomizer()
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

            try:
                module.delay_communications_intra()
                module.flatten_code(unroll = False)
            except RuntimeError: pass

            module.redundant_load_store_elimination()
            module.clean_declarations()

            # In the end, uses the real SIMD_ZERO_* functions if necessary
            # This would have been "perfect" (as much as perfect this
            # method is...), but PIPS isn't aware of (a|v)4sf and
            # other vector types...
            #module.modify(gen_simd_zeros)

            if cond.get("verbose"):
                module.display()


class sacsse(sacbase):
    register_width = 128
    hfile = "sse.h"
    makefile = "Makefile.sse"
    ext = "sse"
    @staticmethod
    def sac(module, **kwargs):
        kwargs["register_width"] = sacsse.register_width
        sacbase.sac(module, **kwargs)

class sac3dnow(sacbase):
    register_width = 64
    hfile = "threednow.h"
    makefile = "Makefile.3dn"
    ext = "3dn"
    @staticmethod
    def sac(module, *args, **kwargs):
        kwargs["register_width"] = sac3dnow.register_width
        # 3dnow supports only floats
        for line in module.code():
            if re.search("double", line) or re.search(r"\b(cos|sin)\b", line):
                raise RuntimeError("Can't vectorize double operations with 3DNow!")
        sacbase.sac(module, *args, **kwargs)

class sacavx(sacbase):
    register_width = 256
    hfile = "avx.h"
    makefile = "Makefile.avx"
    ext = "avx"
    @staticmethod
    def sac(module, *args, **kwargs):
        kwargs["register_width"] = sacavx.register_width
        sacbase.sac(module, *args, **kwargs)

class sacneon(sacbase):
    register_width = 128
    hfile = "neon.h"
    makefile = "Makefile.neon"
    ext = "neon"
    @staticmethod
    def sac(module, *args, **kwargs):
        kwargs["register_width"] = sacneon.register_width
        sacbase.sac(module, *args, **kwargs)

class workspace(pyps.workspace):
    """The SAC subsystem, in Python.

    Add a new transformation, for adapting code to SIMD instruction
    sets (SSE, 3Dnow, AVX and ARM NEON)"""

    patterns_h = "patterns.h"
    patterns_c = "patterns.c"
    simd_c = "SIMD.c"

    def __init__(self, *sources, **kwargs):
        drivers = {"sse": sacsse, "3dnow": sac3dnow, "avx": sacavx, "neon": sacneon}
        self.driver = drivers[kwargs.get("driver", "sse")]
        #Warning: this patches every modules, not only those of this workspace 
        pyps.module.sac=self.driver.sac
        # Add -DRWBITS=self.driver.register_width to the cppflags of the workspace
        kwargs['cppflags'] = kwargs.get('cppflags',"")+" -DRWBITS=%d " % (self.driver.register_width)
        super(workspace,self).__init__(pypsutils.get_runtimefile(self.simd_c,"sac"), pypsutils.get_runtimefile(self.patterns_c,"sac"), *sources, **kwargs)

    def save(self, rep=None):
        """Add $driver.h, which replaces general purpose SIMD instructions
        with machine-specific ones."""
        if rep == None:
            rep = self.tmpdirname
        
        (files,headers) = super(workspace,self).save(rep)
        

        #run gen_simd_zeros on every file
        for file in files:
            with open(file, 'r') as f:
                read_data = f.read()
            read_data = gen_simd_zeros(read_data)
            with open(file, 'w') as f:
                f.write(read_data)
        
        # Generate SIMD.h according to the register width
        # thanks to gcc -E and cproto (ugly, need something
        # better)
        simd_h_fname = os.path.abspath(rep + "/SIMD.h")
        simd_c_fname = os.path.abspath(rep + "/SIMD.c")
        p = subprocess.Popen("gcc -DRWBITS=%d -E %s |cproto" % (self.driver.register_width, simd_c_fname), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (simd_cus_header,serr) = p.communicate()
        if p.returncode != 0:
            raise RuntimeError("Error while creating SIMD.h: command returned %d.\nstdout:\n%s\nstderr:\n%s\n" % (p.returncode, simd_cus_header, serr))

        p = subprocess.Popen("gcc -DRWBITS=%d -E %s |cproto" % (self.driver.register_width, self.simd_c), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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
                pypsutils.addBeginnning(fname, '#include "'+self.patterns_h+'"\n')
        
        # Add header to the save rep
        shutil.copy(pypsutils.get_runtimefile(simd_h,"sac"),rep)
        shutil.copy(pypsutils.get_runtimefile(self.patterns_h,"sac"),rep)
        return files,headers+[os.path.join(rep,simd_h),os.path.join(rep,self.patterns_h)]


    def get_sac_maker(self,Maker=pyps.Maker):
        """Calls sacMaker to return a maker class using the driver set in the workspace"""
        return sacMaker(Maker,self.driver)



def sacMaker(Maker,driver):
    """Returns a maker class inheriting from the Maker class given in the arguments and using the driver given in the arguments"""
    class C(Maker):
        """Maker class inheriting from Maker"""

        def get_ext(self):
            return "."+driver.ext+super(C,self).get_ext()

        def get_makefile_info(self):
            return [ ( "sac", driver.makefile ) ] + super(C,self).get_makefile_info()

        def generate(self,path,sources,cppflags="",ldflags=""):
            newsources = []    
            for fname in sources:
                #change the includes
                filestring = pypsutils.file2string(os.path.join(path,fname))
                filestring= re.sub('#include "'+simd_h+'"','#include "'+driver.hfile+'"',filestring)
                newcfile = "sac_"+fname
                pypsutils.string2file(filestring,os.path.join(path,newcfile))
                newsources.append(newcfile)
            #create symlink .h file
            hpath = os.path.join(path,driver.hfile)
            if not os.path.exists(hpath):
                shutil.copy(pypsutils.get_runtimefile(driver.hfile,"sac"),hpath)
            
            makefile,others = super(C,self).generate(path,newsources,cppflags,ldflags)
            return makefile,others+newsources+[driver.hfile]

    return C    
