from __future__ import with_statement # this is to work with python2.5
from pyps import module
import pyps
import terapyps_asm
import pypsutils
import os, re
import pyrops

_microcode_suffix = "_microcode"


def generate_check_ref(self):
    """Generate a reference run for workspace"""
    self.compile(rule="mrproper")
    o=self.compile()
    (rc,out,err)=self.run(o)
    if rc == 0:
        self.ref=out
    else :
        print err
        exit(1)
pyps.workspace.generate_check_ref=generate_check_ref

def check(self,debug):
    if debug:
        self.compile(rule="mrproper")
        o=self.compile()
        (rc,out,err)=self.run(o)
        if rc == 0:
            if self.ref!=out:
                print "**** check failed *****"
                print "**** expected ****"
                print self.ref
                print "**** found ****"
                print out
                exit(1)
            else:
                print "**** check ok ******"
        else :
            exit(1)
pyps.workspace.check=check

dma="terapix.c"
assembly="terasm.c"
runtime=["terapix", "terasm"]

class workspace(pyps.workspace):
    """A Terapix workspace"""
    def __init__(self, *sources, **kwargs):
        """Add terapix runtime to the workspace"""
        super(workspace,self).__init__(pypsutils.get_runtimefile(dma,"terapyps"),pypsutils.get_runtimefile(assembly,"terapyps"), *sources, **kwargs)

class Maker(pyps.Maker):

    __ext = ".terass"
    __makefile = "Makefile.terapix"

    def get_ext(self):
        return self.__ext+super(Maker,self).get_ext()
    def get_makefile_info(self):
        return [ ("terapyps", self.__makefile) ] + super(Maker,self).get_makefile_info()
    def generate(self,path,sources,cppflags="",ldflags=""):
        class myworkspace(pyrops.pworkspace,workspace):
            def __init__(self,*sources,**kwargs):
                super(myworkspace,self).__init__(*sources,**kwargs)

        pruned_sources=filter(lambda x: not x in [dma,assembly] , sources)
        pruned_sources=map(lambda x: os.path.join(path,x),pruned_sources)
        kwargs={"cppflags":cppflags+" -I"+path}
        w=myworkspace(*pruned_sources,**kwargs)
        # generate assembly
        microcodes=w.filter(lambda m : m.name[-len(_microcode_suffix):] == _microcode_suffix)
        new_sources=[]
        for m in microcodes:
            #for asm in w.fun:
            #    if asm.cu == runtime[1]:
            #        m.expression_substitution(asm.name)
            for width in map(int,re.findall("#pragma terapix (?:\w+) (?:\d+) (\d+)",m.code)):
                print "## found width", width
                newcode=m.code
                numbers=re.findall("(\d+)",m.code)
                print "## found numbers ",numbers
                for num in map(int,numbers):
                    if num % width == 0 and num != 0:
                        newcode=re.sub(str(num),str(num/width)+"*S" ,newcode)
                m.code=newcode


            if w.verbose:m.display()
            new_source=m.name+".asl"
            with file(os.path.join(path,new_source),"w") as fd:
                terapyps_asm.conv(os.path.join(w.dirname,m.show("printed_file")),fd)
            new_sources.append(new_source)
        w.close()
        makefile,others = super(Maker,self).generate(path,new_sources,cppflags,ldflags)
        return makefile,others+new_sources
#def smart_loop_expansion(m,l,sz,debug,center=False):
#    """ smart loop expansion, has a combinaison of loop_expansion_init, statement_insertion and loop_expansion """
#    l.loop_expansion_init(loop_expansion_size=sz)
#    m.statement_insertion()
#    l.loop_expansion(size=sz,center=center)
#    m.partial_eval()
#    #m.invariant_code_motion()
#    m.redundant_load_store_elimination()
#    if debug:m.display()




#module.smart_loop_expansion=smart_loop_expansion

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
    if m.cu in runtime:return
    w=m.workspace
    # do this early, before check generation
    #for l in m.loops():
    #    if l.loops() and l.pragma == "#pragma terapix":
    #        m.terapix_remove_divide()
    #        break
    if debug:m.display()
    w.generate_check_ref()
    # choose the proper analyses and properties
    w.props.constant_path_effects=False
    w.activate(module.must_regions)
    w.activate(module.transformers_inter_full)
    w.activate(module.interprocedural_summary_precondition)
    w.activate(module.preconditions_inter_full)
    w.activate(module.region_chains)
    w.props.semantics_trust_array_declarations=True
    w.props.prettyprint_sequential_style="do"

    if debug:print "tidy the code just in case of"
    m.loop_normalize(one_increment=True,skip_index_side_effect=True,lower_bound=0)
    m.partial_eval()
    if debug:m.display()
    w.check(debug)
     
    #print "I have to do this early"
    m.recover_for_loop()
    m.for_loop_to_do_loop()
    if debug:m.display()
    unknown_width="__TERAPYPS_HEIGHT"
    unknown_height="__TERAPYPS_WIDTH"
    tiling_vector=[unknown_height, unknown_width]

    print "detection and extraction"
    okernels=list()
    for l in m.loops():
        if l.loops() and l.pragma == "#pragma terapix":
            kname=m.name+"_"+l.label
            m.outline(module_name=kname,label=l.label)
            okernels.append(w[kname])

    
    print "tiling"
    for m in okernels:
        for l in m.loops():
            if l.loops():
                    m.run(["sed","-e","3 i    unsigned int "
                        + unknown_height + ", " + unknown_width + ";"
                        + "if("+ unknown_width + ">3 || 3<"+unknown_height +") {",
                        "-e", "$ i }" 
                        ])
                    if debug:m.display()
                    # this take care of expanding the loop in order to match number of processor constraint
                    #m.smart_loop_expansion(l,tiling_vector[0],debug)
                    # this take care of expanding the loop in order to match memory size constraint
                    #m.smart_loop_expansion(l.loops(0),tiling_vector[1],debug)
                    l.symbolic_tiling(force=True,vector=vconv(tiling_vector))
                    m.common_subexpression_elimination()
                    m.invariant_code_motion()
                    if debug:m.display()
                    #if debug:m.display()
                    m.loop_nest_unswitching()
                    m.partial_eval()
                    if debug:m.display()

                    #m.optimize_expressions("ICMCSE",MASK_EFFECTS_ON_PRIVATE_VARIABLES=False)
                    if debug:m.display()
                    #if debug:m.display(activate=module.print_code_preconditions)

        print "group constants and isolate"
        kernels=[]
        tiles=[]
        known_height=1
        known_width=1
        for l0 in m.loops():
            for l1 in l0.loops():
                l2=l1.loops(0) # we re only interested in the first tile
                m.solve_hardware_constraints(label=l2.label,unknown=tiling_vector[0],limit=nbPE,type="NB_PROC")
                m.partial_eval()
                if debug:m.display()
                m.solve_hardware_constraints(label=l2.label,unknown=tiling_vector[1],limit=memoryPE*nbPE,type="VOLUME")
                if debug:m.display()
                # gather the width / height
                known_height = re.findall(unknown_height+"\s*=\s*(\d+)",m.code)[0]
                known_width = re.findall(unknown_width+"\s*=\s*(\d+)",m.code)[0]
                w.check(debug)
                m.partial_eval()
                m.forward_substitute()
                m.redundant_load_store_elimination()
                m.clean_declarations()
                if debug:m.display()
                tname="tile_"+l2.label
                m.outline(label=l2.label,module_name=tname)
                tiles.append(w[tname])

        for T in tiles:
            l2 = T.loops(0)
            kernels+=[l2]
            T.group_constants(layout="terapix",statement_label=l2.label,skip_loop_range=True,literal=False)
            if debug:T.display()
            T.isolate_statement(label=l2.label)
            if debug:T.display()
            T.loop_normalize(one_increment=True,skip_index_side_effect=True,lower_bound=0)
            T.partial_eval()
            w.check(debug)
            #m.iterator_detection()
            #m.array_to_pointer(convert_parameters="POINTER",flatten_only=False)
            #m.display(activate="PRINT_CODE_PROPER_EFFECTS")
            #m.common_subexpression_elimination(skip_lhs=False)
            #m.simd_atomizer(atomize_reference=True,atomize_lhs=True)
            #m.invariant_code_motion(CONSTANT_PATH_EFFECTS=False)
            #m.icm(CONSTANT_PATH_EFFECTS=False)
            #m.display()
            
#            w.activate(module.transformers_intra_full)
#            w.activate(module.intraprocedural_summary_precondition)
#            w.activate(module.preconditions_intra)
            print "outlining to launcher"
            seed,nb="launcher_",0
            launchers=[]
            for k in kernels:
                name=seed+str(nb)
                nb+=1
                T.privatize_module()
                T.outline(module_name=name,label=k.label,smart_reference_computation=True,loop_bound_as_parameter=k.loops(0).label)
                launchers+=[w[name]]
            if debug:T.display()
            if debug:
                for l in launchers:l.display(activate='PRINT_CODE_REGIONS')
            
            print "outlining to microcode"
            microcodes=[]
            for l in launchers:
                if debug:l.display()
                l.terapix_warmup()
                if debug:l.display()
                theloop=l.loops(0)
                #l.loop_normalize(one_increment=True,lower_bound=0)
                #l.redundant_load_store_elimination()
                name=l.name+_microcode_suffix
                loop_to_outline=theloop.loops(0)
                l.outline(module_name=name,label=loop_to_outline.label,smart_reference_computation=True)
                mc=w[name]
                if debug:l.display()
                if debug:mc.display()
                microcodes+=[mc]
            w.check(debug)
            print "refining microcode"
            for m in microcodes:
                if debug:m.display()
                m.loop_normalize(one_increment=True,lower_bound=0)
                m.redundant_load_store_elimination()
                m.flatten_code(flatten_code_unroll=False)
                m.linearize_array(use_pointers=True)
                if debug:m.display()
                w.check(debug)
                m.invariant_code_motion()
                m.icm()
                if debug:m.display()
                m.strength_reduction()
                if debug:m.display()
                w.check(debug)
                m.strength_reduction()
                if debug:m.display()
                w.check(debug)
                #def rloops(l):
                #    lloops=l.loops()
                #    for p in l.loops():
                #        lloops+=rloops(p)
                #    return lloops
                #ll=rloops(m.loops(0))
                #ll.reverse()
                #for l in ll:
                #    l.full_unroll()
                #if debug:m.display()
                #w.check(debug)
                #m.common_subexpression_elimination(skip_added_constant=True)
                #m.common_subexpression_elimination(skip_added_constant=True)
                w.check(debug)
                if debug:m.display()
                #if debug:m.display()
                #m.forward_substitute()
                #if debug:m.display(module.print_code_proper_effects)
                #if debug:m.display(module.print_code_cumulated_effects)
                #if debug:m.display(module.print_code_regions)
                #m.redundant_load_store_elimination()
                m.split_update_operator()
                #m.expression_substitution("refi")
                #if debug:m.display()
                if debug:m.display()
                if m.code.find("MIN")>0 or m.code.find("MAX") >0:
                    m.simd_atomizer(atomize_reference=True,atomize_lhs=True)
                    if debug:m.display()
                #m.generate_two_addresses_code()
                #m.common_subexpression_elimination()
                w.check(debug)
                if debug:m.display()
                m.flatten_code(unroll=False)
                m.clean_declarations()
                m.redundant_load_store_elimination()
                w.check(debug)
                m.normalize_microcode()
                if debug:
                    m.display()
                w.check(debug)
                if debug:
                    m.display()
                    m.callers.display()
            w.check(debug)
        #    for m in microcodes:
        #        for asm in w.fun:
        #            if asm.cu == runtime[1]:
        #                m.expression_substitution(asm.name)
        #


module.terapix_code_generation=terapix_code_generation

