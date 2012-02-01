from validation import vworkspace
import broker,re, os
from p4a_cold_stubs_broker import p4a_cold_stubs_broker


class vbrokerworkspace(vworkspace,broker.workspace):
  ''' This is a composition workspace, it inherit from both the validation workspace
  and the broker workspace. '''
  def __init__(self,*args,**kwargs):
    super(vbrokerworkspace,self).__init__(*args,**kwargs)


class coldValidationWorkspace(vbrokerworkspace):
  ''' This workspace helps for COLD input. It will mostly reduce include size
  because PIPS just have a very slow parser :-( '''
  

  # mapping for non trivial fonction_to_header
  functionsTable = [
                    [ 'tictoc', 'tic', 'toc' ],
                    [ 'io_m', 'mopen', 'mclose', 'meof', 'mfprintf', 'mfscanf' ],
                    [ 'memory_management', 'write_to_scilab', 'read_from_scilab',
                      'read_int_from_scilab', 'read_real_from_scilab',
                      'read_complex_from_scilab', 'read_string_from_scilab',
                      'read_intM_from_scilab', 'read_realM_from_scilab',
                      'read_complexM_from_scilab', 'read_stringM_from_scilab']
                   ]

  # initialized from functionsTable with full header name
  patchedFunctionsTable = []

  # Main include in COLD output
  scilab_rt_h = 'scilab_rt.h'
  
  @classmethod
  def transformFunctionName( _class, name):
    ''' Get include for a function, mostly it's a one-to-one mapping except for
    functions registered in functionsTable'''
    
    ''' First run, initialize patchedFunctionsTable from functionsTable'''
    if not _class.patchedFunctionsTable:
        _class.patchedFunctionsTable = map( lambda x: map( lambda y: 'scilab_rt_' + y, x), _class.functionsTable)

    ''' Look for requested function in patchedFunctionsTable'''
    for l in _class.patchedFunctionsTable:
      if (name in l):
        return l[0]

    return name


  @classmethod
  def buildIncludeListFromFile(_class,cold_c_file):
     ''' Get the runtime function called from the C file and return a list of headers to include'''   
     declarationList = ['scilab_rt_init', 'scilab_rt_terminate', 'scilab_rt_rand_parallel']  # pre include some generic declarations
     fd = open(cold_c_file)
     expr = re.compile('scilab_rt_[^_]*_[^\(]*[\(]') # gather rougly pattern "scilab_rt_blablah" functions

     for line in fd:
       declarationList += expr.findall(line)
     for i in range(len(declarationList)):
       declarationList[i] = re.sub('_[^_]*_[^_]*\($', '', declarationList[i]) # remove parameters prototype


     fd.close()
     return declarationList


  @classmethod
  def getNewInclude(_class, cold_c_file):
    basename = os.path.splitext(os.path.basename(cold_c_file))[0] 
    new_include = basename+'_'+_class.scilab_rt_h
    return new_include  

  @classmethod
  def patchWithNewHeader(_class,cold_c_file):
    new_include = _class.getNewInclude(cold_c_file)
  
    # get the list of headers to include
    declarationList = _class.buildIncludeListFromFile(cold_c_file)
    declarationList = map( lambda x: _class.transformFunctionName(x), declarationList)
    declarationList = list(set(declarationList))  # make this digest

    # Open the full include file and produce a new include
    source_scilab_rt_h = open(os.path.join("stubs/include", _class.scilab_rt_h), 'r')
    target_scilab_rt_h = open(os.path.join(new_include), 'w')

    # filter out originals #include from scilab_rt.h
    for l in source_scilab_rt_h:
      if re.search('#include *"scilab_rt_', l) == None:
         target_scilab_rt_h.write(l)

    # append correponding #include to new include file
    for f in declarationList:
      target_scilab_rt_h.write('#include "' + f + '.h"\n')
      
    source_scilab_rt_h.close()
    target_scilab_rt_h.close()

    # Patch the C source file
    fd = open(cold_c_file)
    content = fd.read()
    fd.close()
    content = re.sub('#include *"'+_class.scilab_rt_h+'"',
                     '#include "'+new_include+'"', content)
                     
    # Hum we patch the original file, which is not what we should do !               
    fd = open(cold_c_file, 'w')
    fd.write(content)
    fd.close()
    

  def __init__(self,*args,**kwargs):
    # path input file so that headers are not too big
    self.cold_c_file = coldValidationWorkspace.getMainSourceFile()
    coldValidationWorkspace.patchWithNewHeader(self.cold_c_file)
    # Init the parent with a dedicated stubs broker
    super(coldValidationWorkspace,self).__init__(brokersList="p4a_cold_stubs_broker",*args,**kwargs)


  def __del__(self):
    # we patched the original source file, we should restore it
    if hasattr(self,"cold_c_file"):
      new_include = coldValidationWorkspace.getNewInclude(self.cold_c_file)
      fd = open(self.cold_c_file)
      content = fd.read()
      fd.close()
      content = re.sub('#include "'+new_include+'"',
                       '#include "'+coldValidationWorkspace.scilab_rt_h+'"',
                       content)
                     
      # Hum we patch the original file, which is not what we should do !               
      fd = open(self.cold_c_file, 'w')
      fd.write(content)
      fd.close()
      
      os.remove(new_include)
    
    
  def __exit__(self,exc_type, exc_val, exc_tb):
    self.__del__()    
    
    

with coldValidationWorkspace(cppflags="-I stubs/include") as w:
  w.props.memory_effects_only=False
  w.props.constant_path_effects = False
  w.props.aliasing_across_types = False
  w.props.semantics_compute_transformers_in_context = False
  w.props.semantics_normalization_level_before_storage = 2
  w.props.semantics_fix_point_operator="derivative"
  w.props.semantics_keep_do_loop_exit_condition  = False
  w.props.trust_constant_path_effects_in_conflicts = True
  w.props.prettyprint_sequential_style = "do"
  
  w.activate("TRANSFORMERS_INTRA_FAST")
  w.activate("PRECONDITIONS_INTRA_FAST")
  w.activate("RICE_FAST_DEPENDENCE_GRAPH")
  w.activate("NEW_CONTROLIZER")

  filter_exclude_re = re.compile("^scilab_rt_.*")
  all_modules = w.filter(lambda module: not module.compilation_unit_p() and not filter_exclude_re.match(module.name))

  all_modules.simplify_control_directly(concurrent=True)
  all_modules.privatize_module()
  all_modules.internalize_parallel_code(concurrent=True)
  all_modules.coarse_grain_parallelization(concurrent=True)
  all_modules.simplify_control_directly(concurrent=True)
  all_modules.flatten_code(concurrent=True, unroll=False)
  all_modules.loop_fusion(concurrent=True) # may add param : greedy=True
  all_modules.quick_scalarization()
  all_modules.scalarization()
  all_modules.clean_declarations()
  all_modules.privatize_module()
  all_modules.localize_declaration()
  all_modules.ompify_code(concurrent=True)
  all_modules.omp_merge_pragma(concurrent=True)
  all_modules.display()
  
