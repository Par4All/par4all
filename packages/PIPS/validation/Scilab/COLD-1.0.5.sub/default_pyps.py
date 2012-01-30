from validation import vworkspace
import broker,re
from p4a_cold_stubs_broker import p4a_cold_stubs_broker

class vbrokerworkspace(vworkspace,broker.workspace):
  def __init__(self,*args,**kwargs):
    super(vbrokerworkspace,self).__init__(brokersList="p4a_cold_stubs_broker",*args,**kwargs)


with vbrokerworkspace(cppflags="-I stubs/include") as w:
  w.props.memory_effects_only=False
  w.props.constant_path_effects = False
  w.props.aliasing_across_types = False
  w.props.semantics_compute_transformers_in_context = False
  w.props.semantics_normalization_level_before_storage = 2
  w.props.semantics_fix_point_operator="derivative"
  w.props.semantics_keep_do_loop_exit_condition  = False
  w.props.trust_constant_path_effects_in_conflicts = True
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
  all_modules.display()
