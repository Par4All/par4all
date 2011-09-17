#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif

#include "defines-local.h" // for STEP_DEFAULT_RT_H
#include "preprocessor.h" // for pips_srcpath_append
#include "transformer.h" // for add_intermediate_value
#include "semantics.h" // for load_statement_precondition
#include "effects-generic.h" // needed by effects-convex.h for descriptor
#include "effects-simple.h" // for effect_to_string
#include "effects-convex.h" // for region

#include "dg.h" // for dg_arc_label, dg_vertex_label
typedef dg_arc_label arc_label;
typedef dg_vertex_label vertex_label;
#include "graph.h" // for graph
#include "ricedg.h" // for vertex_to_statement
#include "bootstrap.h"



GENERIC_LOCAL_FUNCTION(step_path, map_step_point)
GENERIC_LOCAL_FUNCTION(step_interlaced, map_effect_bool)
GENERIC_LOCAL_FUNCTION(step_partial, map_effect_bool)


/*
  Calcul des statements paths

  Pour tous statements S d'un module M,
  on appelle SP(S) (statement_path) le chemin permettant de rejoindre le statement S à partir du statement "body" de M.
*/

static hash_table sp_table = hash_table_undefined;

static bool sp_build(statement stmt, list *sp_current)
{
  *sp_current = CONS(STATEMENT, stmt, *sp_current);

  list sp = gen_nreverse(gen_copy_seq(*sp_current));
  hash_put(sp_table, stmt, sp);

  return true;
}
static void sp_unbuild(statement stmt, list *sp_current)
{
  gen_remove_once(sp_current, stmt);
}

static void sp_init(statement body)
{
  assert(hash_table_undefined_p(sp_table));
  sp_table = hash_table_make(hash_pointer, HASH_DEFAULT_SIZE);

  list sp_current = NIL;
  gen_context_recurse(body, &sp_current, statement_domain, sp_build, sp_unbuild);
}

static void sp_finalize(void)
{
  assert(!hash_table_undefined_p(sp_table));

  HASH_MAP(stmt, sp,
	   {
	     gen_free_list(sp);
	   }, sp_table)

  hash_table_free(sp_table);
  sp_table = hash_table_undefined;
}

static list sp_get(statement stmt)
{
  return hash_get(sp_table, stmt);
}

static statement sp_first_directive_statement(statement stmt)
{
  list sp = sp_get(stmt);
  while(!ENDP(sp))
    {
      statement current = STATEMENT(CAR(sp));
      if(step_directives_bound_p(current))
	return current;
      POP(sp);
    }

  return statement_undefined;
}

/*
static void sp_print(list sp)
{
  int lvl = 0;
  FOREACH(STATEMENT, stmt, sp)
    {
      pips_debug(2, "statement_path stmt_lvl %d\n", lvl++);
      STEP_DEBUG_STATEMENT(2, "", stmt);
    }
}

static void sp_print_all(void)
{
  HASH_MAP(stmt, sp,
	   {
	     STEP_DEBUG_STATEMENT(2, "Statement key", stmt);
	     pips_debug(2, "statement_path length=%d\n", (int)gen_length((list)sp));
	     sp_print((list)sp);
	   }, sp_table)
}

static list sp_factorise(statement s1, statement s2, list *sp1, list* sp2)
{
  list sp_common = NIL;

  *sp1 = sp_get(s1);
  *sp2 = sp_get(s2);

  while(!ENDP(*sp1) && !ENDP(*sp2) && STATEMENT(CAR(*sp1))==STATEMENT(CAR(*sp2)))
    {
      sp_common = CONS(STATEMENT, STATEMENT(CAR(*sp1)), sp_common);
      POP(*sp1);
      POP(*sp2);
    }

  return sp_common;
}
*/

bool step_interlaced_p(region reg)
{
  assert(bound_step_interlaced_p(reg));
  return load_step_interlaced(reg);
}

bool step_partial_p(region reg)
{
  assert(bound_step_partial_p(reg));
  return load_step_partial(reg);
}

bool step_analysed_module_p(const char* module_name)
{
  return db_resource_required_or_available_p(DBR_STEP_SEND_REGIONS,  module_name);
}

void debug_print_effects_list(list l, string txt)
{
  if(ENDP(l))
    pips_debug(1, "%s empty\n", txt);
  else
    {
      bool property_prettyprint_scalar_regions = get_bool_property("PRETTYPRINT_SCALAR_REGIONS");
      set_bool_property("PRETTYPRINT_SCALAR_REGIONS", true);

      FOREACH(EFFECT, eff, l)
	{
	  string str_reg = text_to_string(text_rw_array_regions(CONS(EFFECT, eff, NIL)));
	  pips_debug(1, "%s %p : %s\n", txt, eff, str_reg); free(str_reg);
	}
      set_bool_property("PRETTYPRINT_SCALAR_REGIONS", property_prettyprint_scalar_regions);
    }
}


bool step_analyse_init(const char* __attribute__ ((unused)) module_name)
{

    DB_PUT_MEMORY_RESOURCE(DBR_STEP_COMM, "", make_step_comm(make_map_step_point(), make_map_effect_bool(), make_map_effect_bool()));

#ifdef PIPS_RUNTIME_DIR
    string srcpath=strdup(PIPS_RUNTIME_DIR "/" STEP_DEFAULT_RT_H);
#else
    string srcpath=strdup(concatenate(getenv("PIPS_ROOT"),"/",STEP_DEFAULT_RT_H,NULL));
#endif
    string old_path=pips_srcpath_append(srcpath);
    free(old_path);
    free(srcpath);

    /* init intrinsics */
    static struct intrin { 
        char * name;
        intrinsic_desc_t desc;
    } step_intrinsics [] = {
#include "STEP_RT_intrinsic.h"
        { NULL , {NULL, 0} }
    };
    for(struct intrin *p = step_intrinsics;p->name;++p)
        register_intrinsic_handler(p->name,&(p->desc));

    /* other intrinsics */
    static IntrinsicDescriptor IntrinsicTypeDescriptorTable[] =
    {
#include "STEP_RT_bootstrap.h"
        {NULL, 0, 0, 0, 0}
    };
    for(IntrinsicDescriptor *p=IntrinsicTypeDescriptorTable;p->name;p++)
        register_intrinsic_type_descriptor(p) ;

    return true;
}

void load_step_comm()
{
  step_comm comms = (step_comm)db_get_memory_resource(DBR_STEP_COMM, "", true);

  set_step_path(step_comm_path(comms));
  set_step_interlaced(step_comm_interlaced(comms));
  set_step_partial(step_comm_partial(comms));
}

void reset_step_comm()
{
  reset_step_path();
  reset_step_interlaced();
  reset_step_partial();
}

void store_step_comm()
{
  step_comm comms = (step_comm)db_get_memory_resource(DBR_STEP_COMM, "", true);

  step_comm_path(comms) = get_step_path();
  step_comm_interlaced(comms) = get_step_interlaced();
  step_comm_partial(comms) = get_step_partial();

  DB_PUT_MEMORY_RESOURCE(DBR_STEP_COMM, "", comms);

  reset_step_path();
  reset_step_interlaced();
  reset_step_partial();
}

static void add_path(effect new, entity module, statement stmt, effect previous)
{
  pips_debug(4, "ADD_STEP_PATH\n");
  store_step_path(new, make_step_point(module, stmt, previous));
}

static list get_path(effect start)
{
  list path = NIL;
  effect current = start ;

  while(bound_step_path_p(current))
    {
      step_point point = load_step_path(current);

      path = CONS(STEP_POINT, point, path);

      if (step_point_data(point) == current)
	break;

      current = step_point_data(point);
    }

  ifdebug(3)
    {
      int lvl=1;
      FOREACH(STEP_POINT, point, path)
	{
	  string txt = safe_statement_identification(step_point_stmt(point));
	  pips_debug(1, "lvl=%d module=%s statment= %s", lvl, entity_name(step_point_module(point)), txt);
	  free(txt);
	  debug_print_effects_list(CONS(EFFECT,step_point_data(point),NIL), "region");
	  lvl++;
	}
    }

  return path;
}

static void set_comm(effect data, bool full)
{
  list path = get_path(data);
  assert(!ENDP(path));
  effect first_send = step_point_data(STEP_POINT(CAR(path)));

  if(full)
    store_or_update_step_partial(first_send, false);
  else if(!bound_step_partial_p(first_send))
    store_step_partial(first_send, true);
}

static region rectangularization_region(region reg)
{
  reference r = region_any_reference(reg);
  list ephis = expressions_to_entities(reference_indices(r));
  Pbase phis = list_to_base(ephis);
  gen_free_list(ephis);
  region_system(reg) = sc_rectangular_hull(region_system(reg), phis);
  base_rm(phis);
  return reg;
}

static list compute_send_regions(list write_l, list out_l)
{
  list send_final = NIL;
  list send_l = RegionsIntersection(regions_dup(out_l), regions_dup(write_l), w_w_combinable_p);

  FOREACH(REGION, reg, send_l)
    {
      if (io_effect_p(reg))
	pips_user_warning("STEP : possible IO concurrence\n");
      else
	{
	  region r = region_dup(rectangularization_region(reg));

	  free_action(region_action(r));
	  region_action(r) = make_action_write_memory();
	  send_final = CONS(REGION, r, send_final);
	}
    }
  gen_full_free_list(send_l);

  return send_final;
}

static list compute_recv_regions(list send_l, list in_l)
{
  list recv_final = NIL;
  list send_may_l = NIL;

  FOREACH(REGION, reg, send_l)
    {
      if (region_may_p(reg))
	send_may_l=CONS(REGION, region_dup(reg), send_may_l);
    }
  list recv_l = RegionsMustUnion(regions_dup(in_l), regions_dup(send_may_l), r_w_combinable_p);
  gen_full_free_list(send_may_l);

  FOREACH(REGION, reg, recv_l)
    {
      if (io_effect_p(reg) || std_file_effect_p(reg))
	pips_debug(2,"drop effect on %s\n", entity_name(region_entity(reg)));
      else
	{
	  region r = region_dup(rectangularization_region(reg));

	  free_action(region_action(r));
	  region_action(r) = make_action_read_memory();
	  recv_final = CONS(REGION, r, recv_final);
	}
    }
  gen_full_free_list(recv_l);

  return recv_final;
}

static bool interlaced_basic_workchunk_regions_p(region reg, list index_l)
{
  if(ENDP(index_l))
    return false;

  bool interlaced_p;
  Psysteme s = sc_copy(region_system(reg));
  Psysteme s_prime = sc_copy(s);

  FOREACH(ENTITY, index, index_l)
    {
      add_intermediate_value(index);
      entity index_prime = entity_to_intermediate_value(index);
      s_prime = sc_variable_rename(s_prime, (Variable)index, (Variable)index_prime);

      // contrainte I<I' qui s'ecrit : I-I'+1 <= 0
      Pcontrainte c = contrainte_make(vect_make(VECTEUR_NUL,
				    (Variable) index, VALUE_ONE,
				    (Variable) index_prime, VALUE_MONE,
				    TCST, VALUE_ONE));
      sc_add_inegalite(s, c);
    }

  s = sc_append(s, s_prime);
  sc_rm(s_prime);

  s->base = BASE_NULLE;
  sc_creer_base(s);

  interlaced_p = sc_integer_feasibility_ofl_ctrl(s, NO_OFL_CTRL, true);

  sc_rm(s);
  return interlaced_p;
}

GENERIC_LOCAL_FUNCTION(step_send_regions, statement_effects)
GENERIC_LOCAL_FUNCTION(step_recv_regions, statement_effects)

list load_send_regions_list(statement s)
{
  if (!bound_step_send_regions_p(s))
    return NIL;

  effects e = load_step_send_regions(s);
  ifdebug(8) pips_assert("send regions loaded are consistent", effects_consistent_p(e));
  return(effects_effects(e));
}

void store_send_regions_list(statement s, list l_regions)
{
  effects e = make_effects(l_regions);
  ifdebug(8) pips_assert("send regions to store are consistent", effects_consistent_p(e));
  store_step_send_regions(s, e);
}

void update_send_regions_list(statement s, list l_regions)
{
  if (bound_step_send_regions_p(s))
    {
      effects e = load_step_send_regions(s);
      effects_effects(e) = l_regions;
      update_step_send_regions(s, e);
    }
}

void add_send_regions_list(statement s, list l_regions)
{
  if (bound_step_send_regions_p(s))
    {
      effects e = load_step_send_regions(s);
      effects_effects(e) = gen_nconc(l_regions, effects_effects(e));
      update_step_send_regions(s, e);
    }
  else
    store_send_regions_list(s, l_regions);
}

list load_recv_regions_list(statement s)
{
  if (!bound_step_recv_regions_p(s))
    return NIL;

  effects e = load_step_recv_regions(s);
  ifdebug(8) pips_assert("recv regions loaded are consistent", effects_consistent_p(e));
  return(effects_effects(e));
}

void store_recv_regions_list(statement s, list l_regions)
{
  effects e = make_effects(l_regions);
  ifdebug(8) pips_assert("recv regions to store are consistent", effects_consistent_p(e));
  store_step_recv_regions(s, e);
}

void update_recv_regions_list(statement s, list l_regions)
{
  if (bound_step_recv_regions_p(s))
    {
      effects e = load_step_recv_regions(s);
      effects_effects(e) = l_regions;
      update_step_recv_regions(s, e);
    }
}

void add_recv_regions_list(statement s, list l_regions)
{
  if (bound_step_recv_regions_p(s))
    {
      effects e = load_step_recv_regions(s);
      effects_effects(e) = gen_nconc(l_regions, effects_effects(e));
      update_step_recv_regions(s, e);
    }
  else
    store_recv_regions_list(s, l_regions);
}

static void step_print_directives_regions(step_directive d, list send_l, list recv_l)
{
  ifdebug(3)
    {
      statement stmt_basic_workchunk = step_directive_basic_workchunk(d);
      assert(!statement_undefined_p(stmt_basic_workchunk));

      list rw_l = load_rw_effects_list(stmt_basic_workchunk);
      list in_l = load_in_effects_list(stmt_basic_workchunk);
      list out_l = load_out_effects_list(stmt_basic_workchunk);

      string str_reg;
      bool property_prettyprint_scalar_regions = get_bool_property("PRETTYPRINT_SCALAR_REGIONS");
      set_bool_property("PRETTYPRINT_SCALAR_REGIONS", true);

      str_reg = text_to_string(text_rw_array_regions(rw_l));
      pips_debug(1, "REGIONS RW : %s\n", str_reg); free(str_reg);
      str_reg = text_to_string(text_rw_array_regions(in_l));
      pips_debug(1, "REGIONS IN : %s\n", str_reg); free(str_reg);
      str_reg = text_to_string(text_rw_array_regions(out_l));
      pips_debug(1, "REGIONS OUT : %s\n", str_reg); free(str_reg);

      set_bool_property("PRETTYPRINT_SCALAR_REGIONS", property_prettyprint_scalar_regions);
    }

  debug_print_effects_list(recv_l, "REGION RECV");

  if(ENDP(send_l))
    debug_print_effects_list(send_l, "REGION SEND");
  else
    {
      FOREACH(REGION, r, send_l)
	{
	  pips_assert("interlaced defined", bound_step_interlaced_p(r));
	  list l = CONS(EFFECT, r, NIL);
	  debug_print_effects_list(l, load_step_interlaced(r)?"REGION SEND INTERLACED":"REGION SEND");
	  gen_free_list(l);
	}
    }
}

static list summarize_and_map(list effect_l, entity module, statement body)
{
  list summarized_l = NIL;

  FOREACH(EFFECT, eff, effect_l)
    {
      effect new = copy_effect(eff);

      pips_debug(2, "tmp summarize eff=%p new=%p\n", eff, new);
      add_path(new, module, body, eff);
      summarized_l = CONS(EFFECT, new, summarized_l);
    }

  return summarized_l;
}

static void summarize_and_map_step_regions(statement stmt)
{
  /*
    Si on ne traite pas un statement imbrique dans une directive, on reporte
    les regions SEND et RECV pour les analyses inter-procedurales
    au niveau du statement body (representant le statement du call)
  */
  statement first_directive_stmt = sp_first_directive_statement(stmt);

  if(statement_undefined_p(first_directive_stmt) || first_directive_stmt == stmt)
    {
      entity module = get_current_module_entity();
      statement body = get_current_module_statement();

      assert(stmt != body);
      pips_assert("statement with step regions", bound_step_send_regions_p(stmt) && bound_step_recv_regions_p(stmt));

      add_send_regions_list(body, summarize_and_map(load_send_regions_list(stmt), module, body));
      add_recv_regions_list(body, summarize_and_map(load_recv_regions_list(stmt), module, body));

      pips_debug(2, " SEND and RECV propaged on module %s\n", entity_name(module));
    }
}

static list filtre_and_map_regions(list regions_l, entity module, statement stmt)
{
  list filtred_l = NIL;
  FOREACH(REGION, reg, regions_l)
    {
      if(region_scalar_p(reg))
	{
	  pips_debug(2,"drop scalar %s\n", entity_name(region_entity(reg)));
	  region_free(reg);
	}
      else
	{
	  filtred_l = CONS(REGION, reg, filtred_l);
	  add_path(reg, module, stmt, reg);
	}
    }
  gen_free_list(regions_l);
  return filtred_l;
}

static void compute_directive_regions(statement stmt)
{
  entity module = get_current_module_entity();
  step_directive d = step_directives_get(stmt);
  assert(stmt==step_directive_block(d));

  statement directive_stmt = step_directive_block(d);
  statement stmt_basic_workchunk = step_directive_basic_workchunk(d);
  assert(!statement_undefined_p(stmt_basic_workchunk));

  ifdebug(1)
    {
      pips_debug(1,"\n");
      step_directive_print(d);
    }

  list rw_l = load_rw_effects_list(stmt_basic_workchunk);
  list write_l = regions_write_regions(rw_l);
  list in_l = load_in_effects_list(stmt_basic_workchunk);
  list out_l = load_out_effects_list(stmt_basic_workchunk);

  /*
    Calcul des regions SEND et RECV
  */
  list send_l = filtre_and_map_regions(compute_send_regions(write_l, out_l), module, stmt);
  list recv_l = filtre_and_map_regions(compute_recv_regions(send_l, in_l), module, stmt);

  store_send_regions_list(directive_stmt, send_l);
  store_recv_regions_list(directive_stmt, recv_l);


  list index_l = step_directive_basic_workchunk_index(d);
  FOREACH(REGION, reg, send_l)
    {
      /* Identification des regions SEND interlaced */
      store_step_interlaced(reg, interlaced_basic_workchunk_regions_p(reg, index_l));

      /* initialisation des communications SEND PARTIAL */
      set_comm(reg, false); // false : PARTIAL
    }

  ifdebug(1)
    {
      step_print_directives_regions(d, send_l, recv_l);
      pips_debug(1,"\n");
    }

  summarize_and_map_step_regions(stmt);
}

static list translate_and_map(statement stmt, list effect_l, transformer context)
{
  effect new;
  list translated_l = NIL;
  entity module = get_current_module_entity();
  entity called = call_function(statement_call(stmt));
  list args = call_arguments(statement_call(stmt));

  FOREACH(EFFECT, eff, effect_l)
    {
      /* translate */
      list called_l = CONS(EFFECT, eff, NIL);
      list caller_l = generic_effects_backward_translation(called, args, called_l, context);
      gen_free_list(called_l);

      /* map */
      switch (gen_length(caller_l))
	{
	case 1:
	  new = EFFECT(CAR(caller_l));
	  pips_debug(2, "translate eff=%p new=%p\n", eff, new);
	  add_path(new, module, stmt, eff);
	  translated_l = gen_nconc(translated_l, caller_l);
	case 0:
	  break;
	default:
	  assert(0);
	}
    }

  return translated_l;
}

static void translate_and_map_step_regions(statement stmt)
{
  entity called;
  list caller_l, called_l;
  statement_effects regions;

  assert(statement_call_p(stmt));
  called = call_function(statement_call(stmt));

  assert(entity_module_p(called));
  statement called_body = (statement) db_get_memory_resource(DBR_CODE, entity_user_name(called), true);
  transformer context = load_statement_precondition(stmt);

  ifdebug(1)
    {
      pips_debug(1,"------------------> CALL\n");
      string txt = safe_statement_identification(stmt);
      pips_debug(1, "%s\n", txt);
      free(txt);
    }

  /* SEND */
  regions = (statement_effects)db_get_memory_resource(DBR_STEP_SEND_REGIONS, entity_user_name(called), true);
  called_l = effects_effects(apply_statement_effects(regions, called_body));
  caller_l = translate_and_map(stmt, called_l, context);
  store_send_regions_list(stmt, caller_l);

  ifdebug(2)
    {
      debug_print_effects_list(called_l, "CALLED REGIONS SEND");
      debug_print_effects_list(caller_l, "CALLER REGIONS SEND");
    }

  /* RECV */
  regions = (statement_effects)db_get_memory_resource(DBR_STEP_RECV_REGIONS, entity_user_name(called), true);
  called_l = effects_effects(apply_statement_effects(regions, called_body));
  caller_l = translate_and_map(stmt, called_l, context);
  store_recv_regions_list(stmt, caller_l);

  ifdebug(2)
    {
      debug_print_effects_list(called_l, "CALLED REGIONS RECV");
      debug_print_effects_list(caller_l, "CALLER REGIONS RECV");
    }

  summarize_and_map_step_regions(stmt);
}

static bool compute_step_regions(statement stmt)
{
  assert(!statement_undefined_p(stmt));

  if(step_directives_bound_p(stmt))
    compute_directive_regions(stmt);
  else if (statement_call_p(stmt) && entity_module_p(call_function(statement_call(stmt))))
    translate_and_map_step_regions(stmt);

  return true;
}

static bool concerned_entity_p(effect conflict, list regions)
{
  FOREACH(REGION, reg, regions)
    {
      pips_debug(4, "\n conflict %s regions %s",text_to_string(text_region(conflict)), text_to_string(text_region(reg)));
      if (effect_comparable_p(conflict, reg))
	{
	  pips_debug(4, "\t\tCHECK true\n");
	  return true;
	}
    }
  pips_debug(4, "\t\tCHECK false\n");
  return false;
}

bool step_analyse(const char* module_name)
{
  debug_on("STEP_ANALYSE_DEBUG_LEVEL");
  pips_debug(1, "%d module_name = %s\n", __LINE__, module_name);

  entity module = local_name_to_top_level_entity(module_name);
  set_current_module_entity(module);

  set_rw_effects((statement_effects)db_get_memory_resource(DBR_REGIONS, module_name, true));
  set_out_effects((statement_effects) db_get_memory_resource(DBR_OUT_REGIONS, module_name, true));
  set_in_effects((statement_effects) db_get_memory_resource(DBR_IN_REGIONS, module_name, true));

  statement body = (statement) db_get_memory_resource(DBR_CODE, module_name, true);
  set_current_module_statement(body);
  set_ordering_to_statement(body);

  init_convex_rw_regions(module_name);
  set_methods_for_convex_effects();

  step_directives_init();

  load_step_comm();

  sp_init(body);

  /*
    Compute SEND and RECV regions
  */
  pips_debug(1, "################ REGIONS  %s ###############\n", entity_name(module));
  init_step_send_regions();
  init_step_recv_regions();
  /* Initialisation summery region*/
  add_send_regions_list(body, NIL);
  add_recv_regions_list(body, NIL);

  gen_recurse(body, statement_domain, compute_step_regions, gen_null);


  /*
    DG analyse
  */
  pips_debug(2, "################ DG %s ###############\n", entity_name(module));
  set not_send = set_make(set_pointer);
  set not_recv = set_make(set_pointer);
  graph dependences = (graph) db_get_memory_resource(DBR_CHAINS, module_name, true);
  FOREACH(VERTEX, v1, graph_vertices(dependences))
    {
      list l_send = NIL;
      statement s1 = vertex_to_statement(v1);

      if (bound_step_send_regions_p(s1))
	l_send = load_send_regions_list(s1);

      ifdebug(2)
	{
	  STEP_DEBUG_STATEMENT(2, "statement s1", s1);
	  pips_debug(2, "from %s", safe_statement_identification(s1));
	  debug_print_effects_list(l_send, "SEND S1");
	}

      FOREACH(SUCCESSOR, su, vertex_successors(v1))
	{
	  list l_recv = NIL;
	  statement s2 = vertex_to_statement(successor_vertex(su));

	  if (bound_step_recv_regions_p(s2))
	    l_recv = load_recv_regions_list(s2);

	  ifdebug(2)
	    {
	      STEP_DEBUG_STATEMENT(2, "statement s2", s2);
	      pips_debug(2, "to %s", safe_statement_identification(s2));
	      debug_print_effects_list(l_recv, "RECV S2");
	    }

	  FOREACH(CONFLICT, c, dg_arc_label_conflicts((dg_arc_label)successor_arc_label(su)))
	    {
	      effect source = conflict_source(c);
	      effect sink = conflict_sink(c);
	      if(!io_effect_p(source) && !io_effect_p(sink))
		{
		  pips_debug(2, "Dependence :\n\t\t%s from\t%s\t\t%s to  \t%s",
			     effect_to_string(source), text_to_string(text_region(source)),
			     effect_to_string(sink), text_to_string(text_region(sink)));

		  if(!(effect_write_p(source) && effect_read_p(sink)))
		    pips_debug(2,"not WRITE->READ dependence (ignored)\n");
		  else
		    {
		      pips_debug(2,"new WRITE->READ dependence\n");
		      if(!(concerned_entity_p(source, l_send) && !concerned_entity_p(sink, l_recv)))
			pips_debug(2,"############################# not unoptimizable %s\n", entity_name(effect_entity(source)));
		      else
			{
			  /* l'entité n'est plus optimisable pour S1 (et donc non remontable)*/
			  FOREACH(EFFECT, send_e, l_send)
			    {
			      if(effect_entity(source) == effect_entity(send_e))
				{
				  pips_debug(2,"############################# unoptimizable %p %s\n", send_e, entity_name(effect_entity(send_e)));
				  not_send = set_add_element(not_send, not_send, send_e);
				}
			    }
			}

		      if(!(concerned_entity_p(sink, l_recv) && !concerned_entity_p(source, l_send)))
			pips_debug(2,"############################# not unremontable %s\n", entity_name(effect_entity(sink)));
		      else
			{
			  /* l'entité n'est plus remontable (produit par S1) */
			   FOREACH(EFFECT, recv_e, l_recv)
			    {
			      if(effect_entity(sink) == effect_entity(recv_e))
				{
				  pips_debug(2,"############################# unremontable %p %s\n", recv_e, entity_name(effect_entity(recv_e)));
				  not_recv = set_add_element(not_recv, not_recv, recv_e);
				}
			    }
			}
		    }
		}
	    }
	  pips_debug(2, "statement s2 end\n");
	}
      pips_debug(2, "statement s1 end\n");
    }

  /*
    SUMMARY SEND and RECV REGIONS
  */
  list final_summary_send = NIL;
  FOREACH(EFFECT, eff, load_send_regions_list(get_current_module_statement()))
    {
      assert(bound_step_path_p(eff));
      step_point point = load_step_path(eff);

      if(!set_belong_p(not_send, step_point_data(point)))
	final_summary_send = CONS(EFFECT, eff, final_summary_send);
      else
	pips_debug(2, "drop unoptimizable tmp_summary_send %p -> data %p\n", eff, step_point_data(point));
    }
  update_send_regions_list(body, final_summary_send);

  list final_summary_recv = NIL;
  FOREACH(EFFECT, eff, load_recv_regions_list(get_current_module_statement()))
    {
      if(!set_belong_p(not_recv, eff))
	final_summary_recv = CONS(EFFECT, eff, final_summary_recv);
    }
  update_recv_regions_list(body, final_summary_recv);

  ifdebug(2)
    {
      debug_print_effects_list(final_summary_send, "FINAL SUMMARISED SEND");
      debug_print_effects_list(final_summary_recv, "FINAL SUMMARISED RECV");
    }

  DB_PUT_MEMORY_RESOURCE(DBR_STEP_SEND_REGIONS, module_name, get_step_send_regions());
  DB_PUT_MEMORY_RESOURCE(DBR_STEP_RECV_REGIONS, module_name, get_step_recv_regions());

  /*
    Update comm PARTIAL or FULL
  */
  SET_FOREACH(effect, eff, not_send)
    {
      set_comm(eff, true); // true : FULL
    }
  FOREACH(EFFECT, eff, final_summary_send)
    {
      set_comm(eff, false); // false : PARTIAL
    }

  ifdebug(1)
    if(entity_main_module_p(module))
      {
	pips_debug(1, "################ COMM %s ###############\n", entity_name(module));
	MAP_EFFECT_BOOL_MAP(eff, partial,
			    {
			      step_point point = load_step_path(eff);
			      pips_debug(1, "module : %s\n", entity_user_name(step_point_module(point)));
			      ifdebug(2)
				print_statement(step_point_stmt(point));
			      debug_print_effects_list(CONS(EFFECT,step_point_data(point),NIL), partial?"COMMUNICATION PARTIAL":"COMMUNICATION FULL");
			    }, get_step_partial());
      }

  reset_step_send_regions();
  reset_step_recv_regions();

  sp_finalize();

  store_step_comm();

  step_directives_reset();

  generic_effects_reset_all_methods();
  reset_convex_rw_regions(module_name);

  reset_ordering_to_statement();
  reset_current_module_statement();

  reset_in_effects();
  reset_out_effects();
  reset_rw_effects();

  reset_current_module_entity();

  debug_off();
  return true;
}

