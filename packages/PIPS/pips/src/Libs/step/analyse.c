#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif

#include "defines-local.h"
#include "preprocessor.h" // for pips_srcpath_append
#include "transformer.h" // for add_intermediate_value
#include "semantics.h" // for load_statement_precondition

#include "dg.h" // for dg_arc_label, dg_vertex_label
typedef dg_arc_label arc_label;
typedef dg_vertex_label vertex_label;
#include "graph.h" // for graph
#include "ricedg.h" // for vertex_to_statement
#include "bootstrap.h"


/* The step_analyse phase computes three main resources:
   - step_comm
   - step_send_regions
   - step_recv_regions
*/

/*
  Corresponds to the three fields of the newgen step_comm type.
 */
GENERIC_LOCAL_FUNCTION(step_effect_path, map_effect_step_point)
GENERIC_LOCAL_FUNCTION(step_interlaced, map_effect_bool)
GENERIC_LOCAL_FUNCTION(step_partial, map_effect_bool)

/* Declaration of the step_send_regions and step_send_regions resources.

   Store all different kind of regions:
   - directive regions (at directive statement level)
   - translated regions (at call statement level)
   - summary regions (at body statement level)

*/

GENERIC_LOCAL_FUNCTION(step_send_regions, statement_effects)
GENERIC_LOCAL_FUNCTION(step_recv_regions, statement_effects)

/*
  Calcul des statements paths

  Pour tous les statements S d'un module M,

  on appelle SP(S) (statement_path) le chemin permettant de rejoindre
  le statement S à partir du statement "body" de M.

*/

static hash_table step_statement_path = hash_table_undefined;

static bool step_statement_path_build(statement stmt, list *sp_current)
{
  *sp_current = CONS(STATEMENT, stmt, *sp_current);

  list sp = gen_nreverse(gen_copy_seq(*sp_current));
  hash_put(step_statement_path, stmt, sp);

  return true;
}
static void step_statement_path_unbuild(statement stmt, list *sp_current)
{
  gen_remove_once(sp_current, stmt);
}

static void step_statement_path_init(statement body)
{
  assert(hash_table_undefined_p(step_statement_path));
  step_statement_path = hash_table_make(hash_pointer, HASH_DEFAULT_SIZE);

  list sp_current = NIL;
  gen_context_recurse(body, &sp_current, statement_domain, step_statement_path_build, step_statement_path_unbuild);
}

static void step_statement_path_finalize(void)
{
  assert(!hash_table_undefined_p(step_statement_path));

  HASH_MAP(stmt, sp,
	   {
	     gen_free_list(sp);
	   }, step_statement_path)

  hash_table_free(step_statement_path);
  step_statement_path = hash_table_undefined;
}

static list step_statement_path_get(statement stmt)
{
  return hash_get(step_statement_path, stmt);
}

static statement step_statement_path_first_directive_statement(statement stmt)
{
  list sp = step_statement_path_get(stmt);

  statement first_directive_stmt;

  first_directive_stmt = gen_find_if((gen_filter_func_t)step_directives_bound_p, sp, (void * (*)(const union gen_chunk))gen_identity);

  return first_directive_stmt;
}

/*
static void step_statement_path_print(list sp)
{
  int lvl = 0;
  FOREACH(STATEMENT, stmt, sp)
    {
      pips_debug(2, "statement_path stmt_lvl %d\n", lvl++);
      STEP_DEBUG_STATEMENT(2, "", stmt);
    }
}

static void step_statement_path_print_all(void)
{
  HASH_MAP(stmt, sp,
	   {
	     STEP_DEBUG_STATEMENT(2, "Statement key", stmt);
	     pips_debug(2, "statement_path length=%d\n", (int)gen_length((list)sp));
	     step_statement_path_print((list)sp);
	   }, step_statement_path)
}
*/


/*
  step_statement_path_factorise

  From two statement paths s1 (B, ...., S1) and s2 (B, ...., S3),
  step_statement_path_factorise() returns the common statement path
  (B, ..., C) and two suffix statement paths (C, ..., S1) and (C, ..., S3).

 */

static list step_statement_path_factorise(statement s1, statement s2, list *suffix_sp1, list* suffix_sp2)
{
  list common_sp = NIL;

  pips_debug(5, "begin\n");

  *suffix_sp1 = step_statement_path_get(s1);
  *suffix_sp2 = step_statement_path_get(s2);

  while(!ENDP(*suffix_sp1) && !ENDP(*suffix_sp2) && STATEMENT(CAR(*suffix_sp1))==STATEMENT(CAR(*suffix_sp2)))
    {
      common_sp = CONS(STATEMENT, STATEMENT(CAR(*suffix_sp1)), common_sp);
      POP(*suffix_sp1);
      POP(*suffix_sp2);
    }

  pips_debug(5, "end\n");
  return common_sp;
}


/*
 * step_comm resource management
 *
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

/*
 * step_comm management
 * also used in compile.c
 *
 */

void load_step_comm()
{
  step_comm comms = (step_comm)db_get_memory_resource(DBR_STEP_COMM, "", true);

  set_step_effect_path(step_comm_path(comms));
  set_step_interlaced(step_comm_interlaced(comms));
  set_step_partial(step_comm_partial(comms));
}

void reset_step_comm()
{
  reset_step_effect_path();
  reset_step_interlaced();
  reset_step_partial();
}

void store_step_comm()
{
  step_comm comms = (step_comm)db_get_memory_resource(DBR_STEP_COMM, "", true);

  step_comm_path(comms) = get_step_effect_path();
  step_comm_interlaced(comms) = get_step_interlaced();
  step_comm_partial(comms) = get_step_partial();

  DB_PUT_MEMORY_RESOURCE(DBR_STEP_COMM, "", comms);

  reset_step_effect_path();
  reset_step_interlaced();
  reset_step_partial();
}

/*
 * step_effect_path management
 *
 *
 */

static void step_add_point_into_effect_path(effect new_eff, entity module, statement stmt, effect previous_eff)
{
  pips_debug(4, "begin\n");
  step_point point;
  point = make_step_point(module, stmt, previous_eff);
  store_step_effect_path(new_eff, point);
  pips_debug(4, "end\n");
}

static void step_print_effect_path(list path)
{
  int level=1;
  FOREACH(STEP_POINT, point, path)
    {
      string txt = safe_statement_identification(step_point_stmt(point));
      pips_debug(1, "level=%d module=%s statment= %s", level, entity_name(step_point_module(point)), txt);
      free(txt);
      debug_print_effects_list(CONS(EFFECT, step_point_data(point),NIL), "step_point_data :");
      level++;
    }
}

/*
  Build a step_point list
  First step_point corresponds to start_eff

  In the middle, link from summary to translated regions...

  Last step_point data will correspond to the directive statement and
  the corresponding region.

  FSC A VOIR peut-être dans l'autre sens?

 */

static list step_get_effect_path(effect start_eff)
{
  list path = NIL;
  effect current_eff = start_eff ;

  pips_debug(4, "begin\n");
  while(bound_step_effect_path_p(current_eff))
    {
      step_point point = load_step_effect_path(current_eff);

      path = CONS(STEP_POINT, point, path);

      if (step_point_data(point) == current_eff)
	break;

      current_eff = step_point_data(point);
    }

  ifdebug(5)
    {
      step_print_effect_path(path);
    }

  pips_debug(4, "end\n");
  return path;
}

static effect get_directive_statement_region(effect reg)
{
  effect directive_statement_region;
  pips_debug(4, "begin\n");

  list path = step_get_effect_path(reg);
  assert(!ENDP(path));
  directive_statement_region = step_point_data(STEP_POINT(CAR(path)));

  pips_debug(4, "end\n");
  return directive_statement_region;
}

/*
  SEND regions corresponding to directives will be associated with partial or full

  Only directive regions are associated with communication
  type. TRANSLATED or SUMMARY regions are not.

*/
static void step_set_step_partial(effect send_region, bool partial_p)
{
  pips_debug(4, "begin\n");

  effect directive_statement_send;

  /* Retrieve the corresponding original DIRECTIVE
     region from any king of region: SUMMARY, TRANSLATED or DIRECTIVE region */
  directive_statement_send = get_directive_statement_region(send_region);

  if(!partial_p)
    store_or_update_step_partial(directive_statement_send, false);
  else if(!bound_step_partial_p(directive_statement_send))
    store_step_partial(directive_statement_send, true);
  pips_debug(4, "end\n");
}
static void step_set_communication_type_full(effect send_region)
{
  step_set_step_partial(send_region, false);
}
static void step_set_communication_type_partial(effect send_region)
{
  step_set_step_partial(send_region, true);
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


/*
 *
 * PHASE step_analyse_init
 *
 *
 *
 */

bool step_analyse_init(__attribute__ ((unused)) const char* module_name)
{

  pips_debug(2, "begin\n");
  DB_PUT_MEMORY_RESOURCE(DBR_STEP_COMM, "", make_step_comm(make_map_effect_step_point(), make_map_effect_bool(), make_map_effect_bool()));

  string srcpath = strdup(PIPS_RUNTIME_DIR "/" STEP_DEFAULT_RT_H);
  string old_path = pips_srcpath_append(srcpath);
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

  pips_debug(2, "end\n");
  return true;
}

region rectangularization_region(region reg)
{
  reference r = region_any_reference(reg);
  list ephis = expressions_to_entities(reference_indices(r));
  Pbase phis = list_to_base(ephis);
  gen_free_list(ephis);
  region_system(reg) = sc_rectangular_hull(region_system(reg), phis);
  base_rm(phis);
  return reg;
}

static list create_step_regions(statement stmt, list regions_l, enum action_utype action_tag)
{
  list regions_final = NIL;

  pips_debug(2,"begin\n");

  FOREACH(REGION, reg, regions_l)
    {
      if (step_private_p(stmt,region_entity(reg)))
	{
	  /* remove omp private from send and recv regions */
	  pips_debug(2,"drop private entity %s\n", entity_name(region_entity(reg)));
	  continue;
	}
      if (io_effect_p(reg) || std_file_effect_p(reg))
	{
	  pips_debug(2,"drop I/O effect on %s\n", entity_name(region_entity(reg)));
	  continue;
	}
      if (FILE_star_effect_reference_p(region_any_reference(reg)))
	{
	  pips_debug(2,"drop effect on FILE * : %s\n", entity_name(region_entity(reg)));
	  continue;
	}
      if (entity_scalar_p(region_entity(reg)))
	{
	  pips_debug(2,"drop scalar region %s\n", entity_name(region_entity(reg)));
	  continue;
	}

      /* Remove precondition contrainte */
      transformer stmt_precondition = load_statement_precondition(stmt);
      Psysteme sc = predicate_system(transformer_relation(stmt_precondition));
      region_system(reg) = extract_nredund_subsystem(region_system(reg), sc);

      region r = region_dup(rectangularization_region(reg));

      free_action(region_action(r));
      switch (action_tag)
	{
	case is_action_read:
	  region_action(r) = make_action_read_memory();
	  break;
	case is_action_write:
	  region_action(r) = make_action_write_memory();
	  break;
	default:
	  pips_assert("unknown action_tag", 0);
	}

      regions_final = CONS(REGION, r, regions_final);
    }
  gen_sort_list(regions_final, (gen_cmp_func_t)compare_effect_reference);

  pips_debug(2,"end\n");
  return regions_final;
}

static void step_init_effect_path(entity module, statement stmt, list regions_l)
{
  pips_debug(4,"begin\n");

  FOREACH(REGION, reg, regions_l)
    {
      /* Add an initial point into the path,
	 key effect is equal to data field (effect)
	 used as stop condition in step_get_path() */
      step_add_point_into_effect_path(reg, module, stmt, reg);
    }

  pips_debug(4,"end\n");
}

static bool anymodule_anywhere_region_p(list regions_l)
{
  bool is_anymodule_anywhere = false;

  FOREACH(REGION, reg, regions_l)
    {
      if (anywhere_effect_p(reg))
	{
	  is_anymodule_anywhere = true;
	  break;
	}
    }

  return is_anymodule_anywhere;
}

static list compute_send_regions(list write_l, list out_l, statement stmt)
{
  pips_debug(2,"begin\n");
  list send_final = NIL;
  list write_tmp_l = regions_dup(write_l);
  list out_tmp_l = regions_dup(out_l);

  /*
     SEND = OUT Inter WRITE

     Computation of the approximation (EXACT or MAY)

     1) OUT-MAY Inter WRITE-MAY -> SEND-MAY
     2) OUT-EXACT Inter WRITE-MAY -> SEND-MAY
     3) OUT-EXACT Inter WRITE-EXACT -> SEND-EXACT
     4) OUT-MAY Inter WRITE-EXACT -> SEND-MAY

     Case 4 is a problem, we want OUT-MAY Inter WRITE-EXACT --> SEND-*EXACT*.

     Generally the approximation of the SEND region should be the
     approximation of the WRITE region because OUT should be used only
     to reduce region communication. Thus we artificially transform
     the out approximation as EXACT so that the MAY approximation does
     not impact the SEND approximation.
  */

  FOREACH(REGION, reg, out_tmp_l)
    {
      free_approximation(region_approximation(reg));
      region_approximation(reg)=make_approximation_exact();
    }

  ifdebug(2)
    {
      debug_print_effects_list(out_tmp_l, "OUT :");
      debug_print_effects_list(write_tmp_l, "WRITE :");
    }

  list send_l = RegionsIntersection(out_tmp_l, write_tmp_l, w_w_combinable_p);

  if (anymodule_anywhere_region_p(send_l))
    {
      /* ANYMODULE ANYWHERE send regions are a big problem! FIXME.
      */

      STEP_DEBUG_STATEMENT(0, "ANYWHERE effect on stmt", stmt);
      pips_debug(0, "end stmt\n");
      pips_user_warning("ANYWHERE effect in SEND regions\n");
    }

  send_final = create_step_regions(stmt, send_l, is_action_write);
  gen_full_free_list(send_l);

  ifdebug(2)
    debug_print_effects_list(send_final, "SEND final :");

  pips_debug(2,"end\n");
  return send_final;
}

static list compute_recv_regions(list send_l, list in_l, statement stmt)
{
  pips_debug(2,"begin\n");
  list recv_final = NIL;
  list send_may_l = NIL;
  list recv_l;

  /*
     RECV = IN Union SEND-MAY

     Generally, for a given construct, SEND is not sur-approximated
     then SEND-MAY is empty thus

     RECV = IN

     But, for a given construct, when SEND is sur-approximated (i.e. SEND
     MAY), then the communication is interlaced and a diff is
     necessary at the end to determine what data were updated.

     Thus data in the sur-approximation must have a correct value even
     if they are not read (for the diff).

     Thus when SEND is sur-approximated: RECV = IN Union SEND-MAY
  */

  FOREACH(REGION, reg, send_l)
    {
      if (region_may_p(reg))
	send_may_l = CONS(REGION, region_dup(reg), send_may_l);
    }

  ifdebug(2)
    {
      debug_print_effects_list(send_may_l, "SEND may :");
      debug_print_effects_list(in_l, "IN :");
    }

  recv_l = RegionsMustUnion(regions_dup(in_l), regions_dup(send_may_l), r_w_combinable_p);
  gen_full_free_list(send_may_l);

  if (anymodule_anywhere_region_p(recv_l))
    {
      /* ANYMODULE ANYWHERE recv regions should cause FULL communications
	 thus ANYMODULE ANYWHERE recv regions are removed from recv regions.
       */
      pips_debug(2,"drop ANYMODULE ANYWHERE RECV regions\n");
      recv_final = NIL;
      gen_full_free_list(recv_l);
    }
  else
    {
      recv_final = create_step_regions(stmt, recv_l, is_action_read);
      gen_full_free_list(recv_l);
    }

  ifdebug(2)
    debug_print_effects_list(recv_final, "RECV final :");

  pips_debug(2,"end\n");
  return recv_final;
}

/*
  For a given region and two different iterations
  check if overlap
 */

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
  pips_debug(2,"check interlaced %s : %s\n", entity_name(region_entity(reg)), interlaced_p?"true":"false");
  return interlaced_p;
}

/*
 *
 *
 * SEND/RECV regions list management
 *
 *
 */
static list step_send_regions_list_load(statement s)
{
  pips_debug(4, "begin\n");
  if (!bound_step_send_regions_p(s))
    return NIL;

  effects e = load_step_send_regions(s);
  ifdebug(8) pips_assert("send regions loaded are consistent", effects_consistent_p(e));
  pips_debug(4, "end\n");
  return(effects_effects(e));
}

static void step_send_regions_list_store(statement s, list l_regions)
{
  pips_debug(4, "begin\n");
  effects e = make_effects(l_regions);
  ifdebug(8) pips_assert("send regions to store are consistent", effects_consistent_p(e));
  store_step_send_regions(s, e);
  pips_debug(4, "end\n");
}

static void step_send_regions_list_update(statement s, list l_regions)
{
  pips_debug(4, "begin\n");
  if (bound_step_send_regions_p(s))
    {
      effects e = load_step_send_regions(s);
      effects_effects(e) = l_regions;
      update_step_send_regions(s, e);
    }
  pips_debug(4, "end\n");
}

static void step_send_regions_list_add(statement stmt, list l_regions)
{
  pips_debug(4, "begin\n");
  pips_assert ("step_send_regions", bound_step_send_regions_p(stmt));

  effects e = load_step_send_regions(stmt);
  effects_effects(e) = gen_nconc(l_regions, effects_effects(e));
  update_step_send_regions(stmt, e);

  pips_debug(4, "end\n");
}

static list step_recv_regions_list_load(statement s)
{
  pips_debug(4, "begin\n");
  if (!bound_step_recv_regions_p(s))
    return NIL;

  effects e = load_step_recv_regions(s);
  ifdebug(8) pips_assert("recv regions loaded are consistent", effects_consistent_p(e));

  pips_debug(4, "end\n");
  return(effects_effects(e));
}

static void step_recv_regions_list_store(statement s, list l_regions)
{
  pips_debug(4, "begin\n");
  effects e = make_effects(l_regions);
  ifdebug(8) pips_assert("recv regions to store are consistent", effects_consistent_p(e));
  store_step_recv_regions(s, e);
  pips_debug(4, "end\n");
}

static void step_recv_regions_list_update(statement s, list l_regions)
{
  pips_debug(4, "begin\n");
  if (bound_step_recv_regions_p(s))
    {
      effects e = load_step_recv_regions(s);
      effects_effects(e) = l_regions;
      update_step_recv_regions(s, e);
    }
  pips_debug(4, "end\n");
}

static void step_recv_regions_list_add(statement stmt, list l_regions)
{
  pips_debug(4, "begin\n");

  pips_assert ("step_recv_regions", bound_step_recv_regions_p(stmt));

  effects e = load_step_recv_regions(stmt);
  effects_effects(e) = gen_nconc(l_regions, effects_effects(e));
  update_step_recv_regions(stmt, e);

  pips_debug(4, "end\n");
}

static void step_print_directives_regions(step_directive d, list send_l, list recv_l)
{
  pips_debug(2, "begin\n");
  ifdebug(2)
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
  pips_debug(2, "end\n");
}

static list summarize_and_map_step_regions(list effect_l, entity module, statement body)
{
  pips_debug(4, "begin\n");
  list summarized_l = NIL;

  FOREACH(EFFECT, eff, effect_l)
    {
      effect new_eff = copy_effect(eff);

      pips_debug(4, "tmp summarize eff = %p new_eff = %p\n", eff, new_eff);
      step_add_point_into_effect_path(new_eff, module, body, eff);
      summarized_l = CONS(EFFECT, new_eff, summarized_l);
    }

  pips_debug(4, "end\n");
  return summarized_l;
}

/*
  For each step region (send or recv), summarizes the region at the module level
 */

static void step_summarize_and_map_step_regions(statement stmt)
{
  /*
    For the statement stmt included in a module,
        add the list of regions at the statement level to the list of regions at the module level.

    Summarized_send_l (resp. recv_l) means list of send (resp. recv)
    regions at the module level.
  */
  pips_debug(4, "begin\n");

  /* search for the first stmt of type directive_stmt in the path from the
     body to stmt S

     if the first_directive_stmt is different than stmt, it means that
     stmt is imbricated inside another directive.

     In case of an imbricated statement, regions attached to stmt are
     not summarized at the module level.
  */
  statement first_directive_stmt = step_statement_path_first_directive_statement(stmt);

  if(statement_undefined_p(first_directive_stmt) || first_directive_stmt == stmt)
    {
      entity module = get_current_module_entity();
      statement body = get_current_module_statement();
      list summarized_send_l, summarized_recv_l;

      pips_assert("stmt != body", stmt != body);
      pips_assert("statement with step regions", bound_step_send_regions_p(stmt) && bound_step_recv_regions_p(stmt));

      summarized_send_l = summarize_and_map_step_regions(step_send_regions_list_load(stmt), module, body);
      /* Store SEND summary regions */
      step_send_regions_list_add(body, summarized_send_l);
      summarized_recv_l = summarize_and_map_step_regions(step_recv_regions_list_load(stmt), module, body);
      /* Store RECV summary regions */
      step_recv_regions_list_add(body, summarized_recv_l);

      pips_debug(4, " SEND and RECV propagated on module %s\n", entity_name(module));
    }
  pips_debug(4, "end\n");
}

static void step_initialize_step_partial(list send_l)
{
  FOREACH(REGION, reg, send_l)
    {
      /* All SENDs are initialized as PARTIAL */
      step_set_communication_type_partial(reg);
    }
}


static void step_compute_step_interlaced(list send_l, step_directive d)
{
  list index_l = step_directive_basic_workchunk_index(d);
  FOREACH(REGION, reg, send_l)
    {
      bool interlaced_p = interlaced_basic_workchunk_regions_p(reg, index_l);
      store_step_interlaced(reg, interlaced_p);
    }
}

/*
  return true if some region exists
*/
static bool compute_directive_regions(step_directive drt)
{
  entity module = get_current_module_entity();

  pips_debug(1,"Begin\n");

  statement directive_stmt = step_directive_block(drt);
  statement stmt_basic_workchunk = step_directive_basic_workchunk(drt);
  assert(!statement_undefined_p(stmt_basic_workchunk));

  ifdebug(1)
    {
      step_directive_print(drt);
    }

  list rw_l = load_rw_effects_list(stmt_basic_workchunk);
  list write_l = regions_write_regions(rw_l);
  list in_l = load_in_effects_list(stmt_basic_workchunk);
  list out_l = load_out_effects_list(stmt_basic_workchunk);

  /* TODO ajouter un test si tout est vide. */

  list send_l = compute_send_regions(write_l, out_l, directive_stmt);
  list recv_l = compute_recv_regions(send_l, in_l, directive_stmt);


  /* Store SEND/RECV directive statement regions */
  step_send_regions_list_store(directive_stmt, send_l);
  step_recv_regions_list_store(directive_stmt, recv_l);

#ifdef FRED
  pips_assert("ANYMODULE ANYWHERE SEND regions", !anymodule_anywhere_region_p(send_l));
  pips_assert("ANYMODULE ANYWHERE RECV regions", !anymodule_anywhere_region_p(recv_l));
#endif

  /* Initialization of step_effect_path for each send/recv regions */
  step_init_effect_path(module, directive_stmt, send_l);
  step_init_effect_path(module, directive_stmt, recv_l);

  /* Initialization of step_interlaced and step_partial for each send region */
  step_initialize_step_partial(send_l);
  step_compute_step_interlaced(send_l, drt);

  ifdebug(3)
    {
      step_print_directives_regions(drt, send_l, recv_l);
    }
  pips_debug(1,"End\n");
  return !(ENDP(recv_l) && ENDP(send_l));
}

static list step_translate_and_map(statement stmt, list effects_called_l)
{
  pips_debug(2, "begin\n");

  effect translated_eff;
  list translated_l = NIL;
  entity module = get_current_module_entity();
  entity called = call_function(statement_call(stmt));
  list args = call_arguments(statement_call(stmt));
  transformer context = load_statement_precondition(stmt); /* for generic_effects_backward_translation() */

  make_effects_private_current_context_stack(); /* for generic_effects_backward_translation()
						   cf. summary_rw_effects_engine()
						   dans Libs/effects-generic/rw_effects_engine.c */

  FOREACH(EFFECT, called_eff, effects_called_l)
    {
      /* to avoid low level debug messages of generic_effects_backward_translation */
      setenv("REGION_TRANSLATION_DEBUG_LEVEL", "0", 0);
      debug_on("REGION_TRANSLATION_DEBUG_LEVEL");

      /* create a new translated effect in list caller_l */
      list tmp_l = CONS(EFFECT, called_eff, NIL);
      list caller_l = generic_effects_backward_translation(called, args, tmp_l, context);

      debug_off();

      if(gen_length(caller_l)>1)
	{
	  pips_user_warning("to many effect at callsite : %d effect\n", gen_length(caller_l));
	  pips_debug(1, "input effect list (called_eff) :\n %s", text_to_string(text_rw_array_regions(tmp_l)));
	  pips_debug(0, "output effect list (caller_l) :\n %s", text_to_string(text_rw_array_regions(caller_l)));
	  list keep_l = NIL;

	  FOREACH(EFFECT, eff, caller_l)
	    {
	      if (anywhere_effect_p(eff) && effect_action_tag(called_eff)==effect_action_tag(eff))
		{
		  gen_free_list(keep_l);
		  keep_l = CONS(EFFECT, eff, NIL);
		  pips_debug(0, "keep only %s", text_to_string(text_rw_array_regions(keep_l)));
		  break;
		}

	      reference r = effect_any_reference(eff);
	      if(entity_array_p(reference_variable(r)))
		keep_l = CONS(EFFECT, eff, keep_l);
	      else
		pips_debug(0, "drop effect on not array entity : %s", text_to_string(text_rw_array_regions(CONS(EFFECT, eff, NIL))));
	    }
	  gen_free_list(caller_l);
	  caller_l = keep_l;
	}

      /* map */
      switch (gen_length(caller_l))
	{
	case 1:
	  /* one called effect should produce a unique translated effect */
	  translated_eff = EFFECT(CAR(caller_l));
	  pips_debug(2, "translate called_eff = %p, translated_eff = %p\n", called_eff, translated_eff);
	  step_add_point_into_effect_path(translated_eff, module, stmt, called_eff);
	  translated_l = gen_nconc(translated_l, caller_l);
	case 0:
	  break;
	default:
	  pips_debug(0, "Error\n");
	  pips_debug(0, "input effect list (called_eff) :\n %s\n", text_to_string(text_rw_array_regions(tmp_l)));
	  pips_debug(0, "output effect list (caller_l) :\n %s\n", text_to_string(text_rw_array_regions(caller_l)));
	  pips_assert("gen_length(caller_l)<2", 0);
	}
      gen_free_list(tmp_l);
    }

  free_effects_private_current_context_stack();
  pips_debug(2, "end\n");
  return translated_l;
}

/*
  return true if some region exists
*/
static bool step_translate_and_map_step_regions(statement stmt)
{
  entity called;
  list caller_l, called_l;
  statement_effects regions;
  bool new_region_p = false;

  pips_debug(2, "begin\n");
  assert(statement_call_p(stmt));
  called = call_function(statement_call(stmt));

  assert(entity_module_p(called));
  statement called_body = (statement) db_get_memory_resource(DBR_CODE, entity_user_name(called), true);

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
  caller_l = step_translate_and_map(stmt, called_l);
  /* Store SEND translated regions */
  step_send_regions_list_store(stmt, caller_l);

  new_region_p |= !ENDP(caller_l);

  ifdebug(2)
    {
      debug_print_effects_list(called_l, "CALLED REGIONS SEND");
      debug_print_effects_list(caller_l, "CALLER REGIONS SEND");
    }

  /* RECV */
  regions = (statement_effects)db_get_memory_resource(DBR_STEP_RECV_REGIONS, entity_user_name(called), true);
  called_l = effects_effects(apply_statement_effects(regions, called_body));
  caller_l = step_translate_and_map(stmt, called_l);
  /* Store RECV translated regions */
  step_recv_regions_list_store(stmt, caller_l);

  new_region_p |= !ENDP(caller_l);

  ifdebug(2)
    {
      debug_print_effects_list(called_l, "CALLED REGIONS RECV");
      debug_print_effects_list(caller_l, "CALLER REGIONS RECV");
    }

  pips_debug(2, "end\n");

  return new_region_p;
}

static bool compute_SENDRECV_regions(statement stmt, bool *exist_regions_p)
{
  assert(!statement_undefined_p(stmt));

  STEP_DEBUG_STATEMENT(2, "begin on stmt", stmt);

  if(step_directives_bound_p(stmt))
    {
      step_directive drt = step_directives_load(stmt);
      assert(stmt==step_directive_block(drt));
      *exist_regions_p |= compute_directive_regions(drt);

      /* default: SEND/RECV regions are summarized at the body level of the current module.
	 This will be modified later using the DG
       */
      step_summarize_and_map_step_regions(stmt);
    }
  else
    {
      /* When module call, compute translated SEND and RECV regions

	 Note: all entities present in the field call_function are not
	 a module ie it could also be a numerical constant, an
	 intrinsics...
       */
      if (statement_call_p(stmt) && entity_module_p(call_function(statement_call(stmt))))
	{
	  *exist_regions_p |= step_translate_and_map_step_regions(stmt);
	  /* default: SEND/RECV regions are summarized at the body level of the current module.
	     This will be modified later using the DG
	  */
          step_summarize_and_map_step_regions(stmt);
	}
      else
	pips_debug(2, "no STEP region to compute\n");
    }

  pips_debug(2, "end\n");
  return true;
}

static bool concerned_entity_p(effect eff, list regions)
{
  pips_debug(4, "begin\n");
  FOREACH(REGION, reg, regions)
    {
      pips_debug(5, "\n eff %s regions %s\n", text_to_string(text_region(eff)), text_to_string(text_region(reg)));
      if (effect_comparable_p(eff, reg))
	{
	  pips_debug(4, "end TRUE\n");
	  return true;
	}
    }
  pips_debug(4, "end FALSE\n");
  return false;
}

static void step_get_comparable_effects(list eff_list, effect eff, set *effects_set)
{
  pips_debug(2, "begin\n");
  FOREACH(EFFECT, eff2, eff_list)
    {
      if (effect_comparable_p(eff, eff2))
	{
	  pips_debug(2,"############################# unoptimizable %p %s\n", eff2, entity_name(effect_entity(eff2)));
	  *effects_set = set_add_element(*effects_set, *effects_set, eff2);
	}
    }
  pips_debug(2, "end\n");
}

static void step_compute_CHAINS_DG_remove_summary_regions(successor su, list source_send_l, list sink_recv_l, set *remove_from_summary_send, set *remove_from_summary_recv)
{
  pips_debug(2, "begin\n");
  /* a arc label in this CHAINS DG graph is a list of conflicts, see dg.pdf */

  FOREACH(CONFLICT, c, dg_arc_label_conflicts((dg_arc_label)successor_arc_label(su)))
    {
      effect source_eff = conflict_source(c);
      effect sink_eff = conflict_sink(c);

      pips_debug(2,"Conflict source_eff sink_eff\n");

      if(!(effect_write_p(source_eff) && effect_read_p(sink_eff)))
	{
	  pips_debug(2,"not WRITE->READ dependence (ignored)\n");
	  continue;
	}


      /* WRITE-READ dependence */
      pips_debug(2, "Dependence :\n\t\t%s from\t%s\t\t%s to  \t%s",
		 effect_to_string(source_eff), text_to_string(text_region(source_eff)),
		 effect_to_string(sink_eff), text_to_string(text_region(sink_eff)));

      bool source_send_p = concerned_entity_p(source_eff, source_send_l);
      bool sink_recv_p = concerned_entity_p(sink_eff, sink_recv_l);

      if (source_send_p)
	{
	  if (sink_recv_p)
	    {
		pips_debug(2,"optimizable SEND, SEND and RECV are already summarized at the module level\n");
	    }
	  else
	    {
	      pips_debug(2,"unoptimizable SEND (no matching RECV)\n");

	      step_get_comparable_effects(source_send_l, source_eff, remove_from_summary_send);
	    }

	  continue;
	}

      if (sink_recv_p)
	{
	  pips_debug(2,"no corresponding SEND, remove RECV from summarized list at the module level\n");

	  step_get_comparable_effects(sink_recv_l, sink_eff, remove_from_summary_recv);
	  continue;
	}

      pips_debug(2,"no SEND/RECV for %s\n",effect_to_string(source_eff));
    }
  pips_debug(2, "end\n");
}

static void step_compute_CHAINS_DG_SENDRECV_regions(statement s1, statement s2, list *s1_send_l, list *s2_recv_l)
{
  pips_debug(3, "begin\n");

  /* compute first directive statement on suffix_sp1 and suffix_sp2 */
  list suffix_sp1, suffix_sp2;
  list __attribute__ ((unused)) common_sp = step_statement_path_factorise(s1, s2, &suffix_sp1, &suffix_sp2);

  statement first_directive_stmt1 = gen_find_if((gen_filter_func_t)step_directives_bound_p, suffix_sp1, (void * (*)(const union gen_chunk))gen_identity);
  statement first_directive_stmt2 = gen_find_if((gen_filter_func_t)step_directives_bound_p, suffix_sp2, (void * (*)(const union gen_chunk))gen_identity);


  if(!statement_undefined_p(first_directive_stmt1))
    {
      STEP_DEBUG_STATEMENT(3, "directive on statement path suffix_sp1", first_directive_stmt1);
      /* Get directive region */
      *s1_send_l = step_send_regions_list_load(first_directive_stmt1);
    }
  else
    {
      pips_debug(3, "no directive on statement path suffix_sp1\n");
      /* no directive on path thus if regions corresponding to s1 exist, they are translated regions */
      if (bound_step_send_regions_p(s1))
	*s1_send_l = step_send_regions_list_load(s1);
    }

  if(!statement_undefined_p(first_directive_stmt2))
    {
      STEP_DEBUG_STATEMENT(3, "directive on statement path suffix_sp2", first_directive_stmt2);
      /* Get directive region */
      *s2_recv_l = step_recv_regions_list_load(first_directive_stmt2);
    }
  else
    {
      pips_debug(3, "no directive on statement path suffix_sp2\n");
      /* no directive on path thus if regions corresponding to s1 exist, they are translated regions */
      if (bound_step_recv_regions_p(s2))
	*s2_recv_l = step_recv_regions_list_load(s2);
    }

  pips_debug(3, "end\n");
}

static void step_analyse_CHAINS_DG(const char *module_name, set *remove_from_summary_send, set *remove_from_summary_recv)
{
  graph dependences_graph = (graph) db_get_memory_resource(DBR_CHAINS, module_name, true);

  *remove_from_summary_send = set_make(set_pointer);
  *remove_from_summary_recv = set_make(set_pointer);

  pips_debug(1, "################ CHAINS DG %s ###############\n", module_name);

  /* for graph data structure, see newgen/graph.pdf */
  /* CHAINS is computed by the REGION_CHAINS phase. */

  FOREACH(VERTEX, v1, graph_vertices(dependences_graph))
    {
      FOREACH(SUCCESSOR, su, vertex_successors(v1))
	{
	  statement s1 = vertex_to_statement(v1);
	  statement s2 = vertex_to_statement(successor_vertex(su));
	  list s1_send_l = NIL;
	  list s2_recv_l = NIL;

	  ifdebug(2)
	    {
	      pips_debug(2, "new s1->s2 \n");
	      pips_debug(2, "from %s", safe_statement_identification(s1));
	      pips_debug(2, "to %s", safe_statement_identification(s2));
	      STEP_DEBUG_STATEMENT(2, "statement s1", s1);
	       STEP_DEBUG_STATEMENT(2, "statement s2", s2);
	    }


	  step_compute_CHAINS_DG_SENDRECV_regions(s1, s2, &s1_send_l, &s2_recv_l);

	  ifdebug(2)
	    {
	      debug_print_effects_list(s1_send_l, "SEND S1");
	      debug_print_effects_list(s2_recv_l, "RECV S2");
	    }

	  step_compute_CHAINS_DG_remove_summary_regions(su, s1_send_l, s2_recv_l, remove_from_summary_send, remove_from_summary_recv);

	  pips_debug(2, "end statement s1->s2\n");
	}
    }
  pips_debug(2, "end\n");
}


static void update_SUMMARY_SENDRECV_regions(set remove_from_summary_regions, list tmp_summary_regions_l, list *final_summary_regions_l)
{
  pips_debug(1, "begin\n");

 /*

    The effects in the remove_from_summary_send and
    remove_from_summary_send sets are not SUMMARY. They can be
    DIRECTIVE regions or TRANSLATED regions.

    They are built from SEND and RECV effects corresponding to
    statements S1 and S2 of the DG.

    We need to get pathpoint corresponding to the SUMMARY
    effects to retrieve the original effects (from which summary was
    computed).

    Because the pathpoint corresponds to a SUMMARY, the effect in the
    pathpoint data field can be either TRANSLATED or DIRECTIVE region.

   */

   FOREACH(EFFECT, summ_eff, tmp_summary_regions_l)
    {
      assert(bound_step_effect_path_p(summ_eff));

      step_point point = load_step_effect_path(summ_eff);
      effect from_region = step_point_data(point);

      /*
	from_region can be either DIRECTIVE or TRANSLATED region. See comments above.

	if from_region corresponding to the SUMMARY effect is not is
	the remove set then add the SUMMARY effect into the final SUMMARY regions list.
       */
      if(!set_belong_p(remove_from_summary_regions, from_region))
	*final_summary_regions_l = CONS(EFFECT, summ_eff, *final_summary_regions_l);
      else
	pips_debug(2, "drop unoptimizable tmp_summary %p -> data %p\n", summ_eff, step_point_data(point));
    }

  pips_debug(1, "end\n");
}

static void step_update_SUMMARY_SENDRECV_regions(set remove_from_summary_send, set remove_from_summary_recv, list *full_send_l, list *partial_send_l)
{
  statement body = get_current_module_statement();
  list tmp_summary_send;
  list tmp_summary_recv;
  list final_summary_send = NIL;
  list final_summary_recv = NIL;

  pips_debug(2, "begin\n");

  /* Regions corresponding to body are SYMMARY regions */
  tmp_summary_send = step_send_regions_list_load(body);
  update_SUMMARY_SENDRECV_regions(remove_from_summary_send, tmp_summary_send, &final_summary_send);
  /* Store SEND summary regions */
  step_send_regions_list_update(body, final_summary_send);

  /* Regions corresponding to body are SYMMARY regions */
  tmp_summary_recv = step_recv_regions_list_load(body);
  update_SUMMARY_SENDRECV_regions(remove_from_summary_recv, tmp_summary_recv, &final_summary_recv);
  /* Store RECV summary regions */
  step_recv_regions_list_update(body, final_summary_recv);

  ifdebug(2)
    {
      debug_print_effects_list(final_summary_send, "FINAL SUMMARIZED SEND");
      debug_print_effects_list(final_summary_recv, "FINAL SUMMARIZED RECV");
    }

  /*
     full_send_l are TRANSLATED or DIRECTIVE regions.
     partial_send_l are SUMMARY regions.

    They will both be to retrieve the corresponding original DIRECTIVE
    region.
    This can be done from any SUMMARY or TRANSLATED and DIRECTIVE (of
    course) kind of region.
  */

  *full_send_l = set_to_list(remove_from_summary_send);
  *partial_send_l = final_summary_send;
  pips_debug(2, "end\n");
}

static void step_update_comm(list full_send_l, list partial_send_l)
{
  pips_debug(2, "begin\n");

  FOREACH(EFFECT, eff, full_send_l)
    {
      step_set_communication_type_full(eff);
    }
  FOREACH(EFFECT, eff, partial_send_l)
    {
      step_set_communication_type_partial(eff);
    }

  ifdebug(1)
    if(entity_main_module_p(get_current_module_entity()))
      {
	pips_debug(1, "################ COMM ###############\n");
	MAP_EFFECT_BOOL_MAP(eff, partial,
			    {
			      step_point point = load_step_effect_path(eff);
			      pips_debug(1, "module : %s\n", entity_user_name(step_point_module(point)));
			      ifdebug(2)
				print_statement(step_point_stmt(point));
			      debug_print_effects_list(CONS(EFFECT,step_point_data(point),NIL), partial?"COMMUNICATION PARTIAL":"COMMUNICATION FULL");
			    }, get_step_partial());
      }

  pips_debug(2, "end\n");
}

static bool step_compute_SENDRECV_regions(entity module, statement body)
{
  bool exist_regions_p = false;
  pips_debug(2, "################ REGIONS  %s ###############\n", entity_name(module));

  /* Initialization of summary regions */
  step_send_regions_list_store(body, NIL);
  step_recv_regions_list_store(body, NIL);

  gen_context_recurse(body, &exist_regions_p, statement_domain, compute_SENDRECV_regions, gen_null);

  pips_debug(2, "end exist_regions_p = %d\n", exist_regions_p);
  return exist_regions_p;
}


bool step_analyse(const char* module_name)
{
  debug_on("STEP_ANALYSE_DEBUG_LEVEL");
  bool exist_regions_p;
  pips_debug(2, "begin %d module_name = %s\n", __LINE__, module_name);

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

  step_directives_init(0);

  load_step_comm();

  step_statement_path_init(body);

  init_step_send_regions();
  init_step_recv_regions();

  exist_regions_p = step_compute_SENDRECV_regions(module, body);

  pips_debug(2, "exist_regions_p = %d\n", exist_regions_p);

  if(exist_regions_p)
    {
      set remove_from_summary_send, remove_from_summary_recv;
      list full_send_l, partial_send_l;

      step_analyse_CHAINS_DG(module_name, &remove_from_summary_send, &remove_from_summary_recv);

      /* remove_from_summary_send et remove_from_summary_recv contain
	 only directive regions and translated regions. They do not
	 contain summary regions. */

      step_update_SUMMARY_SENDRECV_regions(remove_from_summary_send, remove_from_summary_recv, &full_send_l, &partial_send_l);

      /*
	full_send_l are TRANSLATED or DIRECTIVE regions.
	partial_send_l are SUMMARY regions.
      */
      step_update_comm(full_send_l, partial_send_l);
    }

  DB_PUT_MEMORY_RESOURCE(DBR_STEP_SEND_REGIONS, module_name, get_step_send_regions());
  DB_PUT_MEMORY_RESOURCE(DBR_STEP_RECV_REGIONS, module_name, get_step_recv_regions());

  reset_step_send_regions();
  reset_step_recv_regions();

  step_statement_path_finalize();

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

  pips_debug(2, "End step_analyse\n");
  debug_off();
  return true;
}

