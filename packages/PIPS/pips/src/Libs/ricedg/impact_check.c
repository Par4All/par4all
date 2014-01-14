/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

  This file is part of PIPS.

  PIPS is free software: you can redistribute it and/or modify it
  under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  any later version.

  PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.

  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with PIPS.  If not, see <http://www.gnu.org/licenses/>.

*/
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
#include "local.h"

#define DEP_FLOW 0
#define DEP_ANTI 1
#define DEP_OUTP 2

#define MUST_APPROXIMATION 0
#define MAY_APPROXIMATION 1
#define NOT_APPROXIMATION 2

/* We want to keep track of the current statement inside the recurse */
/*DEFINE_LOCAL_STACK(current_stmt, statement)*/

static statement mod_stat = statement_undefined;
static entity current_mod = entity_undefined;
static graph dg = NULL; /* data dependence graph */
static entity alias_ent1 = entity_undefined;
static entity alias_ent2 = entity_undefined;
static list stat_reads1 = NIL; /* list of statement_approximation_p */
static list stat_writes1 = NIL; /* list of statement_approximation_p */
static list stat_reads2 = NIL; /* list of statement_approximation_p */
static list stat_writes2 = NIL; /* list of statement_approximation_p */

static list current_path = NIL;
/* This list tells us if two variables have been checked dynamically or not */
static list l_dynamic_check = NIL;
static int statement_in_caller_ordering = 0;
static statement statement_in_caller = statement_undefined;
static const char* caller_name;
static call current_call = call_undefined;
static int number_of_processed_modules = 0;
static int number_of_impact_alias = 0;

static void display_impact_alias_statistics()
{
  user_log("\n Number of processed modules: %d", number_of_processed_modules);
  user_log("\n Number of impact alias: %d\n", number_of_impact_alias);
}

static bool same_call_site_p(call_site cs1, call_site cs2)
{
  entity f1 = call_site_function(cs1);
  entity f2 = call_site_function(cs2);
  int o1 = call_site_ordering(cs1);
  int o2 = call_site_ordering(cs2);
  return (same_entity_p(f1,f2) && (o1==o2));
}

/****************************************************************************

 This function returns true if l1 = conc(cs,l2)

*****************************************************************************/

static bool tail_call_path_p(call_site cs, list l1, list l2)
{
  if (gen_length(l1) == gen_length(l2)+1)
    {
      call_site cs1 = CALL_SITE(CAR(l1));
      return (same_call_site_p(cs,cs1) && included_call_chain_p(l2,CDR(l1)));
    }
  return false;
}

/*****************************************************************************

   This function computes the offset of a storage ram variable :
   offset = initial_offset + subscript_value_stride

*****************************************************************************/

static expression storage_ram_offset(storage s,expression subval)
{
  ram r = storage_ram(s);
  int initial_off = ram_offset(r);
  expression exp = int_to_expression(initial_off);
  if (!expression_equal_integer_p(subval,0))
    {
      if (initial_off == 0)
	exp = copy_expression(subval);
      else
	exp = binary_intrinsic_expression(PLUS_OPERATOR_NAME,
					  int_to_expression(initial_off),
					  copy_expression(subval));
    }
  return exp;
}

/****************************************************************************

   This function computes the offset of a formal storage variable :
   offset = initial_offset + subscript_value_stride

   initial_offset is from alias_association with path' = path - {cs}

*****************************************************************************/

static expression storage_formal_offset(call_site cs,entity actual_var,
					expression subval,list path)
{
  list l_caller_aliases = alias_associations_list((alias_associations)
        db_get_memory_resource(DBR_ALIAS_ASSOCIATIONS,caller_name,true));
  expression exp = expression_undefined;
  MAP(ALIAS_ASSOCIATION, aa,
  {
    entity caller_var = alias_association_variable(aa);
    list caller_path = alias_association_call_chain(aa);
    if (same_entity_p(caller_var,actual_var) && tail_call_path_p(cs,caller_path,path))
      {
	expression initial_off = alias_association_offset(aa);
	if (!expression_undefined_p(initial_off))
	  {
	    if (expression_equal_integer_p(subval,0))
	      exp = copy_expression(initial_off);
	    else
	      exp = binary_intrinsic_expression(PLUS_OPERATOR_NAME,
						copy_expression(initial_off),
						copy_expression(subval));
	  }
	return exp;
      }
  },
      l_caller_aliases);
  return exp;
}


/*****************************************************************************

 If e is a formal parameter, find its rank in the formal parameter list of
 current module in order to find out the corresponding actual argument and
 then its offset

 If e is a common variable in the current module, offset of e is constant

*****************************************************************************/

static expression offset_in_caller(entity e, call_site cs, list path)
{
  ram ra ;
  if (formal_parameter_p(e))
    {
      formal f = storage_formal(entity_storage(e));
      int rank = formal_offset(f);
      list l_args = call_arguments(current_call);
      expression actual_arg = find_ith_argument(l_args,rank);
      reference actual_ref = expression_reference(actual_arg);
      entity actual_var = reference_variable(actual_ref);
      list l_actual_inds = reference_indices(actual_ref);
      /* compute the subscript value, return expression_undefined if
	 if the actual argument is a scalar variable or array name*/
      expression subval = subscript_value_stride(actual_var,l_actual_inds);
      storage s = entity_storage(actual_var);
      ifdebug(3)
	fprintf(stderr, " \n Current actual argument %s",entity_name(actual_var));
      if (storage_ram_p(s))
	/* The actual argument has a ram storage */
	return storage_ram_offset(s,subval);
      if (storage_formal_p(s))
	/* The actual argument is a formal parameter of the current caller,
	   we must take the alias_associations of the caller */
	return storage_formal_offset(cs,actual_var,subval,path);
    }
  // common variable
  ra = storage_ram(entity_storage(e));
  return int_to_expression(ram_offset(ra));
}

static bool search_statement_by_ordering_flt(statement s)
{
    if (statement_ordering(s) == statement_in_caller_ordering) {
        statement_in_caller = s;
	return false;
    }
    return true;
}

static bool statement_equal_p(statement s1, statement s2)
{
    return (statement_number(s1) == statement_number(s2));
}

static vertex statement_to_vertex(statement s, graph g)
{
    MAP(VERTEX, v, {
        statement sv = vertex_to_statement(v);
	if (statement_equal_p(s, sv))
	    return v;
    }, graph_vertices(g));
    return vertex_undefined;
}

/* Check to see if new dependence is covered by arcs in dependence graph at reference level */
bool find_covering_reference_path(set arcs_processed_set,
				  statement s_src,
				  action act_src,
				  entity ent_src,
				  statement s_dest,
				  action act_dest,
				  entity ent_dest)
{
  if (statement_equal_p(s_src, s_dest))
    {
      if (entities_may_conflict_p(ent_src, ent_dest))
	return (action_write_p(act_dest) || action_read_p(act_src));
      else
	return (action_write_p(act_dest) && action_read_p(act_src));
    }
  else
    {
      vertex ver_src = statement_to_vertex(s_src, dg);
      MAP(SUCCESSOR, su, {
	statement s_tmp = vertex_to_statement(successor_vertex(su));
	dg_arc_label dal = (dg_arc_label) successor_arc_label(su);
	MAP(CONFLICT, c, {
	  effect e_tmp_src = conflict_source(c);
	  effect e_tmp_dest = conflict_sink(c);
	  entity ent_tmp_src = reference_variable(effect_any_reference(e_tmp_src));
	  entity ent_tmp_dest = reference_variable(effect_any_reference(e_tmp_dest));
	  action act_tmp_src = effect_action(e_tmp_src);
	  action act_tmp_dest = effect_action(e_tmp_dest);
	  set arcs_processed_tmp_set = set_make(set_pointer);
	  if (set_belong_p(arcs_processed_set, (char *)c)) continue;
	  arcs_processed_tmp_set = set_add_element(arcs_processed_tmp_set, arcs_processed_set, (char *)c);
	  if (entities_may_conflict_p(ent_src, ent_tmp_src))
	    {
	      if (action_write_p(act_tmp_src) || action_read_p(act_src))
		if (find_covering_reference_path(arcs_processed_tmp_set,
						s_tmp, act_tmp_dest, ent_tmp_dest,
						s_dest, act_dest, ent_dest)) return true;
	    }
	  else
	    {
	      if (action_write_p(act_tmp_src) && action_read_p(act_src))
		if (find_covering_reference_path(arcs_processed_tmp_set,
						s_tmp, act_tmp_dest, ent_tmp_dest,
						s_dest, act_dest, ent_dest)) return true;
	    }
	}, dg_arc_label_conflicts(dal));
      }, vertex_successors(ver_src));
    }
  return false;
}

/* Check to see if there is a directed way between 2 statements in the graph specified */
bool check_way_between_two_statements(statement s1, statement s2, graph g)
{
    vertex v1 = statement_to_vertex(s1, g);
    list vertex_processed_list = NIL;
    list vertex_rest_list = NIL;
    ADD_ELEMENT_TO_LIST(vertex_rest_list, VERTEX, v1);

    while(!ENDP(vertex_rest_list)) {
        vertex v = VERTEX(CAR(vertex_rest_list));
	POP(vertex_rest_list);
	if (!gen_in_list_p(v, vertex_processed_list)) {
	    statement sv = vertex_to_statement(v);
	    if (statement_equal_p(sv, s2))
	        return true;
	    else {
	        ADD_ELEMENT_TO_LIST(vertex_processed_list, VERTEX, v);
	        MAP(SUCCESSOR, su, {
		    vertex v2 = successor_vertex(su);
		    if (!gen_in_list_p(v2, vertex_processed_list) && !gen_in_list_p(v2, vertex_rest_list))
		        ADD_ELEMENT_TO_LIST(vertex_rest_list, VERTEX, v2);
		}, vertex_successors(v));
	    }
	}
    }
    return false;
}

/* This function prints the call path , including names of caller functions
   and orderings of call sites in their corresponding functions */
static string print_call_path(list path)
{
  list pc = NIL;
  MAP(CALL_SITE,casi,
  {
    entity casifunc = call_site_function(casi);
    int casiord = call_site_ordering(casi);
    pc = CHAIN_SWORD(pc,"(");
    pc = CHAIN_SWORD(pc,module_local_name(casifunc));
    pc = CHAIN_SWORD(pc,":(");
    pc = CHAIN_SWORD(pc,i2a(ORDERING_NUMBER(casiord)));
    pc = CHAIN_SWORD(pc,",");
    pc = CHAIN_SWORD(pc,i2a(ORDERING_STATEMENT(casiord)));
    pc = CHAIN_SWORD(pc,")) ");
  },path);
  return words_to_string(pc);
}


static void insert_impact_description_as_comment(statement s1, statement s2, bool impact_must_p, int dep_type)
{
  insert_comments_to_statement(s2, strdup(concatenate("C\t", text_to_string(statement_to_text(s1)), "\n", NULL)));
  switch(dep_type) {
  case DEP_FLOW:
    insert_comments_to_statement(s2, "C\tNew flow-dependence with statement\n");
    break;
  case DEP_ANTI:
    insert_comments_to_statement(s2, "C\tNew anti-dependence with statement\n");
    break;
  case DEP_OUTP:
    insert_comments_to_statement(s2, "C\tNew output-dependence with statement\n");
    break;
  }
  insert_comments_to_statement(s2, strdup(concatenate("C\tAttention: impact alias ",
						      impact_must_p ? "MUST":"MAY",
						      " at ",
						      print_call_path(current_path),
						      " between ",
						      entity_local_name(alias_ent1),
						      " and ",
						      entity_local_name(alias_ent2),
						      "\n",
						      NULL)));
  return;
}

/*
list union_list(list l1, list l2) {
    MAP(STATEMENT, s, {
        if (!gen_in_list_p(s, l1))
	  ADD_ELEMENT_TO_LIST(l1, STATEMENT, s);
    }, l2);
    return l1;
}
*/

/* Union is not typed... */
list union_list(list l1, list l2) {
  list cl = list_undefined;

  for(cl=l2; !ENDP(cl); POP(cl)) {
    gen_chunk * gcp = CHUNK(CAR(cl));
	if(!gen_in_list_p(gcp , l1))
	  l1 = gen_nconc(l1, gen_cons(gcp, NIL));
  }

  return l1;
}

static effect get_effect_read_of_statement_on_variable(statement s, entity var)
{
  MAP(EFFECT, eff, {
    entity e = reference_variable(effect_any_reference(eff));
    if (entities_may_conflict_p(e, var) && action_read_p(effect_action(eff)))
      return eff;
  }, statement_to_effects(s));
  return NULL;
}

static effect get_effect_write_of_statement_on_variable(statement s, entity var)
{
  MAP(EFFECT, eff, {
    entity e = reference_variable(effect_any_reference(eff));
    if (entities_may_conflict_p(e, var) && action_write_p(effect_action(eff)))
      return eff;
  }, statement_to_effects(s));
  return NULL;
}

static int __attribute__ ((unused)) expression_approximation(statement s, expression ex)
{
  normalized n = NORMALIZE_EXPRESSION(ex);
  transformer pre = load_statement_precondition(s);
  Psysteme precondition_ps = predicate_system(transformer_relation(pre));
  if (normalized_linear_p(n))
    {
      Pvecteur pv = vect_dup(normalized_linear(n));
      Pcontrainte volatile pc = contrainte_make(pv);
      /* Automatic variables read in a CATCH block need to be declared volatile as
       * specified by the documentation*/
      Psysteme volatile ps = sc_dup(precondition_ps);
      sc_add_ineg(ps, pc);
      CATCH(overflow_error) {
	sc_rm(ps);
	return MAY_APPROXIMATION;
      }
      TRY {
	bool n_positif, n_negatif;
	n_negatif = sc_rational_feasibility_ofl_ctrl(ps,FWD_OFL_CTRL,true);
	(void) vect_chg_sgn(pv);
	n_positif = sc_rational_feasibility_ofl_ctrl(ps,FWD_OFL_CTRL,true);
	fprintf(stderr, "n_negatif : %s\n", n_negatif ? "TRUE" : "FALSE");
	fprintf(stderr, "n_positif : %s\n", n_positif ? "TRUE" : "FALSE");
	UNCATCH(overflow_error);
      }
      sc_rm(ps);
    }
  return MAY_APPROXIMATION;
}

static int loop_executed_approximation(statement s)
{
  range rg = loop_range(statement_loop(s));
  normalized
    n_m1 = NORMALIZE_EXPRESSION(range_lower(rg)),
    n_m2 = NORMALIZE_EXPRESSION(range_upper(rg)),
    n_m3 = NORMALIZE_EXPRESSION(range_increment(rg));

  transformer pre = load_statement_precondition(s);
  Psysteme precondition_ps = predicate_system(transformer_relation(pre));

  if (normalized_linear_p(n_m1) && normalized_linear_p(n_m2) && normalized_linear_p(n_m3))
    {
      bool m3_negatif, m3_positif;
      /* Tester le signe de l'incr�ment en fonction des pr�conditions : */
      Pvecteur pv3 = vect_dup(normalized_linear(n_m3));
      Pcontrainte pc3 = contrainte_make(pv3);
      /* Automatic variables read in a CATCH block need to be declared volatile as
       * specified by the documentation*/
      Psysteme volatile ps = sc_dup(precondition_ps);
      sc_add_ineg(ps, pc3);
      CATCH(overflow_error) {
	sc_rm(ps);
	return MAY_APPROXIMATION;
      }
      TRY {
	m3_negatif = sc_rational_feasibility_ofl_ctrl(ps,FWD_OFL_CTRL,true);
	(void) vect_chg_sgn(pv3);
	m3_positif = sc_rational_feasibility_ofl_ctrl(ps,FWD_OFL_CTRL,true);
	UNCATCH(overflow_error);
      }
      pips_debug(2, "loop_increment_value positif = %d, negatif = %d\n",
		 m3_positif, m3_negatif);

      /* Vire aussi pv3 & pc3 : */
      sc_rm(ps);

      /* le signe est d�termin� et diff�rent de 0 */
      if (m3_positif ^ m3_negatif)
	{
	  Pvecteur pv1, pv2, pv, pv_inverse;
	  Pcontrainte c, c_inverse;

	  pv1 = normalized_linear(n_m1);
	  pv2 = normalized_linear(n_m2);

	  /* pv = m1 - m2 */
	  pv = vect_substract(pv1, pv2);
	  pv_inverse = vect_dup(pv);
	  /* pv_inverse = m2 - m1 */
	  (void)vect_chg_sgn(pv_inverse);

	  c = contrainte_make(pv);
	  c_inverse = contrainte_make(pv_inverse);

	  /* ??? on overflows, go next ... */
	  if(ineq_redund_with_sc_p(precondition_ps, c)) {
	    contrainte_free(c);
	    return (m3_positif ? MUST_APPROXIMATION : NOT_APPROXIMATION);
	  }
	  contrainte_free(c);

	  /* ??? on overflows, should assume MAY_BE_EXECUTED... */
	  if(ineq_redund_with_sc_p(precondition_ps, c_inverse)) {
	    contrainte_free(c_inverse);
	    return (m3_positif ? NOT_APPROXIMATION : MUST_APPROXIMATION);
	  }
	  contrainte_free(c_inverse);
	  return MAY_APPROXIMATION;
	}
    }
  return MAY_APPROXIMATION;
}

static set create_or_get_a_set_from_control(control c,
					    hash_table control_to_set_of_dominators)
{
  if (!hash_defined_p(control_to_set_of_dominators, (char *)c))
    {
      set dominator = set_make(set_pointer);
      hash_put(control_to_set_of_dominators, (char *)c, (char *)dominator);
      return dominator;
    }
  else
    return (set) hash_get(control_to_set_of_dominators, (char *)c);
}

static void __attribute__ ((unused)) computing_dominators(hash_table control_to_set_of_dominators, control n0)
{
  bool change = true;
  list blocs = NIL;
  int count = 0;
  control c_count = NULL;
  set set_N;
  set dominator_dn = create_or_get_a_set_from_control(n0, control_to_set_of_dominators);
  set_add_element(dominator_dn, dominator_dn, (char *)n0);

  /* compute the set N of nodes in flowgraph */
  set_N = set_make(set_pointer);
  CONTROL_MAP(n, set_add_element(set_N, set_N, (char *)n), n0, blocs);
  gen_free_list(blocs);
  blocs = NIL;

  /* Initialization D(n) = N */
  CONTROL_MAP(n, {
    if (n == n0) continue;
    dominator_dn = create_or_get_a_set_from_control(n, control_to_set_of_dominators);
    set_assign(dominator_dn, set_N);
  }, n0, blocs);
  gen_free_list(blocs);
  blocs = NIL;
  set_free(set_N);

  while (change)
    {
      list sub_blocs = NIL;
      change = false;
      CONTROL_MAP(n, {
	set dominator_dp = NULL;
	if (n == n0) continue;
	count++;
	dominator_dn = set_make(set_pointer);
	set_add_element(dominator_dn, dominator_dn, (char *)n);

	BACKWARD_CONTROL_MAP(pred, {
	  if (dominator_dp == NULL)
	    {
	      dominator_dp = set_make(set_pointer);
	      set_assign(dominator_dp,
			 create_or_get_a_set_from_control(pred, control_to_set_of_dominators));
	    } else
	      set_intersection(dominator_dp, dominator_dp,
			       create_or_get_a_set_from_control(pred, control_to_set_of_dominators));
	}, n, sub_blocs);
	gen_free_list(sub_blocs);
	sub_blocs = NIL;

	set_union(dominator_dn, dominator_dn, dominator_dp);

	if (!set_equal_p(dominator_dn,
			create_or_get_a_set_from_control(n, control_to_set_of_dominators)))
	  {
	    change = true;
	    hash_update(control_to_set_of_dominators, (char *)n, dominator_dn);
	  }
	if (count == 11)
	  c_count = n;

      }, n0, blocs);
      gen_free_list(blocs);
      blocs = NIL;
    }
  dominator_dn = create_or_get_a_set_from_control(c_count, control_to_set_of_dominators);
  fprintf(stderr, "dominators of statement\n");
  safe_print_statement(control_statement(c_count));
  count = 0;
  SET_MAP(dom, {
    control c = (control) dom;
    fprintf(stderr, "#%d : ", ++count);
    safe_print_statement(control_statement(c));
  }, dominator_dn);
}

static int __attribute__ ((unused)) control_approximation_between_statement_p(statement s1, statement s2)
{
  /* control_graph does not work until load_ctrl_graph is called
   * control
   *  c1 = unstructured_control(control_graph(s1)),
   *  c2 = unstructured_control(control_graph(s2));
   *
   * load_ctrl_graph work only if full_control_graph is called before
   * and clean_ctrl_graph is called after
   */
  control
    c1 = load_ctrl_graph(s1),
    c2 = load_ctrl_graph(s2);
  list blocs = NIL;
  statement s;
  int
    statement_exec_appro = MUST_APPROXIMATION,
    statement_exec_appro1 = MUST_APPROXIMATION,
    statement_exec_appro2 = MUST_APPROXIMATION;

  BACKWARD_CONTROL_MAP(pred, {
    s = control_statement(pred);
    if (gen_length(control_successors(pred)) <= 1) continue;
    switch (instruction_tag(statement_instruction(s))) {
    case is_instruction_test:
      /*fprintf(stderr, "TEST......\n");*/
      return false;
      break;
    case is_instruction_loop:
      statement_exec_appro = loop_executed_approximation(s);
      break;
    case is_instruction_whileloop:
      /*fprintf(stderr, "START.....\n");
      safe_print_statement(s1);
      safe_print_statement(s);
      statement_exec_appro = expression_approximation(s, whileloop_condition(instruction_whileloop(statement_instruction(s))));*/
      break;
    case is_instruction_call:
      /* consideration of parameters here */
      break;
    case is_instruction_unstructured:
    case is_instruction_block:
    case is_instruction_goto:
    default:
      break;
    }

    switch (statement_exec_appro) {
    case MUST_APPROXIMATION:
      break;
    case MAY_APPROXIMATION:
      statement_exec_appro1 = statement_exec_appro;
      break;
    case NOT_APPROXIMATION:
      return NOT_APPROXIMATION;
    }

  }, c1, blocs);
  gen_free_list(blocs);

  blocs = NIL;
  BACKWARD_CONTROL_MAP(pred, {
    s = control_statement(pred);
    if (gen_length(control_successors(pred)) <= 1) continue;
    switch (instruction_tag(statement_instruction(s))) {
    case is_instruction_test:
      /*fprintf(stderr, "TEST......\n");*/
      return false;
      break;
    case is_instruction_loop:
      statement_exec_appro = loop_executed_approximation(s);
      break;
    case is_instruction_whileloop:
      /*fprintf(stderr, "START.....\n");
      safe_print_statement(s2);
      safe_print_statement(s);
      statement_exec_appro = expression_approximation(s, whileloop_condition(instruction_whileloop(statement_instruction(s))));*/
      break;
    case is_instruction_call:
      /* consideration of parameters here */
      break;
    case is_instruction_unstructured:
    case is_instruction_block:
    case is_instruction_goto:
    default:
      break;
    }

    switch (statement_exec_appro) {
    case MUST_APPROXIMATION:
      break;
    case MAY_APPROXIMATION:
      statement_exec_appro2 = statement_exec_appro;
      break;
    case NOT_APPROXIMATION:
      return NOT_APPROXIMATION;
    }

  }, c2, blocs);
  gen_free_list(blocs);

  return ((statement_exec_appro1 ==  MUST_APPROXIMATION) && (statement_exec_appro2 == MUST_APPROXIMATION))
    ? MUST_APPROXIMATION : MAY_APPROXIMATION;
}

static void check_for_effected_statement(statement s, list le)
{
  set arcs_processed_set = set_make(set_pointer);
  list stat_reads1_old = gen_copy_seq(stat_reads1);
  list stat_writes1_old = gen_copy_seq(stat_writes1);
  list stat_reads2_old = gen_copy_seq(stat_reads2);
  list stat_writes2_old = gen_copy_seq(stat_writes2);

  MAP(EFFECT, eff, {
    entity ent_dest = reference_variable(effect_any_reference(eff));
    action act_dest = effect_action(eff);
    bool impact_must_p = false; /* default value of dependence : MAY */
    effect eff_tmp;

    if (entities_may_conflict_p(ent_dest, alias_ent1)) {
      if (action_read_p(act_dest)) {
	gen_free_list(stat_reads1);
	/* used for remember to rebuild the list of read statements for alias_ent1 */
	stat_reads1 = CONS(STATEMENT, s, NIL);
	MAP(STATEMENT, sw2, {
	  if (!statement_undefined_p(sw2)) { /* new flow-dependence created */
	    set_clear(arcs_processed_set);
	    if (!find_covering_reference_path(arcs_processed_set,
					      sw2, make_action_write_memory(), alias_ent2,
					      s, act_dest, alias_ent1))
	      {
		number_of_impact_alias++;
		/*switch (control_approximation_between_statement_p(sw2, s)) {*/
		switch(MUST_APPROXIMATION) {
		case MUST_APPROXIMATION:
		  eff_tmp = get_effect_write_of_statement_on_variable(sw2, alias_ent2);
		  if (!effect_undefined_p(eff_tmp))
		    impact_must_p = approximation_exact_p(effect_approximation(eff_tmp));
		  impact_must_p = impact_must_p &&
		    approximation_exact_p(effect_approximation(eff));
		  insert_impact_description_as_comment(sw2, s, impact_must_p, DEP_FLOW);
		  break;
		case MAY_APPROXIMATION:
		  insert_impact_description_as_comment(sw2, s, false, DEP_FLOW);
		  break;
		case NOT_APPROXIMATION:
		  /* dependence does not exist so no impact alias*/
		  break;
		}
	      }
	  }
	}, stat_writes2_old);
      } else {
	gen_free_list(stat_writes1);
	/* used for remember to rebuild the list of write statements for alias_ent1 */
	stat_writes1 = CONS(STATEMENT, s, NIL);
	MAP(STATEMENT, sr2, {
	  if (!statement_undefined_p(sr2)) { /* new anti-dependence created */
	    set_clear(arcs_processed_set);
	    if (!find_covering_reference_path(arcs_processed_set,
					      sr2, make_action_read_memory(), alias_ent2,
					      s, act_dest, alias_ent1))
	      {
		number_of_impact_alias++;
		/*switch (control_approximation_between_statement_p(sr2, s)) {*/
		switch(MUST_APPROXIMATION) {
		case MUST_APPROXIMATION:
		  eff_tmp = get_effect_read_of_statement_on_variable(sr2, alias_ent2);
		  if (!effect_undefined_p(eff_tmp))
		    impact_must_p = approximation_exact_p(effect_approximation(eff_tmp));
		  impact_must_p = impact_must_p &&
		    approximation_exact_p(effect_approximation(eff));
		  insert_impact_description_as_comment(sr2, s, impact_must_p, DEP_ANTI);
		  break;
		case MAY_APPROXIMATION:
		  insert_impact_description_as_comment(sr2, s, false, DEP_ANTI);
		  break;
		case NOT_APPROXIMATION:
		  /* dependence does not exist so don't the impact alias*/
		  break;
		}
	      }
	  }
	}, stat_reads2_old);
	MAP(STATEMENT, sw2, {
	  if (!statement_undefined_p(sw2)) { /* new output-dependence created */
	    set_clear(arcs_processed_set);
	    if (!find_covering_reference_path(arcs_processed_set,
					      sw2, make_action_write_memory(), alias_ent2,
					      s, act_dest, alias_ent1))
	      {
		number_of_impact_alias++;
		/*switch (control_approximation_between_statement_p(sw2, s)) {*/
		switch(MUST_APPROXIMATION) {
		case MUST_APPROXIMATION:
		  eff_tmp = get_effect_write_of_statement_on_variable(sw2, alias_ent2);
		  if (!effect_undefined_p(eff_tmp))
		    impact_must_p = approximation_exact_p(effect_approximation(eff_tmp));
		  impact_must_p = impact_must_p &&
		    approximation_exact_p(effect_approximation(eff));
		  insert_impact_description_as_comment(sw2, s, impact_must_p, DEP_OUTP);
		  break;
		case MAY_APPROXIMATION:
		  insert_impact_description_as_comment(sw2, s, false, DEP_OUTP);
		  break;
		case NOT_APPROXIMATION:
		  /* dependence does not exist so don't the impact alias*/
		  break;
		}
	      }
	  }
	}, stat_writes2_old);
      }
    }
    if (entities_may_conflict_p(ent_dest, alias_ent2)) {
      if (action_read_p(act_dest)) {
	gen_free_list(stat_reads2);
	/* rebuild the list of read statements for alias_ent2 */
	stat_reads2 = CONS(STATEMENT, s, NIL);
	MAP(STATEMENT, sw1, {
	  if (!statement_undefined_p(sw1)) { /* new flow-dependence created */
	    set_clear(arcs_processed_set);
	    if (!find_covering_reference_path(arcs_processed_set,
					      sw1, make_action_write_memory(), alias_ent1,
					      s, act_dest, alias_ent2))
	      {
		number_of_impact_alias++;
		/*switch (control_approximation_between_statement_p(sw1, s)) {*/
		switch(MUST_APPROXIMATION) {
		case MUST_APPROXIMATION:
		  eff_tmp = get_effect_write_of_statement_on_variable(sw1, alias_ent1);
		  if (!effect_undefined_p(eff_tmp))
		    impact_must_p = approximation_exact_p(effect_approximation(eff_tmp));
		  impact_must_p = impact_must_p &&
		    approximation_exact_p(effect_approximation(eff));
		  insert_impact_description_as_comment(sw1, s, impact_must_p, DEP_FLOW);
		  break;
		case MAY_APPROXIMATION:
		  insert_impact_description_as_comment(sw1, s, false, DEP_FLOW);
		  break;
		case NOT_APPROXIMATION:
		  /* dependence does not exist so don't the impact alias*/
		  break;
		}
	      }
	  }
	}, stat_writes1_old);
      } else {
	gen_free_list(stat_writes2);
	/* rebuild the list of write statements for alias_ent2 */
	stat_writes2 = CONS(STATEMENT, s, NIL);
	MAP(STATEMENT, sr1, {
	  if (!statement_undefined_p(sr1)) { /* new anti-dependence created */
	    set_clear(arcs_processed_set);
	    if (!find_covering_reference_path(arcs_processed_set,
					      sr1, make_action_read_memory(), alias_ent1,
					      s, act_dest, alias_ent2))
	      {
		number_of_impact_alias++;
		/*switch(control_approximation_between_statement_p(sr1, s)) {*/
		switch(MUST_APPROXIMATION) {
		case MUST_APPROXIMATION:
		  eff_tmp = get_effect_read_of_statement_on_variable(sr1, alias_ent1);
		  if (!effect_undefined_p(eff_tmp))
		  impact_must_p = approximation_exact_p(effect_approximation(eff_tmp));
		  impact_must_p = impact_must_p &&
		    approximation_exact_p(effect_approximation(eff));
		  insert_impact_description_as_comment(sr1, s, impact_must_p, DEP_ANTI);
		  break;
		case MAY_APPROXIMATION:
		  insert_impact_description_as_comment(sr1, s, false, DEP_ANTI);
		  break;
		case NOT_APPROXIMATION:
		  /* dependence does not exist so don't the impact alias*/
		  break;
		}
	      }
	  }
	}, stat_reads1_old);
	MAP(STATEMENT, sw1, {
	  if (!statement_undefined_p(sw1)) { /* new output-dependence created */
	    set_clear(arcs_processed_set);
	    if (!find_covering_reference_path(arcs_processed_set,
					      sw1, make_action_write_memory(), alias_ent1,
					      s, act_dest, alias_ent2))
	      {
		number_of_impact_alias++;
		/*switch (control_approximation_between_statement_p(sw1, s)) {*/
		switch(MUST_APPROXIMATION) {
		case MUST_APPROXIMATION:
		  eff_tmp = get_effect_write_of_statement_on_variable(sw1, alias_ent1);
		  if (!effect_undefined_p(eff_tmp))
		    impact_must_p = approximation_exact_p(effect_approximation(eff_tmp));
		  impact_must_p = impact_must_p &&
		    approximation_exact_p(effect_approximation(eff));
		  insert_impact_description_as_comment(sw1, s, impact_must_p, DEP_OUTP);
		  break;
		case MAY_APPROXIMATION:
		  insert_impact_description_as_comment(sw1, s, false, DEP_OUTP);
		  break;
		case NOT_APPROXIMATION:
		  /* dependence does not exist so don't the impact alias*/
		  break;
		}
	      }
	  }
	}, stat_writes1_old);
      }
    }
  }, le);
  set_free(arcs_processed_set);
  gen_free_list(stat_reads1_old);
  gen_free_list(stat_writes1_old);
  gen_free_list(stat_reads2_old);
  gen_free_list(stat_writes2_old);
  return;
}

static void check_new_arc_for_structured_statement(statement s)
{
    instruction istat;
    list /* of effects */ le = NIL;
    list /* of effects */ l = NIL;
    list blocs = NIL;

    if ((statement_undefined_p(s)) || (s == NULL)) return;

    /* precondition is used to determine if the statement is executed */
    /*if (!statement_feasible_p(s)) return;*/

    istat = statement_instruction(s);

    switch(instruction_tag(istat)) {
    case is_instruction_block:
        MAP(STATEMENT, s,
	    check_new_arc_for_structured_statement(s),
	    instruction_block(istat));
	break;
    case is_instruction_test:
      {
	list stat_reads1_true = NIL;
	list stat_reads2_true = NIL;
	list stat_writes1_true = NIL;
	list stat_writes2_true = NIL;

	list stat_reads1_old = NIL;
	list stat_reads2_old = NIL;
	list stat_writes1_old = NIL;
	list stat_writes2_old = NIL;

	le = proper_effects_of_expression(test_condition(instruction_test(istat)));
	check_for_effected_statement(s, le);

	/* save the old read, write statements */
        stat_reads1_old = gen_copy_seq(stat_reads1);
	stat_reads2_old = gen_copy_seq(stat_reads2);
	stat_writes1_old = gen_copy_seq(stat_writes1);
	stat_writes2_old = gen_copy_seq(stat_writes2);

	/* read, write statements may be modified after the line below */
        check_new_arc_for_structured_statement(test_true(instruction_test(istat)));

	/* store the new version */
	stat_reads1_true = stat_reads1;
	stat_reads2_true = stat_reads2;
	stat_writes1_true = stat_writes1;
	stat_writes2_true = stat_writes2;

	/* restore the old version */
	stat_reads1 = stat_reads1_old;
	stat_reads2 = stat_reads2_old;
	stat_writes1 = stat_writes1_old;
	stat_writes2 = stat_writes2_old;

	stat_reads1_old = NIL;
	stat_reads2_old = NIL;
	stat_writes1_old = NIL;
	stat_writes2_old = NIL;

	/* read, write statements may be modified after the line below */
	check_new_arc_for_structured_statement(test_false(instruction_test(istat)));

	stat_reads1 = union_list(stat_reads1, stat_reads1_true);
	stat_reads2 = union_list(stat_reads2, stat_reads2_true);
	stat_writes1 = union_list(stat_writes1, stat_writes1_true);
        stat_writes2 = union_list(stat_writes2, stat_writes2_true);

	gen_free_list(stat_reads1_true);
	gen_free_list(stat_reads2_true);
	gen_free_list(stat_writes1_true);
	gen_free_list(stat_writes2_true);

	break;
      }
    case is_instruction_loop:
      /*safe_print_statement(s);
	switch (loop_executed_approximation(s)) {
	case MUST_BE_EXECUTED:
	  fprintf(stderr, "MUST_BE_EXECUTED\n");
	  break;
	case MAY_BE_EXECUTED:
	  fprintf(stderr, "MAY_BE_EXECUTED\n");
	  break;
	case MUST_NOT_BE_EXECUTED:
	  fprintf(stderr, "MUST_NOT_BE_EXECUTED\n");
	  break;
	  }*/
        le = proper_effects_of_expression(range_lower(loop_range(instruction_loop(istat))));
	l = proper_effects_of_expression(range_upper(loop_range(instruction_loop(istat))));
	le = union_list(le, l);
	l = proper_effects_of_expression(range_increment(loop_range(instruction_loop(istat))));
	le = union_list(le, l);
	check_for_effected_statement(s, le);
        check_new_arc_for_structured_statement(loop_body(instruction_loop(istat)));
	break;
    case is_instruction_whileloop:
        le = proper_effects_of_expression(whileloop_condition(instruction_whileloop(istat)));
	check_for_effected_statement(s, le);
        check_new_arc_for_structured_statement(whileloop_body(instruction_whileloop(istat)));
	break;
    case is_instruction_goto:
        check_new_arc_for_structured_statement(instruction_goto(istat));
        break;
    case is_instruction_unstructured:
	CONTROL_MAP(c, {
	  fprintf(stderr, "i am here UNSTRUCTURED ");
	  safe_print_statement(control_statement(c));
	  check_new_arc_for_structured_statement(control_statement(c));
	}, unstructured_control(instruction_unstructured(istat)), blocs);
	gen_free_list(blocs);
	break;
    case is_instruction_call:
        /* consideration of parameters here */
        check_for_effected_statement(s, statement_to_effects(s));
	break;
    default:
	pips_internal_error("case default reached");
        break;
    }
}

static void impact_check_two_scalar_variables_in_path(entity e1, entity e2, expression off1, expression off2, list path)
{
    expression diff = eq_expression(off1, off2);
    clean_all_normalized(diff);
    ifdebug(3) {
        fprintf(stderr, "entity1 local name: %s\n", entity_local_name(e1));
	fprintf(stderr, "entity2 local name: %s\n",  entity_local_name(e2));
	print_expression(off1);
	print_expression(off2);
	fprintf(stderr, "call path: %s\n", print_call_path(path));
    }
    if (trivial_expression_p(diff) == -1)
        /* off1 != off2 => Okay, no alias between these 2 variables */
        return;
    else {
        /* alias */

        alias_ent1 = e1;
	alias_ent2 = e2;
	current_path = path;

        check_new_arc_for_structured_statement(mod_stat);
	/*gen_recurse(mod_stat, statement_domain, check_new_arc_for_structured_statement, gen_null);*/

	gen_free_list(stat_reads1);
	gen_free_list(stat_reads2);
	gen_free_list(stat_writes1);
	gen_free_list(stat_writes2);
	stat_reads1 = NIL;
	stat_reads2 = NIL;
	stat_writes1 = NIL;
	stat_writes2 = NIL;

	alias_ent1 = entity_undefined;
	alias_ent2 = entity_undefined;
	current_path = NIL;
    }
    return;
}

static void impact_check_in_path(entity e1, entity e2, expression off1, expression off2, list path)
{
    if (entity_atomic_reference_p(e1) && entity_atomic_reference_p(e2))
        impact_check_two_scalar_variables_in_path(e1, e2, off1, off2, path);
    if (entity_atomic_reference_p(e1) && !entity_atomic_reference_p(e2))
      {
        fprintf(stderr, "alias entre variable scalaire e1 et variable tableau e2\n");
	impact_check_two_scalar_variables_in_path(e1, e2, off1, off2, path);
      }
    if (!entity_atomic_reference_p(e1) && entity_atomic_reference_p(e2))
      {
        fprintf(stderr, "alias entre variable tableau e1 et variable scalaire e2\n");
	impact_check_two_scalar_variables_in_path(e1, e2, off1, off2, path);
      }
    if (!entity_atomic_reference_p(e1) && !entity_atomic_reference_p(e2))
      {
	fprintf(stderr, "alias entre 2 variables tableau\n");
	impact_check_two_scalar_variables_in_path(e1, e2, off1, off2, path);
      }
}

static bool written = false;
static entity current_entity  = entity_undefined;

/* This function returns true if the variable is written directly in the current module,
   or by its callees */
static bool variable_is_written_by_statement_flt(statement s)
{
    if (statement_call_p(s)) {
        list l_rw = statement_to_effects(s);
	MAP(EFFECT, eff, {
	    action a = effect_action(eff);
	    if (action_write_p(a)) {
	        reference r = effect_any_reference(eff);
		entity e = reference_variable(r);
		if (same_entity_p(e,current_entity)) {
		    ifdebug(3) {
		        fprintf(stderr,"\n Write on entity %s :\n",entity_name(e));
			fprintf(stderr,"\n Current entity %s :\n",entity_name(current_entity));
		    }
		    written = true;
		    /* gen_recurse_stop(NULL); */
		    return false;
		}
	    }
	}, l_rw);
	return false;
    }
    return true;
}

static bool variable_is_written_p(entity ent)
{
    written = false;
    current_entity = ent;
    gen_recurse(mod_stat,statement_domain,
		variable_is_written_by_statement_flt,gen_null);
    current_entity = entity_undefined;
    return written;
}

static void set_dynamic_checked(entity e1, entity e2)
{
  MAP(DYNAMIC_CHECK,dc,
  {
    if ((dynamic_check_first(dc)==e1) && (dynamic_check_second(dc)==e2))
      dynamic_check_checked(dc) = true;
  }, l_dynamic_check);
}

static bool dynamic_checked_p(entity e1, entity e2)
{
    MAP(DYNAMIC_CHECK, dc, {
        if ((dynamic_check_first(dc)==e1)&&(dynamic_check_second(dc)==e2))
	    return dynamic_check_checked(dc);
    }, l_dynamic_check);
    return false;
}

static void init_dynamic_check_list(entity current_mod)
{
    list l_decls = code_declarations(entity_code(current_mod));
    list l_formals = NIL;
    list l_commons = NIL;

    /* search for formal parameters in the declaration list */
    MAP(ENTITY, e, {
        if (formal_parameter_p(e))
	    l_formals = gen_nconc(l_formals,CONS(ENTITY,e,NIL));
    }, l_decls);

    MAP(ENTITY, e, {
        if (variable_in_common_p(e))
	    l_commons = gen_nconc(l_commons,CONS(ENTITY,e,NIL));
    }, l_decls);

    MAP(ENTITY,e1, {
        MAP(ENTITY,e2, {
	    dynamic_check dc = make_dynamic_check(e1,e2,false);
	    l_dynamic_check = gen_nconc(l_dynamic_check,CONS(DYNAMIC_CHECK,dc,NIL));
	}, l_formals);

	MAP(ENTITY,e2, {
	    dynamic_check dc = make_dynamic_check(e1,e2,false);
	    l_dynamic_check = gen_nconc(l_dynamic_check,CONS(DYNAMIC_CHECK,dc,NIL));
	}, l_commons);
    }, l_formals);
}

static void impact_check_two_variables(entity e1, entity e2, expression off1, expression off2, list path)
{
    if (variable_is_written_p(e1) || variable_is_written_p(e2)) {
        if (!expression_undefined_p(off1) && !expression_undefined_p(off2)) {
	    /* good offset --> check */
	    impact_check_in_path(e1, e2, off1, off2, path);
	} else {
	    /* As we do not have exact offsets of variables, we have to go to the
	       caller's frame to check for alias impact. The direct caller is
	       CAR(call_path) because of the following concatenation in alias_propagation:
	       path = CONS(CALL_SITE,cs,gen_full_copy_list(alias_association_call_chain(aa)));

	       To find a call site from its ordering, we have to do a gen_recurse
	       in the caller module. */
	    call_site cs = CALL_SITE(CAR(path));
	    statement caller_statement;
	    entity current_caller = call_site_function(cs);
	    caller_name = module_local_name(current_caller);
	    caller_statement = (statement)db_get_memory_resource(DBR_CODE,caller_name,true);

	    statement_in_caller_ordering = call_site_ordering(cs);
	    statement_in_caller = statement_undefined;
	    gen_recurse(caller_statement,statement_domain,
			search_statement_by_ordering_flt,gen_null);

	    if (!statement_undefined_p(statement_in_caller)
		&& statement_call_p(statement_in_caller)) {
	        expression new_off1, new_off2;
		current_call = statement_call(statement_in_caller);
		new_off1 = offset_in_caller(e1, cs, path);
		new_off2 = offset_in_caller(e2, cs, path);
		if (!expression_undefined_p(new_off1) && !expression_undefined_p(new_off2)) {
		    /* good offset --> check */
		    impact_check_in_path(e1, e2, new_off1, new_off2, path);
		} else {
		  /* Try with special cases : CALL FOO(R(TR(K)),R(TR(K))) ???????
		     Does this case exist when we create special section + offset
		     for same actual arguments ??? */
		  /* use dynamic alias check*/
		  set_dynamic_checked(e1, e2);
		}
		current_call = call_undefined;
	    } else
	        pips_user_warning("Problem with statement ordering *\n");
	    statement_in_caller = statement_undefined;
	    statement_in_caller_ordering = 0;
	}
    }
}

bool impact_check(char * module_name)
{
    list l_module_aliases = NIL;
    /*hash_table control_to_set_of_dominators = hash_table_make(hash_pointer, 0);*/

    number_of_processed_modules++;
    current_mod = local_name_to_top_level_entity(module_name);

    debug_on("RICEDG_DEBUG_LEVEL");


    l_module_aliases = alias_associations_list((alias_associations)
					       db_get_memory_resource(DBR_ALIAS_ASSOCIATIONS,
								      module_name, true));

    if (l_module_aliases != NIL) {
        dg = (graph) db_get_memory_resource(DBR_DG, module_name, true);
	/*full_control_graph(module_name);*/
	mod_stat = (statement)db_get_memory_resource(DBR_CODE, module_name, true);
        set_current_module_entity(current_mod);

	set_ordering_to_statement(mod_stat);
	init_dynamic_check_list(current_mod);

	/*computing_dominators(control_to_set_of_dominators, load_ctrl_graph(mod_stat));

	set_precondition_map((statement_mapping)
			     db_get_memory_resource(DBR_PRECONDITIONS,
						    module_name,
						    true));*/
	while (!ENDP(l_module_aliases)) {
	    alias_association aa1 = ALIAS_ASSOCIATION(CAR(l_module_aliases));
	    entity e1 = alias_association_variable(aa1);
	    entity sec1 = alias_association_section(aa1);
	    list path1 = alias_association_call_chain(aa1);
	    expression off1 = alias_association_offset(aa1);
	    int l1 = alias_association_lower_offset(aa1);
	    int u1 = alias_association_upper_offset(aa1);
	    l_module_aliases = CDR(l_module_aliases);

	    /* Looking for another formal variable in the list of alias
	       associations that has same section and included call path.
	       If this variable is checked dynamically with e1 => no need
	       to continue */
	    MAP(ALIAS_ASSOCIATION, aa2, {
	        entity e2 = alias_association_variable(aa2);
		entity sec2 = alias_association_section(aa2);
		list path2 = alias_association_call_chain(aa2);
		if (!same_entity_p(e1,e2) && same_entity_p(sec1,sec2) &&
		    !dynamic_checked_p(e1, e2) && included_call_chain_p(path1,path2)) {

		    int l2 = alias_association_lower_offset(aa2);
		    int u2 = alias_association_upper_offset(aa2);

		    if (((u1==-1)||(u1>=l2))&&((u2==-1)||(u2>=l1))) {
		        expression off2 = alias_association_offset(aa2);
			if (gen_length(path1) < gen_length(path2))
			    impact_check_two_variables(e1, e2, off1, off2, path2);
			else
			    impact_check_two_variables(e1, e2, off1, off2, path1);
		    }
		}
	    }, l_module_aliases);

	    /* Looking for common variables in module or callee of modules
	       to check for alias impact ... */
	    MAP(ENTITY, e2, {
	        if (variable_in_common_p(e2)) {
		    ram ra = storage_ram(entity_storage(e2));
		    entity sec2 = ram_section(ra);
		    if (!dynamic_checked_p(e1, e2) && same_entity_p(sec1,sec2)) {
		        /* formal parameter has a same section with other common variable */
		        int l2 = ram_offset(ra);
			int u2 = l2;
			if (array_entity_p(e2))
			  {
			    int tmp;
			    if (SizeOfArray(e2, &tmp))
			      u2 = tmp - SizeOfElements(variable_basic(type_variable(entity_type(e2)))) + l2;
			    else
			      user_log("Varying size of common variable");
			  }
			/* If u1 is defined (different to -1) and u1<l2, there is no alias impact
			   The same for: l1 is defined (different to -1) and u2<l1 */
			if (((u1==-1)||(u1>=l2)) && (u2>=l1)) {
			    expression off2 = int_to_expression(l2);
			    /* The common variable always have a good offset off2 */
			    impact_check_two_variables(e1, e2, off1, off2, path1);
			}
		    }
		}
	    }, code_declarations(entity_code(current_mod)));
	}
	l_dynamic_check = NIL;

	DB_PUT_MEMORY_RESOURCE(DBR_CODE,module_name,mod_stat);

	/*hash_table_free(control_to_set_of_dominators);*/
	/*clean_ctrl_graph();*/
	reset_current_module_entity();
	reset_ordering_to_statement();
	/*reset_precondition_map();*/
	mod_stat = statement_undefined;
    }
    display_impact_alias_statistics();
    l_module_aliases = NIL;
    current_mod = entity_undefined;
    return true;
}
