#include "local.h"
#include "icfg.h"

#define DEP_FLOW 1
#define DEP_ANTI 2
#define DEP_OUTP 3

static statement mod_stat = statement_undefined;
static entity current_mod = entity_undefined;
static graph dg = NULL; /* data dependance graph */
static entity alias_ent1 = entity_undefined;
static entity alias_ent2 = entity_undefined;
static list stat_reads1 = NIL; /* list of statements */
static list stat_writes1 = NIL; /* list of statements */
static list stat_reads2 = NIL; /* list of statements */
static list stat_writes2 = NIL; /* list of statements */
static list current_path = NIL;
static int statement_in_caller_ordering = 0;
static statement statement_in_caller = statement_undefined;
static string caller_name;
static call current_call = call_undefined;

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
  return FALSE;
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
        db_get_memory_resource(DBR_ALIAS_ASSOCIATIONS,caller_name,TRUE)); 
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
	return FALSE;
    }
    return TRUE;
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
	        return TRUE;
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
    return FALSE;
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
    pc = CHAIN_SWORD(pc,int_to_string(ORDERING_NUMBER(casiord)));
    pc = CHAIN_SWORD(pc,",");
    pc = CHAIN_SWORD(pc,int_to_string(ORDERING_STATEMENT(casiord)));
    pc = CHAIN_SWORD(pc,")) ");
  },path);
  return words_to_string(pc);
}

static void print_impact_description(statement s1, statement s2, int dep_type)
{
    fprintf(stderr, "Impact alias at the call path %s, between %s and %s\n",
	    print_call_path(current_path), entity_local_name(alias_ent1), entity_local_name(alias_ent2));
    switch(dep_type) {
    case DEP_FLOW:
        fprintf(stderr, "New flow-dependance between\n");
	break;
    case DEP_ANTI:
        fprintf(stderr, "New anti-dependance between\n");
	break;
    case DEP_OUTP:
        fprintf(stderr, "New output-dependance between\n");
	break;
    }
    dump_text(statement_to_text(s1));
    fprintf(stderr, "and\n");
    dump_text(statement_to_text(s2));
    return;
}

list union_list(list l1, list l2) {
    MAP(STATEMENT, s, {
        if (!gen_in_list_p(s, l1))
	  ADD_ELEMENT_TO_LIST(l1, STATEMENT, s);
    }, l2);
    return l1;
}

static bool check_new_arc_for_unstructured_statement(statement s)
{
    instruction i = statement_instruction(s);
    if (instruction_block_p(i) || instruction_sequence_p(i) || instruction_test_p(i)
	|| instruction_whileloop_p(i) || instruction_loop_p(i))
      return TRUE;
    
    MAP(EFFECT, eff, {
        entity e = reference_variable(effect_reference(eff));
	action act = effect_action(eff);

	if (entity_conflict_p(e, alias_ent1)) {
	    if (action_read_p(act)) {
	        gen_free_list(stat_reads1);
		stat_reads1 = CONS(STATEMENT, s, NIL);
		MAP(STATEMENT, sw2, {
		    if (!statement_undefined_p(sw2)) { /* new flow-dependance created */
		        if (!check_way_between_two_statements(sw2, s, dg))
			    print_impact_description(sw2, s, DEP_FLOW);
		    }
		}, stat_writes2);
	    } else {
	        gen_free_list(stat_writes1);
		stat_writes1 = CONS(STATEMENT, s, NIL);
		MAP(STATEMENT, sr2, {
		    if (!statement_undefined_p(sr2)) { /* new anti-dependance created */
		        if (!check_way_between_two_statements(sr2, s, dg))
			    print_impact_description(sr2, s, DEP_ANTI);
		    }
		}, stat_reads2);
		MAP(STATEMENT, sw2, {
		    if (!statement_undefined_p(sw2)) { /* new output-dependance created */
		        if (!check_way_between_two_statements(sw2, s, dg))
			    print_impact_description(sw2, s, DEP_OUTP);
		    }
		}, stat_writes2);
	    }
	}
	if (entity_conflict_p(e, alias_ent2)) {
	    if (action_read_p(act)) {
	        gen_free_list(stat_reads2);
		stat_reads2 = CONS(STATEMENT, s, NIL);
		MAP(STATEMENT, sw1, {
		    if (!statement_undefined_p(sw1)) { /* new flow-dependance created */
		        if (!check_way_between_two_statements(sw1, s, dg))
			    print_impact_description(sw1, s, DEP_FLOW);
		    }
		}, stat_writes1);
	    } else {
	        gen_free_list(stat_writes2);
		stat_writes2 = CONS(STATEMENT, s, NIL);
		MAP(STATEMENT, sr1, {
		    if (!statement_undefined_p(sr1)) { /* new anti-dependance created */
		        if (!check_way_between_two_statements(sr1, s, dg))
			    print_impact_description(sr1, s, DEP_ANTI);
		    }
		}, stat_reads1);
		MAP(STATEMENT, sw1, {
		    if (!statement_undefined_p(sw1)) { /* new output-dependance created */
		        if (!check_way_between_two_statements(sw1, s, dg))
			    print_impact_description(sw1, s, DEP_OUTP);
		    }
		}, stat_writes1);
	    }
	}
    }, statement_to_effects(s));
    return TRUE;
}

static void check_new_arc_for_structured_statement(statement s)
{
    instruction istat;
    if ((statement_undefined_p(s)) || (s == NULL)) return;
    istat = statement_instruction(s);

    switch(instruction_tag(istat)) {
    case is_instruction_block:
        MAP(STATEMENT, s, check_new_arc_for_structured_statement(s), instruction_block(istat));
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

	/* save the old read, write statements */
        stat_reads1_old = gen_full_copy_list(stat_reads1);
	stat_reads2_old = gen_full_copy_list(stat_reads2);
	stat_writes1_old = gen_full_copy_list(stat_writes1);
	stat_writes2_old = gen_full_copy_list(stat_writes2);

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
        check_new_arc_for_structured_statement(loop_body(instruction_loop(istat)));
	break;
    case is_instruction_whileloop:
    case is_instruction_goto:
        break;
    case is_instruction_unstructured:
        check_new_arc_for_structured_statement(instruction_unstructured(istat));
	break;
    case is_instruction_call:
        MAP(EFFECT, eff, {
	    entity e = reference_variable(effect_reference(eff));
	    action act = effect_action(eff);

	    if (entity_conflict_p(e, alias_ent1)) {
	        if (action_read_p(act)) {
		    gen_free_list(stat_reads1);
		    stat_reads1 = CONS(STATEMENT, s, NIL);
		    MAP(STATEMENT, sw2, {
		        if (!statement_undefined_p(sw2)) { /* new flow-dependance created */
			    if (!check_way_between_two_statements(sw2, s, dg))
			        print_impact_description(sw2, s, DEP_FLOW);
			}
		    }, stat_writes2);
		} else {
		    gen_free_list(stat_writes1);
		    stat_writes1 = CONS(STATEMENT, s, NIL);
		    MAP(STATEMENT, sr2, {
		        if (!statement_undefined_p(sr2)) { /* new anti-dependance created */
			    if (!check_way_between_two_statements(sr2, s, dg))
			        print_impact_description(sr2, s, DEP_ANTI);
			}
		    }, stat_reads2);
		    MAP(STATEMENT, sw2, {
		        if (!statement_undefined_p(sw2)) { /* new output-dependance created */
			    if (!check_way_between_two_statements(sw2, s, dg))
			        print_impact_description(sw2, s, DEP_OUTP);
			}
		    }, stat_writes2);
		}
	    }
	    if (entity_conflict_p(e, alias_ent2)) {
	        if (action_read_p(act)) {
		    gen_free_list(stat_reads2);
		    stat_reads2 = CONS(STATEMENT, s, NIL);
		    MAP(STATEMENT, sw1, {
		        if (!statement_undefined_p(sw1)) { /* new flow-dependance created */
			    if (!check_way_between_two_statements(sw1, s, dg))
			        print_impact_description(sw1, s, DEP_FLOW);
			}
		    }, stat_writes1);
		} else {
		    gen_free_list(stat_writes2);
		    stat_writes2 = CONS(STATEMENT, s, NIL);
		    MAP(STATEMENT, sr1, {
		        if (!statement_undefined_p(sr1)) { /* new anti-dependance created */
			    if (!check_way_between_two_statements(sr1, s, dg))
			        print_impact_description(sr1, s, DEP_ANTI);
			}
		    }, stat_reads1);
		    MAP(STATEMENT, sw1, {
		        if (!statement_undefined_p(sw1)) { /* new output-dependance created */
			    if (!check_way_between_two_statements(sw1, s, dg))
			        print_impact_description(sw1, s, DEP_OUTP);
			}
		    }, stat_writes1);
		}
	    }
	}, statement_to_effects(s));
	break;
    default:
	/*pips_error("check_new_arc_statement", "case default reached\n");*/
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
	/*gen_recurse(mod_stat, statement_domain, check_new_arc_for_unstructured_statement, gen_null);*/

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
    if (entity_scalar_p(e1) && entity_scalar_p(e2))
        impact_check_two_scalar_variables_in_path(e1, e2, off1, off2, path);
}

static bool written = FALSE;
static entity current_entity  = entity_undefined;

/* This function returns TRUE if the variable is written directly in the current module, 
   or by its callees */
static bool variable_is_written_by_statement_flt(statement s)
{
    if (statement_call_p(s)) {
        list l_rw = statement_to_effects(s);
	MAP(EFFECT, eff, {
	    action a = effect_action(eff);
	    if (action_write_p(a)) {
	        reference r = effect_reference(eff);
		entity e = reference_variable(r);
		if (same_entity_p(e,current_entity)) {
		    ifdebug(3) {
		        fprintf(stderr,"\n Write on entity %s :\n",entity_name(e));
			fprintf(stderr,"\n Current entity %s :\n",entity_name(current_entity));
		    }
		    written = TRUE;
		    /* gen_recurse_stop(NULL); */
		    return FALSE;
		}
	    }
	}, l_rw); 
	return FALSE;
    }
    return TRUE; 
}

static bool variable_is_written_p(entity ent)
{
    written = FALSE;
    current_entity = ent;
    gen_recurse(mod_stat,statement_domain,
		variable_is_written_by_statement_flt,gen_null);
    current_entity = entity_undefined;
    return written;
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
	    caller_statement = (statement)db_get_memory_resource(DBR_CODE,caller_name,TRUE);

	    statement_in_caller_ordering = call_site_ordering(cs);
	    statement_in_caller = statement_undefined;
	    gen_recurse(caller_statement,statement_domain,
			search_statement_by_ordering_flt,gen_null);
	    
	    if (!statement_undefined_p(statement_in_caller) && statement_call_p(statement_in_caller)) {
	        expression new_off1, new_off2;
		current_call = statement_call(statement_in_caller);
		new_off1 = offset_in_caller(e1, cs, path);
		new_off2 = offset_in_caller(e2, cs, path);
		if (!expression_undefined_p(new_off1) && !expression_undefined_p(new_off2)) {
		    /* good offset --> check */
		    impact_check_in_path(e1, e2, new_off1, new_off2, path);
		} else {
		  /* to be written */
		}
		current_call = call_undefined;
	    } else
	        pips_user_warning("Problem with statement ordering *\n");
	    statement_in_caller = statement_undefined;
	    statement_in_caller_ordering = 0;
	}
    }
}

static bool dynamic_checked_p(entity e1, entity e2, list l_dynamic_check)
{
    MAP(DYNAMIC_CHECK, dc, { 
        if ((dynamic_check_first(dc)==e1)&&(dynamic_check_second(dc)==e2))
	    return dynamic_check_checked(dc);
    }, l_dynamic_check);
    return FALSE;
}

static list init_dynamic_check_list(entity current_mod)
{
    list l_decls = code_declarations(entity_code(current_mod)); 
    list l_formals = NIL;
    list l_commons = NIL;
    list l_dynamic_check = NIL;

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
	    dynamic_check dc = make_dynamic_check(e1,e2,FALSE);
	    l_dynamic_check = gen_nconc(l_dynamic_check,CONS(DYNAMIC_CHECK,dc,NIL));
	}, l_formals);

	MAP(ENTITY,e2, {
	    dynamic_check dc = make_dynamic_check(e1,e2,FALSE);
	    l_dynamic_check = gen_nconc(l_dynamic_check,CONS(DYNAMIC_CHECK,dc,NIL));
	}, l_commons);
    }, l_formals);

    return l_dynamic_check;
}

bool impact_check(char * module_name)
{
    list l_module_aliases = NIL;
    list l_dynamic_check = NIL;
    current_mod = local_name_to_top_level_entity(module_name);

    debug_on("RICEDG_DEBUG_LEVEL");

    dg = (graph) db_get_memory_resource(DBR_DG, module_name, TRUE);
    l_module_aliases = alias_associations_list((alias_associations)
					       db_get_memory_resource(DBR_ALIAS_ASSOCIATIONS, module_name, TRUE));
    if (l_module_aliases != NIL) {
	mod_stat = (statement)db_get_memory_resource(DBR_CODE, module_name, TRUE);
        set_current_module_entity(current_mod);

	initialize_ordering_to_statement(mod_stat);
	l_dynamic_check = init_dynamic_check_list(current_mod);

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
		    !dynamic_checked_p(e1, e2, l_dynamic_check) && included_call_chain_p(path1,path2)) {  

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
		    if (!dynamic_checked_p(e1, e2, l_dynamic_check) && same_entity_p(sec1,sec2)) {
		        /* formal parameter has a same section with other common variable */
		        int l2 = ram_offset(ra);
			int u2 = l2;
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
    
	reset_current_module_entity();
	reset_ordering_to_statement();
	mod_stat = statement_undefined;
    }

    return TRUE; 
}



















