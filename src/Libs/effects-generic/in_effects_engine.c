/* package generic effects :  Be'atrice Creusillet 5/97
 *
 * File: in_effects_engine.c
 * ~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 * This File contains the generic functions necessary for the computation of 
 * all types of in effects.
 *
 */
#include <stdio.h>
#include <string.h>

#include "genC.h"

#include "linear.h"
#include "ri.h"
#include "database.h"

#include "ri-util.h"
#include "control.h"
#include "constants.h"
#include "misc.h"
#include "text-util.h"
#include "text.h"
#include "makefile.h"

#include "properties.h"
#include "pipsmake.h"

#include "transformer.h"
#include "semantics.h"
#include "pipsdbm.h"
#include "resources.h"


#include "effects-generic.h"


/* ============================================================================= 
 *
 * GENERIC INTERPROCEDURAL IN EFFECTS ANALYSIS
 *
 * =============================================================================
 */


/* bool summary_in_effects_engine(char *module_name) 
 * input    : the name of the current module.
 * output   : the list of summary in effects
 * modifies : nothing.
 * comment  : computes the summary in effects of the current module, using the
 *            in effects of its embedding statement.	
 */
bool
summary_in_effects_engine(char *module_name)
{

    list l_glob = NIL, l_loc = NIL; 
    statement module_stat;

    set_current_module_entity(local_name_to_top_level_entity(module_name)); 
    set_current_module_statement( (statement)
	db_get_memory_resource(DBR_CODE, module_name, TRUE) );
    module_stat = get_current_module_statement();

    (*effects_computation_init_func)(module_name);

    set_in_effects((*db_get_in_effects_func)(module_name));

    l_loc = load_in_effects_list(module_stat);
    l_glob = (*effects_local_to_global_translation_op)(l_loc);
    
    (*db_put_summary_in_effects_func)(module_name, l_glob);

    reset_current_module_entity();
    reset_current_module_statement();
    reset_in_effects();

    (*effects_computation_reset_func)(module_name);

    return TRUE;
}


/***************************** GENERIC INTRAPROCEDURAL IN EFFECTS ANALYSIS */

/*
typedef void (*void_fun)();
static void list_gen_consistent_p(list l)
{
    pips_debug(1, "length is %d\n", gen_length(l));
    gen_map((void_fun)gen_consistent_p, l);
    gen_consistent_p(get_in_effects());
}
*/
#define debug_consistent(l) /* ifdebug(9) list_gen_consistent_p(l) */

static bool 
in_effects_stmt_filter(statement s)
{
    pips_debug(1, "Entering statement %03d :\n", statement_ordering(s));
    effects_private_current_stmt_push(s);
    return TRUE;
}

static void
in_effects_of_statement(statement s)
{
    store_invariant_in_effects_list(s, NIL);
    debug_consistent(NIL);
    effects_private_current_stmt_pop();
    pips_debug(1, "End statement %03d :\n", statement_ordering(s));
}

static list 
r_in_effects_of_sequence(list l_inst)
{
    statement first_statement;
    list remaining_block = NIL;
    
    list s1_lin; /* in effects of first statement */
    list rb_lin; /* in effects of remaining block */
    list l_in = NIL; /* resulting in effects */
    list s1_lr; /* rw effects of first statement */
    transformer t1; /* transformer of first statement */
 
    first_statement = STATEMENT(CAR(l_inst));
    remaining_block = CDR(l_inst);
	    
    s1_lin = effects_dup(load_in_effects_list(first_statement));
	
    /* Is it the last instruction of the block */
    if (!ENDP(remaining_block))
    {
	
	s1_lr = load_rw_effects_list(first_statement);
	ifdebug(6) 
	    {
		pips_debug(6," rw effects for first statement:\n");
		(*effects_prettyprint_func)(s1_lr);
	    }

	t1 = (*load_transformer_func)(first_statement);    
	rb_lin = r_in_effects_of_sequence(remaining_block);
	    
	(*effects_transformer_composition_op)(rb_lin, t1); 
	    
	/* IN(block) = (IN(rest_of_block) - W(S1)) U IN(S1) */
	l_in = (*effects_union_op)(
	    s1_lin,
	    (*effects_sup_difference_op)(rb_lin, effects_dup(s1_lr),
					 r_w_combinable_p),
	    effects_same_action_p);
    }	
    else 
    {
	l_in = s1_lin;
    }
    
    ifdebug(6) 
    {
	pips_debug(6,"cumulated_in_effects:\n");
	(*effects_prettyprint_func)(l_in);
    }

    store_cumulated_in_effects_list(first_statement, effects_dup(l_in));
    debug_consistent(l_in);
    return l_in;
}

static void
in_effects_of_sequence(sequence block)
{
    list l_in = NIL;
    statement current_stat = effects_private_current_stmt_head();
    list l_inst = sequence_statements(block);

    pips_debug(2, "Effects for statement %03d:\n",
	       statement_ordering(current_stat)); 

    if (ENDP(l_inst))
    {
	if (get_bool_property("WARN_ABOUT_EMPTY_SEQUENCES"))
	    pips_user_warning("empty sequence\n");
    }
    else
    {
	l_in = r_in_effects_of_sequence(l_inst);
    }

    ifdebug(2)
	{
	    pips_debug(2, "IN effects for statement%03d:\n",
		       statement_ordering(current_stat));  
	    (*effects_prettyprint_func)(l_in);
	    pips_debug(2, "end\n");
	}

    store_in_effects_list(current_stat, l_in);    
    debug_consistent(l_in);
}



static void
in_effects_of_test(test t)
{
    statement current_stat = effects_private_current_stmt_head();
    list lt, lf, lc_in;
    list l_in = NIL;

    pips_debug(2, "Effects for statement %03d:\n",
	       statement_ordering(current_stat)); 

      /* IN effects of the true branch */
    lt = effects_dup(load_in_effects_list(test_true(t))); /* FC: dup */
      /* IN effects of the false branch */
    lf = effects_dup(load_in_effects_list(test_false(t))); /* FC: dup */

      /* IN effects of the combination of both */
    l_in = (*effects_test_union_op)(lt, lf, effects_same_action_p);

    /* IN effects of the condition */
    /* they are equal to the proper effects of the statement if there are
     * no side-effects. */
    lc_in = effects_dup(load_proper_rw_effects_list(current_stat));

    /* in regions of the test */
    l_in = (*effects_union_op)(l_in, lc_in, effects_same_action_p);

    ifdebug(2)
    {
	pips_debug(2, "IN effects for statement %03d:\n",
		   statement_ordering(current_stat));  
	(*effects_prettyprint_func)(l_in);
	pips_debug(2, "end\n");
    }

    store_in_effects_list(current_stat, l_in);      
    debug_consistent(l_in);
}

/* list in_effects_of_loop(loop l)
 * input    : a loop, its transformer and its context.
 * output   : the corresponding list of in regions.
 * modifies : in_regions_map.
 * comment  : IN(loop) = proj[i] (proj[i'] (IN(i) - W(i', i'<i))) U IN(i=1))
 */
static void
in_effects_of_loop(loop l)
{
    statement current_stat = effects_private_current_stmt_head();
    
    range r;
    statement b;
    entity i, i_prime, new_i;
    
    list lbody_in; /* in regions of the loop body */
    list global_in, global_in_read_only;/* in regions of non local variables */
    list global_write; /* invariant write regions of non local variables */
    list l_prop, l_prop_read, l_prop_write; /* proper effects of header */
    list l_in = NIL;
    transformer loop_trans = transformer_undefined;

    pips_debug(1, "begin\n");
    
    i = loop_index(l);
    r = loop_range(l);
    b = loop_body(l);

    pips_debug(1, "loop index %s.\n", entity_minimal_name(i));

    /* IN EFFECTS OF HEADER */
    /* We assume that there is no side effect in the loop header;
     * thus the IN effects of the header are similar to its proper effects.
     */
    l_prop = load_proper_rw_effects_list(current_stat);
    l_prop = proper_to_summary_effects(l_prop);    
    l_prop_read = effects_read_effects_dup(l_prop);
    l_prop_write = effects_write_effects_dup(l_prop);
    /* END - IN EFFECTS OF HEADER */
    
    /* INVARIANT WRITE EFFECTS OF LOOP BODY STATEMENT. */
    global_write = 
	effects_write_effects_dup(load_invariant_rw_effects_list(b));

    ifdebug(4){
	pips_debug(4, "W(i)= \n");
	(*effects_prettyprint_func)(global_write);
    }

    /* IN EFFECTS OF LOOP BODY STATEMENT. Effects on locals are masked. */
    lbody_in = load_in_effects_list(b);

    store_cumulated_in_effects_list(b, effects_dup(lbody_in));
    debug_consistent(lbody_in);

    global_in = effects_dup_without_variables(lbody_in, loop_locals(l));

    ifdebug(4){
	pips_debug(4, "initial IN(i)= \n");
	(*effects_prettyprint_func)(global_in);
    }
	        
    
    /* COMPUTATION OF INVARIANT IN EFFECTS */
        
    /* We get the loop transformer, which gives the loop invariant */
    /* We must remove the loop index from the list of modified variables */
    loop_trans = (*load_transformer_func)(current_stat);
    loop_trans = transformer_remove_variable_and_dup(loop_trans, i);
    
    /* And we compute the invariant IN effects. */
    (*effects_transformer_composition_op)(global_in, loop_trans);
    update_invariant_in_effects_list(b, effects_dup(global_in));
    
    ifdebug(4){
	pips_debug(4, "invariant IN(i)= \n");
	(*effects_prettyprint_func)(global_in);
    }
	        
     
    /* OPTIMIZATION : */
    /* If there is no write effect on a variable imported by the loop body,
     * then, the same effect is imported by the whole loop.
     */
    global_in_read_only =
	effects_entities_inf_difference(effects_dup(global_in),
					effects_dup(global_write),
					r_w_combinable_p);
    global_in = 
	effects_entities_intersection(global_in,
				      effects_dup(global_write),
				      r_w_combinable_p);
    
    ifdebug(4){
	pips_debug(4, "reduced IN(i)= \n");
	(*effects_prettyprint_func)(global_in);
    }
    

    if (!ENDP(global_in))
    {
	/* If the loop range cannot be represented in the chosen representation
	 * then, no useful computation can be performed.
	 */
	
	if (! normalizable_and_linear_loop_p(i, r))
	{
	    pips_debug(7, "non linear loop range.\n");
	    effects_to_may_effects(global_in);
	}
	else
	{	    
	    descriptor range_descriptor  = descriptor_undefined;
	    Value incr;
	    Pvecteur v_i_i_prime = VECTEUR_UNDEFINED;

	    pips_debug(7, "linear loop range.\n");
		
	    /* OPTIMIZATION: */
	    /* keep only in global_write the write regions corresponding to 
	     * regions in global_in. */
	    global_write =
		effects_entities_intersection(global_write,
					      effects_dup(global_in),
					      w_r_combinable_p);	
	    ifdebug(4){
		pips_debug(4, "reduced W(i)= \n");
		(*effects_prettyprint_func)(global_write);
	    }
    
	    
	    /* VIRTUAL NORMALIZATION OF LOOP (the new increment is equal 
	     * to +/-1). 
	     * This may result in a new loop index, new_i, with an updated 
	     * range descriptor. Effects are updated at the same time. 
	     */
	    range_descriptor = (*loop_descriptor_make_func)(l); 
	    (*effects_loop_normalize_func)(global_write, i, r,
					   &new_i, range_descriptor, TRUE);
	    (*effects_loop_normalize_func)(global_in, i, r,
					   &new_i, range_descriptor, FALSE);
	    
	    if (!same_entity_p(i,new_i))
		add_intermediate_value(new_i);
	    i = new_i;
  
       	    /* COMPUTATION OF IN EFFECTS. We must remove the effects written 
	     * in previous iterations i.e. IN(i) - U_i'(i'<i)[W(i')] for a 
	     * positive increment, and  IN(i) - U_i'(i < i')[W(i')]
	     * for a negative one.
	     */
	    
	    /* computation of W(i') */     
	    /* i' is here an integer scalar variable */
	    i_prime = entity_to_intermediate_value(i);
	    (*effects_descriptors_variable_change_func)
		(global_write, i, i_prime);
	    	    	    
	    ifdebug(4){
		pips_debug(4, "W(i')= \n");
		(*effects_prettyprint_func)(global_write);
	    }
    

	    /* We must take into account the fact that i<i' or i'<i. */
	    /* This is somewhat implementation dependent. BC. */
	    
	    if (get_descriptor_range_p())
	    {
		incr = vect_coeff
		    (TCST, (Pvecteur) normalized_linear(
			NORMALIZE_EXPRESSION(range_increment(r))));
		v_i_i_prime = vect_make(
		    VECTEUR_NUL, 
		    (Variable) value_pos_p(incr)? i_prime : i, VALUE_ONE,
		    (Variable) value_pos_p(incr)? i : i_prime, VALUE_MONE,
		    TCST, VALUE_ONE);
		range_descriptor =
		    descriptor_inequality_add(range_descriptor, v_i_i_prime);
	    }

	    global_write = (*effects_union_over_range_op)
		(global_write, i_prime, r, range_descriptor);
	    free_descriptor(range_descriptor);

	    ifdebug(4){
		pips_debug(4, "U_i'[W(i')] = \n");
		(*effects_prettyprint_func)(global_write);
	    }
	    
	    /* IN = IN(i) - U_i'[W(i')] */
	    global_in = (*effects_sup_difference_op)(global_in, global_write, 
						     r_w_combinable_p);	    
	    ifdebug(4){
		pips_debug(4, "IN(i) - U_i'[W(i')] = \n");
		(*effects_prettyprint_func)(global_in);
	    }
	    
	    /* We eliminate the loop index */
	    (*effects_union_over_range_op)(global_in, i, range_undefined, 
					   descriptor_undefined);	  
	}
    }
    
    /* we project the read_only regions along the actual loop index i */    
    (*effects_union_over_range_op)(global_in_read_only, loop_index(l), 
				   r, descriptor_undefined);

    global_in = gen_nconc(global_in, global_in_read_only);
    
    /* we remove the write effects from the proper regions of the loop */
    l_in = (*effects_sup_difference_op)
	(global_in, l_prop_write, r_w_combinable_p);
 
    /* we merge these regions with the proper in regions of the loop */
    l_in = (*effects_union_op)(l_in, l_prop_read, effects_same_action_p);

    store_in_effects_list(current_stat,l_in);
    debug_consistent(l_in);

    pips_debug(1, "end\n");
}


static list
in_effects_of_external(entity func, list real_args)
{
    list le = NIL;
    char *func_name = module_local_name(func);
    statement current_stat = effects_private_current_stmt_head();

    pips_debug(4, "translating effects for %s\n", func_name);

    if (! entity_module_p(func)) 
    {
	pips_error("in_effects_of_external", "%s: bad function\n", func_name);
    }
    else 
    {
	list func_eff;
	transformer context;

        /* Get the in summary effects of "func". */	
	func_eff = (*db_get_summary_in_effects_func)(func_name);
	/* Translate them using context information. */
	context = (*load_context_func)(current_stat);
	le = (*effects_backward_translation_op)
	    (func, real_args, func_eff, context);
    }
    return le;
}

static void 
in_effects_of_call(call c)
{
    statement current_stat = effects_private_current_stmt_head();

    list l_in = NIL;
    entity e = call_function(c);
    tag t = value_tag(entity_initial(e));
    string n = module_local_name(e);

    list pc = call_arguments(c);

    pips_debug(1, "begin\n");
    switch (t) {
      case is_value_code:
        pips_debug(5, "external function %s\n", n);
        l_in = in_effects_of_external(e, pc);
	debug_consistent(l_in);
	l_in = proper_to_summary_effects(l_in);
	debug_consistent(l_in);
        break;

      case is_value_intrinsic:
        pips_debug(5, "intrinsic function %s\n", n);
	debug_consistent(l_in);
	l_in = load_rw_effects_list(current_stat);
	debug_consistent(l_in);
	ifdebug(2){
	    pips_debug(2, "R/W effects: \n");
	    (*effects_prettyprint_func)(l_in);
	}	
	l_in = effects_read_effects_dup(l_in);
	debug_consistent(l_in);
        break;

      default:
        pips_error("in_regions_of_call", "unknown tag %d\n", t);
    }

    ifdebug(2){
	pips_debug(2, "IN effects: \n");
	(*effects_prettyprint_func)(l_in);
    }

    store_in_effects_list(current_stat,l_in);
    debug_consistent(l_in);

    pips_debug(1, "end\n");
}



static void 
in_effects_of_unstructured(unstructured u)
{
    statement current_stat = effects_private_current_stmt_head();
    list blocs = NIL ;
    control ct;
    list l_in = NIL, l_tmp = NIL;

    pips_debug(1, "begin\n");

    ct = unstructured_control( u );

    if(control_predecessors(ct) == NIL && control_successors(ct) == NIL)
    {
	/* there is only one statement in u; */
	pips_debug(6, "unique node\n");
	l_in = effects_dup(load_in_effects_list(control_statement(ct)));
    }
    else
    {	
	transformer t_unst = (*load_transformer_func)(current_stat);
	CONTROL_MAP(c,{
	    l_tmp = load_in_effects_list(control_statement(c));	
	    l_in = (*effects_test_union_op) (l_in, effects_dup(l_tmp),
					     effects_same_action_p);
	}, ct, blocs) ;
	(*effects_transformer_composition_op)(l_in, t_unst);
	effects_to_may_effects(l_in);

	gen_free_list(blocs) ;
    }    

    store_in_effects_list(current_stat, l_in);
    debug_consistent(l_in);
    pips_debug(1, "end\n");
}


static void
in_effects_of_module_statement(statement module_stat)
{    
    make_effects_private_current_stmt_stack();
  
     pips_debug(1,"begin\n");
    
    gen_multi_recurse(
	module_stat, 
	statement_domain, in_effects_stmt_filter, in_effects_of_statement,
	sequence_domain, gen_true, in_effects_of_sequence,
	test_domain, gen_true, in_effects_of_test,
	call_domain, gen_true, in_effects_of_call,
	loop_domain, gen_true, in_effects_of_loop,
	unstructured_domain, gen_true, in_effects_of_unstructured,
	expression_domain, gen_false, gen_null, /* NOT THESE CALLS */
	NULL);     

    pips_debug(1,"end\n");
    free_effects_private_current_stmt_stack();
}


/* bool in_regions(char *module_name): 
 * input    : the name of the current module.
 * requires : that transformers and precondition maps be set if needed.
 *            (it depends on the chosen instanciation of *load_context_func
 *             and *load_transformer_func).
 * output   : nothing !
 * modifies : 
 * comment  : computes the in effects of the current module.	
 */
bool
in_effects_engine(char *module_name)
{
    statement module_stat;
    make_effects_private_current_context_stack();
    set_current_module_entity(local_name_to_top_level_entity(module_name)); 

    /* Get the code of the module. */
    set_current_module_statement( (statement)
	db_get_memory_resource(DBR_CODE, module_name, TRUE) );
    module_stat = get_current_module_statement();

    (*effects_computation_init_func)(module_name);

    debug_on("IN_EFFECTS_DEBUG_LEVEL");
    pips_debug(1, "begin for module %s\n", module_name);

    /* set necessary effects maps */
    set_rw_effects((*db_get_rw_effects_func)(module_name));
    set_invariant_rw_effects((*db_get_invariant_rw_effects_func)(module_name));
    set_proper_rw_effects((*db_get_proper_rw_effects_func)(module_name));

    /* initialise the maps for in regions */
    init_in_effects();
    init_invariant_in_effects();
    init_cumulated_in_effects();
  
    /* Compute the effects of the module. */
    in_effects_of_module_statement(module_stat);      
    
    /* Put computed resources in DB. */
    (*db_put_in_effects_func)
	(module_name, get_in_effects());
    (*db_put_invariant_in_effects_func)
	(module_name, get_invariant_in_effects());
    (*db_put_cumulated_in_effects_func)
	(module_name, get_cumulated_in_effects());

    pips_debug(1, "end\n");

    debug_off();

    reset_current_module_entity();
    reset_current_module_statement();

    reset_rw_effects();
    reset_invariant_rw_effects();
    reset_proper_rw_effects();   
 
    reset_in_effects();
    reset_invariant_in_effects();
    reset_cumulated_in_effects();

    (*effects_computation_reset_func)(module_name);

    free_effects_private_current_context_stack();

    return TRUE;
}
