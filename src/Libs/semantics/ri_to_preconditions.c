 /* semantical analysis
  *
  * phasis 2: propagate preconditions from statement to sub-statement,
  * starting from the module 1st statement
  *
  * For (simple) interprocedural analysis, this phasis should be performed
  * top-down on the call tree.
  *
  * Most functions are called xxx_to_postcondition although the purpose is
  * to compute preconditions. However transformer are applied to preconditions
  * to produce postconditions. Thus these modules store the preconditions
  * and then compute an independent (i.e. no sharing) postcondition which
  * is returned to be used by the caller.
  *
  * Preconditions are *NEVER* shared. Sharing would introduce desastrous
  * side effects when they are updated as for equivalenced variables and
  * would make freeing them impossible. Thus on a recursive path from
  * statement_to_postcondition() to itself the precondition must have been
  * reallocated even when its value is not changed as between a block
  * precondition and the first statement of the block precondition. In the
  * same way statement_to_postcondition() should never returned a
  * postcondition aliased with its precondition argument. Somewhere
  * in the recursive call down the data structures towards
  * call_to_postcondition() some allocation must take place even if the
  * statement as no effect on preconditions.
  *
  * Preconditions can be used to evaluate sub-expressions because Fortran
  * standard prohibit side effects within an expression. For instance, in:
  *  J = I + F(I)
  * function F cannot modify I.
  *
  * Ambiguity: the type "transformer" is used to abstract statement effects
  * as well as effects combined from the beginning of the module to just
  * before the current statement (precondition) to just after the current
  * statement (postcondition). This is because it was not realized that
  * preconditions could advantageously be seen as transformers of the initial
  * state when designing the ri.
  */

#include <stdio.h>
#include <string.h>
/* #include <stdlib.h> */

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "control.h"
#include "effects-generic.h"
#include "effects-simple.h"

#include "misc.h"

#include "properties.h"

#include "ray_dte.h"
#include "sommet.h"
#include "sg.h"
#include "polyedre.h"

#include "transformer.h"

#include "semantics.h"

/* another non recursive section used to filter out preconditions */
static list module_global_arguments = NIL;

static list
get_module_global_arguments()
{
    return module_global_arguments;
}

void 
set_module_global_arguments(args)
list args;
{
    module_global_arguments = args;
}
/* end of the non recursive section */

transformer statement_to_postcondition(transformer, statement);

static transformer 
block_to_postcondition(
    transformer b_pre,
    list b)
{
    statement s;
    transformer post;
    transformer s_pre = transformer_undefined;
    list ls = b;

    debug(8,"block_to_postcondition","begin pre=%x\n", b_pre);

    /* The first statement of the block must receive a copy
     * of the block precondition to avoid data sharing
     */

    if(ENDP(ls))
	/* to avoid accumulating equivalence equalities */
	post = transformer_dup(b_pre);
    else {
	s = STATEMENT(CAR(ls));
	s_pre = transformer_dup(b_pre);
	post = statement_to_postcondition(s_pre, s);
	for (POP(ls) ; !ENDP(ls); POP(ls)) {
	    s = STATEMENT(CAR(ls));
	    /* the precondition has been allocated as post */
	    s_pre = post;
	    post = statement_to_postcondition(s_pre, s);
	}
    }

    debug(8,"block_to_postcondition","post=%x end\n", post);
    return post;
}

static transformer 
unstructured_to_postcondition(
    transformer pre,
    unstructured u,
    transformer tf)
{
    transformer post;
    control c;

    debug(8,"unstructured_to_postcondition","begin\n");

    pips_assert("unstructured_to_postcondition", u!=unstructured_undefined);

    c = unstructured_control(u);
    if(control_predecessors(c) == NIL && control_successors(c) == NIL) {
	/* there is only one statement in u; no need for a fix-point */
	debug(8,"unstructured_to_postcondition","unique node\n");
	/* FI: pre should not be duplicated because
	 * statement_to_postcondition() means that pre is not
	 * going to be changed, just post produced.
	 */
	post = statement_to_postcondition(transformer_dup(pre),
					  control_statement(c));
    }
    else {
	/* Do not try anything clever! God knows what may happen in
	   unstructured code. Postcondition post is not computed recursively
	   from its components but directly derived from u's transformer.
	   Preconditions associated to its components are then computed
	   independently, hence the name unstructured_to_postconditionS
	   instead of unstructured_to_postcondition */
	/* propagate as precondition an invariant for the whole
	   unstructured u assuming that all nodes in the CFG are fully
	   connected, unless tf is not feasible because the unstructured
	   is never exited or exited thru a call to STOP which invalidates
	   the previous assumption. */
      transformer tf_u = transformer_undefined;
      transformer pre_u = transformer_undefined;

	debug(8,"unstructured_to_postcondition",
	      "complex: based on transformer\n");
	if(transformer_empty_p(tf)) {
	  tf_u = unstructured_to_global_transformer(u);
	}
	else {
	  tf_u = tf;
	}
	pre_u = invariant_wrt_transformer(pre, tf_u);
	ifdebug(8) {
	  debug(8,"unstructured_to_postcondition",
	      "filtered precondition pre_u:\n");
	  (void) print_transformer(pre_u) ;
	}
	/* FI: I do not know if I should duplicate pre or not. */
	/* FI: well, dumdum, you should have duplicated tf! */
	/* FI: euh... why? According to comments about transformer_apply()
	 * neither arguments are modified...
	 */
	/* post = unstructured_to_postconditions(pre_u, pre, u); */
	post = unstructured_to_accurate_postconditions(pre_u, pre, u);
	pips_assert("A valid postcondition is returned",
		    !transformer_undefined_p(post));
	if(transformer_undefined_p(post)) {
	  post = transformer_apply(transformer_dup(tf), pre);
	}
	transformer_free(pre_u);
    }

    debug(8,"unstructured_to_postcondition","end\n");

    return post;
}

static transformer 
test_to_postcondition(
    transformer pre,
    test t,
    transformer tf)
{
#   define DEBUG_TEST_TO_POSTCONDITION 7
    expression e = test_condition(t);
    statement st = test_true(t);
    statement sf = test_false(t);
    transformer post;

    debug(DEBUG_TEST_TO_POSTCONDITION,"test_to_postcondition","begin\n");

    /* there are three levels of flow sensitivity and we have only a
       boolean flag! FI */

    /* test conditions are assumed to have no side effects; it might
       be somewhere in the standard since functions called in an expression e
       cannot (should not) modify any variable used in e */

    if(pips_flag_p(SEMANTICS_FLOW_SENSITIVE) /* && !transformer_identity_p(tf) */) {
	/* convex hull might avoided if it is not required or if it is certainly useless 
	 * but test information should always be propagated 
	 */
	transformer pret =
	    precondition_add_condition_information(transformer_dup(pre),e,
						  TRUE);
	transformer pref = transformer_undefined;

	transformer postt;
	transformer postf;

	/* "strong" transformer normalization to detect dead code generated by the
	 * test condition
	 */
	/* A normalization of degree 3 is fine */
	/* transformer_normalize(pret, 3); */
	transformer_normalize(pret, 7);

	/* FI, CA: the following "optimization" was added to avoid a call
	 * to Chernikova convex hull that core dumps:-(. 8  September 1993
	 *
	 * From a theoretical point of view, pref could always be computed.
	 *
	 * FI: removed because it is mathematically wrong in many cases;
	 * the negation of the test condition is lost! I keep the structure
	 * just in case another core dump occurs (25 April 1997).
	 */
	if(!empty_statement_p(sf)||TRUE) {
	  
	    pref = precondition_add_condition_information(transformer_dup(pre),e,
							  FALSE);
	    /* transformer_normalize(pref, 3); */
	    transformer_normalize(pref, 7);
	}
	else {
	    /* do not try to compute a refined precondition for an empty block
	     * keep the current precondition to store in the precondition statement mapping
	     */
	    pref = transformer_dup(pre);
	}

	ifdebug(DEBUG_TEST_TO_POSTCONDITION) {
	    debug(DEBUG_TEST_TO_POSTCONDITION,"test_to_postcondition","pret=\n");
	    (void) print_transformer(pret);
	    debug(DEBUG_TEST_TO_POSTCONDITION,"test_to_postcondition","pref=\n");
	    (void) print_transformer(pref);
	}

	postt = statement_to_postcondition(pret, st);
	postf = statement_to_postcondition(pref, sf);
	post = transformer_convex_hull(postt, postf);
	transformer_free(postt);
	transformer_free(postf);
    }
    else {
	(void) statement_to_postcondition(pre, st);
	(void) statement_to_postcondition(pre, sf);
	post = transformer_apply(tf, pre);
    }

    ifdebug(DEBUG_TEST_TO_POSTCONDITION) {
	debug(DEBUG_TEST_TO_POSTCONDITION,"test_to_postcondition","end post=\n");
	(void) print_transformer(post);
    }

    return post;
}

static transformer 
call_to_postcondition(
    transformer pre,
    call c,
    transformer tf)
{
    transformer post = transformer_undefined;
    entity e = call_function(c);
    tag tt;

    pips_debug(8,"begin\n");

    switch (tt = value_tag(entity_initial(e))) {
      case is_value_intrinsic:
	/* there is room for improvement because assign is now the only 
	   well handled intrinsic */
	pips_debug(5, "intrinsic function %s\n",
	      entity_name(e));
	post = transformer_apply(tf, pre);
	/* propagate precondition pre as summary precondition 
	   of user functions */
	/* FI: don't! Summary preconditions are computed independently*/
	/*
	   if(get_bool_property(SEMANTICS_INTERPROCEDURAL)) {
	   list args = call_arguments(c);
	   expressions_to_summary_precondition(pre, args);
	   }
	   */
	break;
      case is_value_code:
	pips_debug(5, "external function %s\n", entity_name(e));
	if(get_bool_property(SEMANTICS_INTERPROCEDURAL)) {
	    /*
	      list args = call_arguments(c);

	    transformer pre_callee = transformer_dup(pre);
	    pre_callee = 
		add_formal_to_actual_bindings(c, pre_callee);
	    add_module_call_site_precondition(e, pre_callee);
	    */
	    /*
	    expressions_to_summary_precondition(pre, args);
	    */
	}
	post = transformer_apply(tf, pre);
	break;
      case is_value_symbolic:
      case is_value_constant:
	pips_internal_error("call to symbolic or constant %s\n", 
			    entity_name(e));
	break;
      case is_value_unknown:
	pips_internal_error("unknown function %s\n", entity_name(e));
	break;
      default:
	pips_internal_error("unknown tag %d\n", tt);
    }

    pips_debug(8,"end\n");

    return post;
}

/******************************************************** DATA PRECONDITIONS */

/* a remake that works in all cases. FC.
 */
static transformer 
data_to_prec_for_variables(entity m, list /* of entity */le) 
{
  transformer pre = transformer_identity();
  linear_hashtable_pt b = linear_hashtable_make(); /* already seen */
  
  pips_debug(8, "begin for %s\n", module_local_name(m));
  
  /* look for entities with an initial value. */
  MAP(ENTITY, e,
  {
    value val = entity_initial(e);

    if(value_constant_p(val))
    {
      constant c = value_constant(val);
      if (constant_int_p(c))
      {
	int int_val = constant_int(value_constant(val));
	if(entity_has_values_p(e) && !linear_hashtable_isin(b, e))
	{
	  Pvecteur v = vect_make(VECTEUR_NUL,
				 (Variable) e, VALUE_ONE,
				 TCST, int_to_value(-int_val));
	  pre = transformer_equality_add(pre, v);
	  linear_hashtable_put_once(b, e, e);
	}
      }
      else if (constant_call_p(c))
      {
	if (entity_has_values_p(e) && !linear_hashtable_isin(b, e))
	{
	  Pvecteur v = vect_make(VECTEUR_NUL,
				 (Variable) e, VALUE_ONE,
				 (Variable) constant_call(c), VALUE_MONE,
				 TCST, VALUE_ZERO);
	  pre = transformer_equality_add(pre, v);
	  linear_hashtable_put_once(b, e, e);
	}
      }
    }
  },
      le);
      
  linear_hashtable_free(b);
  pips_assert("some transformer", pre != transformer_undefined);
  
  ifdebug(8) {
    dump_transformer(pre);
    pips_debug(8, "end for %s\n", module_local_name(m));
  }
  
  return pre;
}

/* returns an allocated list of entities that appear in lef.
 * an entity may appear several times.
 */
list effects_to_entity_list(list lef)
{
  list le = NIL;
  MAP(EFFECT, e, 
      le = CONS(ENTITY, reference_variable(effect_reference(e)), le),
      lef);
  return gen_nreverse(le);
}

/* restricted to variables with effects. */
transformer data_to_precondition(entity m)
{
  list lef = load_module_intraprocedural_effects(m);
  list le = effects_to_entity_list(lef);
  transformer pre = data_to_prec_for_variables(m, le);
  gen_free_list(le);
  return pre;
}

/* any variable is included. */
transformer all_data_to_precondition(entity m)
{
  transformer pre =
    data_to_prec_for_variables(m, code_declarations(entity_code(m)));
  return pre;
}

static transformer 
instruction_to_postcondition(
    transformer pre,
    instruction i,
    transformer tf)
{
    transformer post = transformer_undefined;
    test t;
    loop l;
    whileloop wl;
    call c;

    pips_debug(9,"begin pre=%p tf=%p\n", pre, tf);

    switch(instruction_tag(i)) {
      case is_instruction_block:
	post = block_to_postcondition(pre, instruction_block(i));
	break;
      case is_instruction_test:
	t = instruction_test(i);
	post = test_to_postcondition(pre, t, tf);
	break;
      case is_instruction_loop:
	l = instruction_loop(i);
	post = loop_to_postcondition(pre, l, tf);
	break;
      case is_instruction_whileloop:
	wl = instruction_whileloop(i);
	post = whileloop_to_postcondition(pre, wl, tf);
	break;
      case is_instruction_goto:
	pips_error("instruction_to_postcondition",
		   "unexpected goto in semantics analysis");
	/* never reached: post = pre; */
	break;
      case is_instruction_call:
	c = instruction_call(i);
	post = call_to_postcondition(pre, c, tf);
	break;
      case is_instruction_unstructured:
	post = unstructured_to_postcondition(pre, instruction_unstructured(i),
					     tf);
	break ;
      default:
	pips_error("instruction_to_postcondition","unexpected tag %d\n",
	      instruction_tag(i));
    }
    pips_debug(9,"resultat post, %p:\n", post);
    ifdebug(9) (void) print_transformer(post);
    return post;
}

/* Assume that all references are legal. Assume that variables used in
   array declarations are not modified in the module. */
static void add_reference_information(transformer pre, statement s, bool renaming)
{
  list efs = load_proper_rw_effects_list(s);

  MAP(EFFECT, e, {
    reference r = effect_reference(e);
    list li = reference_indices(r);

    if(!ENDP(li)){
      entity v = reference_variable(r);
      variable tv = type_variable(entity_type(v));
      list ld = NIL;

      pips_assert("Variable must be of type 'variable'",
		  type_variable_p(entity_type(v)));
      ld = variable_dimensions(tv);
      pips_assert("Reference dimension = array dimension",
		  gen_length(li)==gen_length(ld));
      MAP(EXPRESSION, i, {
	normalized ni = NORMALIZE_EXPRESSION(i);
	if(normalized_linear_p(ni)) {
	  Pvecteur vi = normalized_linear(ni);
	  dimension d = DIMENSION(CAR(ld));
	  normalized nl = NORMALIZE_EXPRESSION(dimension_lower(d));
	  normalized nu = NORMALIZE_EXPRESSION(dimension_upper(d));
	  if(normalized_linear_p(nl) && normalized_linear_p(nu)) {
	    Pvecteur vl = normalized_linear(nl);
	    Pvecteur vu = normalized_linear(nu);

	    if(value_mappings_compatible_vector_p(vi)) {
	      if(value_mappings_compatible_vector_p(vl)) {
		Pvecteur cv = vect_substract(vl, vi);

		if(renaming)
		  upwards_vect_rename(cv, pre);
		if(!vect_constant_p(cv) || vect_coeff(TCST, cv) > 0) {
		  transformer_inequality_add(pre, cv);
		}
		else {
		  vect_rm(cv);
		}
	      }

	      if(value_mappings_compatible_vector_p(vu)) {
		Pvecteur cv = vect_substract(vi, vu);

		if(renaming)
		  upwards_vect_rename(cv, pre);
		if(!vect_constant_p(cv) || vect_coeff(TCST, cv) > 0) {
		  transformer_inequality_add(pre, cv);
		}
		else {
		  vect_rm(cv);
		}
	      }
	    }

	  }
	}
	POP(ld);
      }, li);
    }
  }, efs);
}

void precondition_add_reference_information(transformer pre, statement s)
{
  add_reference_information(pre, s, FALSE);
}

void transformer_add_reference_information(transformer tf, statement s)
{
  add_reference_information(tf, s, TRUE);
}

transformer 
statement_to_postcondition(
    transformer pre,
    statement s)
{
    transformer post = transformer_undefined;
    instruction i = statement_instruction(s);
    transformer tf = load_statement_transformer(s);

    /* ACHTUNG! "pre" is likely to be misused! FI, Sept. 3, 1990 */

    debug(1,"statement_to_postcondition","begin\n");

    pips_assert("The statement precondition is defined", pre != transformer_undefined);

    ifdebug(1) {
	int so = statement_ordering(s);
	(void) fprintf(stderr, "statement %03d (%d,%d), precondition %p:\n",
		       statement_number(s), ORDERING_NUMBER(so),
		       ORDERING_STATEMENT(so), pre);
	(void) print_transformer(pre) ;
    }

    pips_assert("The statement transformer is defined", tf != transformer_undefined);
    ifdebug(1) {
	int so = statement_ordering(s);
	(void) fprintf(stderr, "statement %03d (%d,%d), transformer %p:\n",
		       statement_number(s), ORDERING_NUMBER(so),
		       ORDERING_STATEMENT(so), tf);
	(void) print_transformer(tf) ;
    }

    if (!statement_reachable_p(s))
    {
	/* FC: if the code is not reachable (thanks to STOP or GOTO), which
	 * is a structural information, the precondition is just empty.
	 */
      /* Psysteme s = predicate_system(transformer_relation(pre)); */

      free_predicate(transformer_relation(pre));
      gen_free_list(transformer_arguments(pre));
      transformer_arguments(pre) = NIL;
      transformer_relation(pre) = make_predicate(sc_empty(BASE_NULLE));
    }

    if (load_statement_precondition(s) == transformer_undefined) {
	/* keep only global initial scalar integer values;
	   else, you might end up giving the same xxx#old name to
	   two different local values */
	list non_initial_values =
	    arguments_difference(transformer_arguments(pre),
				 get_module_global_arguments());

	MAPL(cv,
	 {
	   entity v = ENTITY(CAR(cv));
	   ENTITY(CAR(cv)) = entity_to_old_value(v);
	 },
	     non_initial_values);

	/* add array references information */
	if(get_bool_property("SEMANTICS_TRUST_ARRAY_REFERENCES")) {
	    precondition_add_reference_information(pre, s);
	}

	post = instruction_to_postcondition(pre, i, tf);

	/* add equivalence equalities */
	pre = tf_equivalence_equalities_add(pre);

	/* eliminate redundancy */
	/* FI: nice... but time consuming! */
	/* Version 3 is OK. Equations are over used and make
	 * inequalities uselessly conplex
	 */
	/* pre = transformer_normalize(pre, 3); */

	pre = transformer_normalize(pre, 6);

	if(!transformer_consistency_p(pre)) {
	    int so = statement_ordering(s);
	    fprintf(stderr, "statement %03d (%d,%d), precondition %p end:\n",
			   statement_number(s), ORDERING_NUMBER(so),
			   ORDERING_STATEMENT(so), pre);
	    print_transformer(pre);
	    pips_internal_error("Non-consistent precondition after update\n");
	}

	/* store the precondition in the ri */
	store_statement_precondition(s,
				     transformer_filter(pre,
							non_initial_values));

	gen_free_list(non_initial_values);
    }
    else {
	pips_debug(8,"precondition already available");
	/* pre = statement_precondition(s); */
	(void) print_transformer(pre);
	pips_error("statement_to_postcondition",
		   "precondition already computed\n");
    }

    /* post = instruction_to_postcondition(pre, i, tf); */

    ifdebug(1) {
	int so = statement_ordering(s);
	fprintf(stderr, "statement %03d (%d,%d), precondition %p end:\n",
		statement_number(s), ORDERING_NUMBER(so),
		ORDERING_STATEMENT(so), load_statement_precondition(s));
	print_transformer(load_statement_precondition(s)) ;
    }

    ifdebug(1) {
	int so = statement_ordering(s);
	fprintf(stderr, "statement %03d (%d,%d), postcondition %p:\n",
		statement_number(s), ORDERING_NUMBER(so),
		ORDERING_STATEMENT(so), post);
	print_transformer(post) ;
    }

    pips_assert("statement_to_postcondition: unexpected sharing",post!=pre);

    debug(1,"statement_to_postcondition","end\n");

    return post;
}
