 /* semantical analysis
  *
  * phasis 1: compute transformers from statements and statements effects
  *
  * For (simple) interprocedural analysis, this phasis should be performed
  * bottom-up on the call tree.
  *
  * Francois Irigoin, April 1990
  *
  * $Id$
  *
  * $Log: ri_to_transformers.c,v $
  * Revision 1.57  2001/07/19 17:58:09  irigoin
  * Two bug fixes + reformatting with a smaller indent
  *
  * Revision 1.56  2001/07/13 15:02:58  irigoin
  * Restructured version with separate processing of loops and
  * expressions. Multitype version.
  *
  * Revision 1.55  2001/02/07 18:14:21  irigoin
  * New C format + support for recomputing loop fixpoints with precondition information
  *
  * Revision 1.54  2000/11/23 17:17:31  irigoin
  * Function moved into unstructured.c, typing in debugging statement,
  * consistency checks
  *
  * Revision 1.53  2000/11/03 17:15:07  irigoin
  * Declarations and references are trusted or not. Better handling of unstructured.
  *
  * Revision 1.52  2000/07/20 14:29:42  coelho
  * cleaner.
  *
  * Revision 1.51  2000/07/20 14:25:26  coelho
  * leaks--
  *
  * Revision 1.50  2000/07/20 13:55:40  irigoin
  * cleaning...
  *
  * Revision 1.49  2000/05/25 08:37:55  coelho
  * no more successor when adding an eq of ineq.
  *
  * Revision 1.48  1999/01/07 16:44:14  irigoin
  * Bug fix in user_call_to_transformer() to handle aliasing between two formal parameters. See spice02.f in Validation.
  *
  * Revision 1.47  1999/01/07 07:52:32  irigoin
  * Bug fix in minmax_to_transformer()
  *
  */

#include <stdio.h>
#include <string.h>
/* #include <stdlib.h> */

#include "genC.h"
#include "database.h"
#include "linear.h"
#include "ri.h"
#include "text.h"
#include "text-util.h"
#include "ri-util.h"
#include "constants.h"
#include "control.h"
#include "effects-generic.h"
#include "effects-simple.h"

#include "misc.h"

#include "properties.h"

#include "vecteur.h"
#include "contrainte.h"
#include "ray_dte.h"
#include "sommet.h"
#include "sg.h"
#include "polyedre.h"

#include "transformer.h"

#include "semantics.h"

extern Psysteme sc_projection_by_eq(Psysteme sc, Pcontrainte eq, Variable v);

transformer effects_to_transformer(list e) /* list of effects */
{
  /* algorithm: keep only write effects on variables with values */
  list args = NIL;
  Pbase b = VECTEUR_NUL;
  Psysteme s = sc_new();

  MAP(EFFECT, ef, 
  {
    reference r = effect_reference(ef);
    action a = effect_action(ef);
    entity v = reference_variable(r);
    
    if(action_write_p(a) && entity_has_values_p(v)) 
    {
      entity new_val = entity_to_new_value(v);
      args = arguments_add_entity(args, new_val);
      b = vect_add_variable(b, (Variable) new_val);
    }
  },
      e);
  
  s->base = b;
  s->dimension = vect_size(b);

  return make_transformer(args, make_predicate(s));
}

transformer filter_transformer(transformer t, list e) 
{
  /* algorithm: keep only information about scalar variables with values
   * appearing in effects e and store it into a newly allocated transformer
   */
  Pbase b = VECTEUR_NUL;
  Psysteme s = SC_UNDEFINED;
  Psysteme sc = predicate_system(transformer_relation(t));
  list args = NIL;
  
  MAP(EFFECT, ef, 
  {
    reference r = effect_reference(ef);
    /* action a = effect_action(ef); */
    entity v = reference_variable(r);
    
    if(/* action_write_p(a) && */ entity_has_values_p(v)) {
      /* I do not know yet if I should keep old values... */
      entity new_val = entity_to_new_value(v);
      b = vect_add_variable(b, (Variable) new_val);
      
      if(entity_is_argument_p(v, transformer_arguments(t))) {
	args = arguments_add_entity(args, v);
      }
    }
  },
      e);
  
  /* FI: I should check if sc is sc_empty but I haven't (yet) found a
     cheap syntactic test */
  s = sc_restricted_to_variables_transitive_closure(sc, b);

  return make_transformer(args, make_predicate(s));
}


/* Recursive Descent in Data Structure Statement */

/* SHARING : returns the transformer stored in the database. Make a copy 
 * before using it. The copy is not made here because the result is not 
 * always used after a call to this function, and itwould create non 
 * reachable structures. Another solution would be to store a copy and free 
 * the unused result in the calling function but transformer_free does not 
 * really free the transformer. Not very clean. 
 * BC, oct. 94 
 */

static transformer 
block_to_transformer(list b)
{
  statement s;
  transformer btf;
  transformer stf = transformer_undefined;
  list l = b;

  pips_debug(8,"begin\n");

  if(ENDP(l))
    btf = transformer_identity();
  else {
    s = STATEMENT(CAR(l));
    stf = statement_to_transformer(s);
    btf = transformer_dup(stf);
    for (POP(l) ; !ENDP(l); POP(l)) {
      s = STATEMENT(CAR(l));
      stf = statement_to_transformer(s);
      btf = transformer_combine(btf, stf);
      ifdebug(1) 
	pips_assert("consistent transformer", 
		    transformer_consistency_p(btf));
    }
  }

  pips_debug(8, "end\n");
  return btf;
}

static void 
unstructured_to_transformers(unstructured u)
{
  list blocs = NIL ;
  control ct = unstructured_control(u) ;
  
  pips_debug(5,"begin\n");
  
  /* There is no need to compute transformers for unreachable code,
   * using CONTROL_MAP, but this may create storage and prettyprinter
   * problems because of the data structure inconsistency.
   */
  CONTROL_MAP(c, {
    statement st = control_statement(c) ;
    (void) statement_to_transformer(st) ;
  }, ct, blocs) ;
  
  gen_free_list(blocs) ;
  
  pips_debug(5,"end\n");
}

static transformer 
unstructured_to_transformer(unstructured u, list e) /* effects */
{
  transformer ctf;
  transformer tf;
  control c;

  pips_debug(8,"begin\n");

  pips_assert("unstructured_to_transformer", u!=unstructured_undefined);

  c = unstructured_control(u);
  if(control_predecessors(c) == NIL && control_successors(c) == NIL) {
    /* there is only one statement in u; no need for a fix-point */
    pips_debug(8,"unique node\n");
    ctf = statement_to_transformer(control_statement(c));
    tf = transformer_dup(ctf);
  }
  else {
    statement exit = control_statement(unstructured_exit(u));
      
    pips_debug(8,"complex: based on effects\n");
      
    unstructured_to_transformers(u);
      
    tf = unstructured_to_accurate_transformer(u, e);
  }

  pips_debug(8,"end\n");

  return tf;
}

list 
effects_to_arguments(list fx) /* list of effects */
{
  /* algorithm: keep only write effects on scalar variable with values */
  list args = NIL;

  MAP(EFFECT, ef, 
  {
    reference r = effect_reference(ef);
    action a = effect_action(ef);
    entity e = reference_variable(r);
	
    if(action_write_p(a) && entity_has_values_p(e)) {
      args = arguments_add_entity(args, e);
    }
  },
      fx);

  return args;
}

/* The loop initialization is performed before tf */
transformer transformer_add_loop_index_initialization(transformer tf, loop l)
{
  entity i = loop_index(l);
  range r = loop_range(l);
  normalized nlb = NORMALIZE_EXPRESSION(range_lower(r));

  if(entity_has_values_p(i) && normalized_linear_p(nlb)) {
    Psysteme sc = (Psysteme) predicate_system(transformer_relation(tf));
    Pcontrainte eq = CONTRAINTE_UNDEFINED;
    Pvecteur v_lb = vect_dup(normalized_linear(nlb));
    Pbase b_tmp, b_lb = make_base_from_vect(v_lb); 
    entity i_init = entity_to_old_value(i);

    vect_add_elem(&v_lb, (Variable) i_init, VALUE_MONE);
    eq = contrainte_make(v_lb);
    /* The new variables in eq must be added to sc; otherwise,
     * further consistency checks core dump. bc.
     */
    /* sc_add_egalite(sc, eq); */
    /* The call to sc_projection_with_eq frees eq */
    sc = sc_projection_by_eq(sc, eq, (Variable) i_init);
    b_tmp = sc_base(sc);
    sc_base(sc) = base_union(b_tmp, b_lb);
    sc_dimension(sc) = base_dimension(sc_base(sc));
    base_rm(b_tmp);
    base_rm(b_lb);
    if(SC_RN_P(sc)) {
      /* FI: a NULL is not acceptable; I assume that we cannot
       * end up with a SC_EMPTY...
       */
      predicate_system_(transformer_relation(tf)) =
	newgen_Psysteme
	(sc_make(CONTRAINTE_UNDEFINED, CONTRAINTE_UNDEFINED));
    }
    else
      predicate_system_(transformer_relation(tf)) = 
	newgen_Psysteme(sc);
  }
  else if(entity_has_values_p(i)) {
    /* Get rid of the initial value since it is unknowable */
    entity i_init = entity_to_old_value(i);
    list l_i_init = CONS(ENTITY, i_init, NIL);

    tf = transformer_projection(tf, l_i_init);
  }

return tf;
}

transformer transformer_add_loop_index_incrementation(transformer tf, loop l)
{
  entity i = loop_index(l);
  range r = loop_range(l);
  expression incr = range_increment(r);
  Pvecteur v_incr = VECTEUR_UNDEFINED;

  pips_assert("Transformer tf is consistent before update",
	      transformer_consistency_p(tf));

  /* it does not contain the loop index update
     the loop increment expression must be linear to find inductive 
     variables related to the loop index */
  if(!VECTEUR_UNDEFINED_P(v_incr = expression_to_affine(incr))) {
    if(entity_has_values_p(i)) {
      if(value_mappings_compatible_vector_p(v_incr)) {
	tf = transformer_add_variable_incrementation(tf, i, v_incr);
      }
      else {
	entity i_old = entity_to_old_value(i);
	entity i_new = entity_to_new_value(i);
	Psysteme sc = predicate_system(transformer_relation(tf));
	Pbase b = sc_base(sc);
	
	transformer_arguments(tf) = arguments_add_entity(transformer_arguments(tf), i);
	b = base_add_variable(b, (Variable) i_old);
	b = base_add_variable(b, (Variable) i_new);
	sc_base(sc) = b;
	sc_dimension(sc) = base_dimension(sc_base(sc));
      }
    }
    else {
      pips_user_warning("non-integer or equivalenced loop index %s?\n",
			entity_local_name(i));
    }
  }
  else {
    if(entity_has_values_p(i)) {
      entity i_old = entity_to_old_value(i);
      entity i_new = entity_to_new_value(i);
      Psysteme sc = predicate_system(transformer_relation(tf));
      Pbase b = sc_base(sc);

      transformer_arguments(tf) = arguments_add_entity(transformer_arguments(tf), i);
      b = base_add_variable(b, (Variable) i_old);
      b = base_add_variable(b, (Variable) i_new);
      sc_base(sc) = b;
      sc_dimension(sc) = base_dimension(sc_base(sc));
    }
  }

  pips_assert("Transformer tf is consistent after update",
	      transformer_consistency_p(tf));

  return tf;
}

/* The transformer associated to a DO loop does not include the exit 
 * condition because it is used to compute the precondition for any 
 * loop iteration.
 *
 * There is only one attachment for the unbounded transformer and
 * for the bounded one.
 */

static transformer 
loop_to_transformer(loop l)
{
  /* loop transformer tf = tfb* or tf = tfb+ or ... */
  transformer tf;
  /* loop body transformer */
  transformer tfb;
  range r = loop_range(l);
  statement s = loop_body(l);

  pips_debug(8,"begin\n");

  /* compute the loop body transformer */
  tfb = transformer_dup(statement_to_transformer(s));
  tfb = transformer_add_loop_index_incrementation(tfb, l);

  /* compute tfb's fix point according to pips flags */
  tf = (* transformer_fix_point_operator)(tfb);

  ifdebug(8) {
    pips_debug(8, "intermediate fix-point tf=\n");
    fprint_transformer(stderr, tf, external_value_name);
  }

  /* add initialization for the unconditional initialization of the loop
     index variable */
  tf = transformer_add_loop_index_initialization(tf, l);

  ifdebug(8) {
    debug(8, "loop_to_transformer", "full fix-point tf=\n");
    fprint_transformer(stderr, tf, external_value_name);
    debug(8, "loop_to_transformer", "end\n");
  }

  /* we have a problem here: to compute preconditions within the
     loop body we need a tf using private variables; to return
     the loop transformer, we need a filtered out tf; only
     one hook is available in the ri..; let'a assume there
     are no private variables and that if they are privatizable
     they are not going to get in our way */

  ifdebug(8) {
    (void) fprintf(stderr,"%s: %s\n","loop_to_transformer",
		   "resultat tf =");
    (void) (void) print_transformer(tf);
    debug(8,"loop_to_transformer","end\n");
  }

  return tf;
}

/* The index variable is always initialized and then the loop is either
   entered and exited or not entered */
transformer 
refine_loop_transformer(transformer t, loop l)
{
  transformer tf = transformer_undefined;
  transformer t_enter = transformer_undefined;
  transformer t_skip = transformer_undefined;
  transformer pre = transformer_undefined;
  /* loop body transformer */
  transformer tfb = transformer_undefined;
  range r = loop_range(l);
  statement s = loop_body(l);

  pips_debug(8,"begin\n");

  /* compute the loop body transformer */
  tfb = transformer_dup(load_statement_transformer(s));
  tfb = transformer_add_loop_index_incrementation(tfb, l);

  /* compute the transformer when the loop is entered */
  t_enter = transformer_combine(tfb, t);

  /* add the entry condition */
  /* but it seems to be in t already */
  /* t_enter = transformer_add_loop_index_initialization(t_enter, l); */

  /* add the exit condition, without any information pre to estimate the
     increment */
  pre = transformer_identity();
  t_enter = add_loop_index_exit_value(t_enter, l, pre, NIL);

  ifdebug(8) {
    pips_debug(8, "entered loop transformer t_enter=\n");
    fprint_transformer(stderr, t_enter, external_value_name);
  }

  /* add initialization for the unconditional initialization of the loop
     index variable */
  t_skip = transformer_identity();
  t_skip = add_loop_index_initialization(t_skip, l);
  t_skip = add_loop_skip_condition(t_skip, l);

  ifdebug(8) {
    pips_debug(8, "skipped loop transformer t_skip=\n");
    fprint_transformer(stderr, t_skip, external_value_name);
  }

  /* It might be better not to compute useless transformer, but it's more
     readbale that way. Since pre is information free, only loops with
     constant lower and upper bound and constant increment can benefit
     from this. */
  if(empty_range_wrt_precondition_p(r, pre)) {
    tf = t_skip;
    free_transformer(t_enter);
  }
  else if(non_empty_range_wrt_precondition_p(r, pre)) {
    tf = t_enter;
    free_transformer(t_skip);
  }
  else {
    tf = transformer_convex_hull(t_enter, t_skip);
    free_transformer(t_enter);
    free_transformer(t_skip);
  }

  free_transformer(pre);

  ifdebug(8) {
    pips_debug(8, "full refined loop transformer tf=\n");
    fprint_transformer(stderr, tf, external_value_name);
    pips_debug(8, "end\n");
  }

  return tf;
}

/* This function computes the effect of K loop iteration, with K positive.
 * This function does not take the loop exit into account because its result
 * is used to compute the precondition of the loop body.
 * Hence the loop exit condition only is added when preconditions are computed.
 * This is confusing when transformers are prettyprinted with the source code.
 */

static transformer 
whileloop_to_transformer(whileloop l, list e) /* effects of whileloop l */
{
  /* loop transformer tf = tfb* or tf = tfb+ or ... */
  transformer tf;
  /* loop body transformer */
  transformer tfb;
  expression cond = whileloop_condition(l);
  statement s = whileloop_body(l);

  debug(8,"whileloop_to_transformer","begin\n");

  if(pips_flag_p(SEMANTICS_FIX_POINT)) {
    /* compute the whileloop body transformer */
    tfb = transformer_dup(statement_to_transformer(s));

    /* If the while entry condition is usable, it must be added
     * on the old values
     */
    tfb = transformer_add_condition_information(tfb, cond, TRUE);

    /* compute tfb's fix point according to pips flags */
    if(pips_flag_p(SEMANTICS_INEQUALITY_INVARIANT)) {
      tf = transformer_halbwachs_fix_point(tfb);
    }
    else if (transformer_empty_p(tfb)) {
      /* The loop is never entered */
      tf = transformer_identity();
    }
    else {
      transformer ftf = (* transformer_fix_point_operator)(tfb);

      if(*transformer_fix_point_operator==transformer_equality_fix_point) {
	Psysteme fsc = predicate_system(transformer_relation(ftf));
	Psysteme sc = SC_UNDEFINED;
	    
	/* Dirty looking fix for a fix point computation error:
	 * sometimes, the basis is restricted to a subset of
	 * the integer scalar variables. Should be useless with proper
	 * fixpoint opertors.
	 */
	tf = effects_to_transformer(e);
	sc = (Psysteme) predicate_system(transformer_relation(tf));

	sc = sc_append(sc, fsc);

	free_transformer(ftf);
      }
      else {
	tf = ftf;
      }

      ifdebug(8) {
	pips_debug(8, "intermediate fix-point tf=\n");
	fprint_transformer(stderr, tf, external_value_name);
      }

    }
    /* we have a problem here: to compute preconditions within the
       whileloop body we need a tf using private variables; to return
       the loop transformer, we need a filtered out tf; only
       one hook is available in the ri..; let'a assume there
       are no private variables and that if they are privatizable
       they are not going to get in our way */
  }
  else {
    /* basic cheap version: do not use the whileloop body transformer and
       avoid fix-points; local variables do not have to be filtered out
       because this was already done while computing effects */

    (void) statement_to_transformer(s);
    tf = effects_to_transformer(e);
  }

  ifdebug(8) {
    (void) fprintf(stderr,"%s: %s\n","whileloop_to_transformer",
		   "resultat tf =");
    (void) (void) print_transformer(tf);
  }
  debug(8,"whileloop_to_transformer","end\n");
  return tf;
}

static transformer 
test_to_transformer(test t, list ef) /* effects of t */
{
  statement st = test_true(t);
  statement sf = test_false(t);
  transformer tf;

  debug(8,"test_to_transformer","begin\n");

  /* the test condition cannot be used to improve transformers
     it may be used later when propagating preconditions 
     Francois Irigoin, 15 April 1990 

     Why?
     Francois Irigoin, 31 July 1992

     Well, you can benefit from STOP statements.
     But you do not know if the variable values in
     the condition are the new or the old values...
     Francois Irigoin, 8 November 1995

  */

  if(pips_flag_p(SEMANTICS_FLOW_SENSITIVE)) {
    /*
      tft = statement_to_transformer(st);
      tff = statement_to_transformer(sf);
      tf = transformer_convex_hull(tft, tff);
    */
    expression e = test_condition(t);
    transformer tftwc;
    transformer tffwc;
    list ta = NIL;
    list fa = NIL;

    tftwc = transformer_dup(statement_to_transformer(st));
    tffwc = transformer_dup(statement_to_transformer(sf));

    /* Look for variables modified in one branch only */
    /* This is performed in transformer_convex_hull
       ta = arguments_difference(transformer_arguments(tftwc),
       transformer_arguments(tffwc));
       fa = arguments_difference(transformer_arguments(tffwc),
       transformer_arguments(tftwc));

       MAPL(ca, {
       entity v = ENTITY(CAR(ca));

       tffwc = transformer_add_identity(tffwc, v);
       }, ta);

       MAPL(ca, {
       entity v = ENTITY(CAR(ca));

       tftwc = transformer_add_identity(tftwc, v);
       }, fa);
    */

    tftwc = transformer_add_condition_information(tftwc, e, TRUE);
    tffwc = transformer_add_condition_information(tffwc, e, FALSE);

    tf = transformer_convex_hull(tftwc, tffwc);
    transformer_free(tftwc);
    transformer_free(tffwc);
    free_arguments(ta);
    free_arguments(fa);
  }
  else {
    (void) statement_to_transformer(st);
    (void) statement_to_transformer(sf);
    tf = effects_to_transformer(ef);
  }

  debug(8,"test_to_transformer","end\n");
  return tf;
}

static transformer 
intrinsic_to_transformer(
    entity e, list pc, list ef) /* effects of intrinsic call */
{
  transformer tf;
  /* should become a parameter, but one thing at a time */
  transformer pre = transformer_undefined;

  debug(8,"intrinsic_to_transformer","begin\n");

  if(ENTITY_ASSIGN_P(e)) {
    tf = any_assign_to_transformer(pc, ef, pre);
  }
  else if(ENTITY_STOP_P(e))
    tf = transformer_empty();
  else
    tf = effects_to_transformer(ef);

  debug(8,"intrinsic_to_transformer","end\n");

  return tf;
}

static transformer user_call_to_transformer(entity, list, list);

static transformer 
call_to_transformer(call c, list ef) /* effects of call c */
{
  transformer tf = transformer_undefined;
  entity e = call_function(c);
  cons *pc = call_arguments(c);
  tag tt;

  pips_debug(8,"begin\n");

  switch (tt = value_tag(entity_initial(e))) {
  case is_value_code:
    /* call to an external function; preliminary version:
       rely on effects */
    pips_debug(5, "external function %s\n", entity_name(e));
    if(get_bool_property(SEMANTICS_INTERPROCEDURAL)) {
      tf = user_call_to_transformer(e, pc, ef);
      reset_temporary_value_counter();
    }
    else
      tf = effects_to_transformer(ef);
    break;
  case is_value_symbolic:
  case is_value_constant:
    tf = transformer_identity();
    break;
  case is_value_unknown:
    pips_internal_error("function %s has an unknown value\n", entity_name(e));
    break;
  case is_value_intrinsic:
    pips_debug(5, "intrinsic function %s\n", entity_name(e));
    tf = intrinsic_to_transformer(e, pc, ef);
    break;
  default:
    pips_internal_error("unknown tag %d\n", tt);
  }
  pips_assert("transformer tt is consistent", 
	      transformer_consistency_p(tf)); 


  pips_debug(8,"end\n");

  return(tf);
}

transformer 
user_function_call_to_transformer(
				  entity e, /* a value */
				  expression expr) /* a call to a function */
{
  syntax s = expression_syntax(expr);
  call c = syntax_call(s);
  entity f = call_function(c);
  list pc = call_arguments(c);
  transformer t_caller = transformer_undefined;
  basic rbt = basic_of_call(c);
  list ef = expression_to_proper_effects(expr);

  pips_debug(8, "begin\n");
  pips_assert("s is a call", syntax_call_p(s));

  /* if(basic_int_p(rbt)) { */
  if(basic_equal_p(rbt, variable_basic(type_variable(entity_type(e))))) {
    string fn = module_local_name(f);
    entity rv = global_name_to_entity(fn, fn);
    entity orv = entity_undefined;
    Psysteme sc = SC_UNDEFINED;
    Pcontrainte c = CONTRAINTE_UNDEFINED;
    Pvecteur eq = VECTEUR_NUL;
    transformer t_equal = transformer_undefined;

    pips_assert("rv is defined",
		!entity_undefined_p(rv));

    /* Build a transformer reflecting the call site */
    t_caller = user_call_to_transformer(f, pc, ef);

    ifdebug(8) {
      pips_debug(8, "Transformer %p for callee %s:\n",
		 t_caller, entity_local_name(f));
      dump_transformer(t_caller);
    }

    /* Build a transformer representing the equality of
     * the function value to e
     */
    eq = vect_make(eq,
		   (Variable) e, VALUE_ONE,
		   (Variable) rv, VALUE_MONE,
		   TCST, VALUE_ZERO);
    c = contrainte_make(eq);
    sc = sc_make(c, CONTRAINTE_UNDEFINED);
    t_equal = make_transformer(NIL,
			       make_predicate(sc));

    /* Consistency cannot be checked on a non-local transformer */
    /* pips_assert("t_equal is consistent",
       transformer_consistency_p(t_equal)); */

    ifdebug(8) {
      pips_debug(8,
		 "Transformer %p for equality of %s with %s:\n",
		 t_equal, entity_local_name(e), entity_name (rv));
      dump_transformer(t_equal);
    }

    /* Combine the effect of the function call and of the equality */
    t_caller = transformer_combine(t_caller, t_equal);
    free_transformer(t_equal);

    /* Get rid of the temporary representing the function's value */
    orv = global_new_value_to_global_old_value(rv);
    t_caller = transformer_filter(t_caller,
				  CONS(ENTITY, rv, CONS(ENTITY, orv, NIL)));


    ifdebug(8) {
      pips_debug(8,
		 "Final transformer %p for assignment of %s with %s:\n",
		 t_caller, entity_local_name(e), entity_name(rv));
      dump_transformer(t_caller);
    }

    /* FI: e is added in arguments because user_call_to_transformer()
     * uses effects to make sure arrays and non scalar integer variables
     * impact is taken into account
     */
    /*
      transformer_arguments(t_caller) =
      arguments_rm_entity(transformer_arguments(t_caller), e);
    */

    /* FI, FI: il vaudrait mieux ne pas eliminer e d'abord1 */
    /* J'ai aussi des free a decommenter */
    /*
      if(ENDP(transformer_arguments(t_caller))) {
      transformer_arguments(t_caller) = 
      gen_nconc(transformer_arguments(t_caller), CONS(ENTITY, e, NIL));
      }
      else {
      t_caller = transformer_value_substitute(t_caller, rv, e);
      }
    */
    /* Not checkable with temporary variables
       pips_assert("transformer t_caller is consistent", 
       transformer_consistency_p(t_caller));
    */
  }
  else {
    pips_assert("transformer t_caller is undefined", 
		transformer_undefined_p(t_caller));
  }

  gen_free_list(ef);

  pips_debug(8, "end with t_caller=%p\n", t_caller);

    
  return t_caller;
}

/* transformer translation
 */
transformer 
transformer_intra_to_inter(
    transformer tf,
    list le)
{
  cons * lost_args = NIL;
  /* Filtered TransFormer ftf */
  transformer ftf = transformer_dup(tf);
  cons * old_args = transformer_arguments(ftf);
  Psysteme sc = SC_UNDEFINED;
  Pbase b = BASE_UNDEFINED;
  Pbase eb = BASE_UNDEFINED;

  pips_debug(8,"begin\n");
  pips_debug(8,"argument tf=%p\n",ftf);
  ifdebug(8) (void) dump_transformer(ftf);

  /* get rid of tf's arguments that do not appear in effects le */

  /* build a list of arguments to suppress */
  /* FI: I do not understand anymore why corresponding old values do not have
   * to be suppressed too (6 July 1993)
   *
   * FI: because only read arguments are eliminated, non? (12 November 1995)
   */
  MAPL(ca, 
  {entity e = ENTITY(CAR(ca));
  if(!effects_write_entity_p(le, e) &&
     !storage_return_p(entity_storage(e))) 
    lost_args = arguments_add_entity(lost_args, e);
  },
       old_args);

  /* get rid of them */
  ftf = transformer_projection(ftf, lost_args);

  /* free the temporary list of entities */
  gen_free_list(lost_args);
  lost_args = NIL;

  debug(8,"transformer_intra_to_inter","after first filtering ftf=%x\n",ftf);
  ifdebug(8) (void) dump_transformer(ftf);

  /* get rid of local read variables */

  /* FI: why not use this loop to get rid of *all* local variables, read or written? */

  sc = (Psysteme) predicate_system(transformer_relation(ftf));
  b = sc_base(sc);
  for(eb=b; !BASE_UNDEFINED_P(eb); eb = eb->succ) {
    entity e = (entity) vecteur_var(eb);

    if(e != (entity) TCST) {
      entity v = value_to_variable(e);

      /* Variables with no impact on the caller world are eliminated.
       * However, the return value associated to a function is preserved.
       */
      if( ! effects_read_or_write_entity_p(le, v) &&
	  !storage_return_p(entity_storage(v)) &&
	  !entity_constant_p(v)) {
	lost_args = arguments_add_entity(lost_args, e);
      }
    }
  }

  /* get rid of them */
  ftf = transformer_projection(ftf, lost_args);

  /* free the temporary list of entities */
  gen_free_list(lost_args);
  lost_args = NIL;

  debug(8,"transformer_intra_to_inter","return ftf=%x\n",ftf);
  ifdebug(8) (void) dump_transformer(ftf);
  debug(8,"transformer_intra_to_inter","end\n");

  return ftf;
}

/* Effects are necessary to clean up the transformer t_caller. For
   instance, an effect on variable X may not be taken into account in
   t_callee but it may be equivalenced thru a common to a variable i which
   is analyzed in the caller. If X is written, I value is lost. See
   Validation/equiv02.f. */

static transformer 
user_call_to_transformer(
    entity f,
    list pc,
    list ef)
{
  transformer t_callee = transformer_undefined;
  transformer t_caller = transformer_undefined;
  transformer t_effects = transformer_undefined;
  entity caller = entity_undefined;
  list all_args = list_undefined;

  pips_debug(8, "begin\n");
  pips_assert("f is a module", entity_module_p(f));

  if(!get_bool_property(SEMANTICS_INTERPROCEDURAL)) {
    /*
      pips_user_warning(
      "unknown interprocedural transformer for %s\n",
      entity_local_name(f));
    */
    t_caller = effects_to_transformer(ef);
  }
  else {
    /* add equations linking formal parameters to argument expressions
       to transformer t_callee and project along the formal parameters */
    /* for performance, it  would be better to avoid building formals
       and to inline entity_to_formal_parameters */
    /* it wouls also be useful to distinguish between in and out
       parameters; I'm not sure the information is really available
       in a field ??? */
    list formals = module_to_formal_analyzable_parameters(f);
    list formals_new = NIL;
    cons * ce;

    t_callee = load_summary_transformer(f);

    ifdebug(8) {
      Psysteme s = 
	(Psysteme) predicate_system(transformer_relation(t_callee));
      pips_debug(8, "Transformer for callee %s:\n", 
		 entity_local_name(f));
      dump_transformer(t_callee);
      sc_fprint(stderr, s, (char * (*)(Variable)) dump_value_name);
    }

    t_caller = transformer_dup(t_callee);

    /* take care of analyzable formal parameters */

    for( ce = formals; !ENDP(ce); POP(ce)) {
      entity fp = ENTITY(CAR(ce));
      int r = formal_offset(storage_formal(entity_storage(fp)));
      expression expr = find_ith_argument(pc, r);

      if(expr == expression_undefined)
	pips_user_error("not enough args for %d formal parm."
			" %s in call to %s from %s\n",
			r, entity_local_name(fp), entity_local_name(f),
			get_current_module_entity());
      else {
	/* type checking. You already know that fp is a scalar variable */
	type tfp = entity_type(fp);
	basic bfp = variable_basic(type_variable(tfp));
	basic bexpr = basic_of_expression(expr);

	if(!basic_equal_p(bfp, bexpr)) {
	  pips_user_warning("Type incompatibility (formal %s/ actual %s)"
			    " for formal parameter %s (rank %d)"
			    " in call to %s from %s\n",
			    basic_to_string(bfp), basic_to_string(bexpr),
			    entity_local_name(fp), r, module_local_name(f),
			    get_current_module_entity());
	  continue;
	}
      }

      if(entity_is_argument_p(fp, transformer_arguments(t_callee))) {
	/* formal parameter e is modified. expr must be a reference */
	syntax sexpr = expression_syntax(expr);

	if(syntax_reference_p(sexpr)) {
	  entity ap = reference_variable(syntax_reference(sexpr));

	  if(entity_has_values_p(ap)) {
	    Psysteme s = (Psysteme) predicate_system(transformer_relation(t_caller));
	    entity ap_new = entity_to_new_value(ap);
	    entity ap_old = entity_to_old_value(ap);

	    if(base_contains_variable_p(s->base, (Variable) ap_new)) {
	      pips_user_error(
			      "Variable %s seems to be aliased thru variable %s"
			      " at a call site to %s in %s\n"
			      "PIPS semantics analysis assumes no aliasing as"
			      " imposed by the Fortran standard.\n",
			      entity_name(fp),
			      entity_name(value_to_variable(ap_new)),
			      module_local_name(f),
			      get_current_module_name());
	    }
	    else { /* normal case: ap_new==fp_new, ap_old==fp_old */
	      entity fp_new = external_entity_to_new_value(fp);
	      entity fp_old = external_entity_to_old_value(fp);

	      t_caller = transformer_value_substitute
		(t_caller, fp_new, ap_new);
	      t_caller = transformer_value_substitute
		(t_caller, fp_old, ap_old);
	    }
	  }
	  else { /* Variable ap is not analyzed. The information about fp
                    will be lost. */
	    ;
	  }
	}
	else {
	  /* Attemps at modifying a value: expr is call, fp is modified */
	  /* Actual argument is not a reference: it might be a user error!
	   * Transformers do not carry the may/must information.
	   * A check with effect list ef should be performed...
	   *
	   * FI: does effect computation emit a MUST/MAYwarning?
	   */
	  entity fp_new = external_entity_to_new_value(fp);
	  entity fp_old = external_entity_to_old_value(fp);
	  list args = arguments_add_entity(arguments_add_entity(NIL, fp_new), fp_old);
			
	  pips_user_warning("value (!) might be modified by call to %s\n"
			    "%dth formal parameter %s\n",
			    entity_local_name(f), r, entity_local_name(fp));
	  t_caller = transformer_filter(t_caller, args);
	  free_arguments(args);
	}
      }
      else {
	/* Formal parameter fp is not modified. Add fp == expr, if possible. */
	/* We should evaluate expr under a precondition pre... which has
	   not been passed down. We set pre==tf_undefined. */
	entity fp_new = external_entity_to_new_value(fp);
	transformer t_expr = any_expression_to_transformer(fp_new, expr,
							   transformer_undefined,
							   FALSE);

	if(!transformer_undefined_p(t_expr)) {
	  t_expr = transformer_temporary_value_projection(t_expr);
	  /* temporary value counter cannot be reset because other
             temporary values may be in use in a case the user call is a
             user function call */
	  /* reset_temporary_value_counter(); */
	  t_caller = transformer_safe_image_intersection(t_caller, t_expr);
	  free_transformer(t_expr);
	}
      }
    }
  
    pips_debug(8, "Before formal new values left over are eliminated\n");
    ifdebug(8)   dump_transformer(t_caller);
	

    /* formal new and old values left over are eliminated */
    MAPL(ce,{entity e = ENTITY(CAR(ce));
    entity e_new = external_entity_to_new_value(e); 
    formals_new = CONS(ENTITY, e_new, formals_new);
    /* test to insure that entity_to_old_value exists */
    if(entity_is_argument_p(e_new, 
			    transformer_arguments(t_caller))) {
      entity e_old = external_entity_to_old_value(e);
      formals_new = CONS(ENTITY, e_old, formals_new);
    }},
	 formals);
		 
    t_caller = transformer_filter(t_caller, formals_new);
		 
    free_arguments(formals_new);
    free_arguments(formals);
  }

  ifdebug(8) {
    Psysteme s = predicate_system(transformer_relation(t_caller));
    pips_debug(8,
	       "After binding formal/real parameters and eliminating formals\n");
    dump_transformer(t_caller);
    sc_fprint(stderr, s, (char * (*)(Variable)) dump_value_name);
  }

  /* take care of global variables */
  caller = get_current_module_entity();
  translate_global_values(caller, t_caller);

  /* FI: are invisible variables taken care of by translate_global_values()? 
   * Yes, now...
   * A variable may be invisible because its location is reached
   * thru an array or thru a non-integer scalar variable in the
   * current module, for instance because a COMMON is defined
   * differently. A variable whose location is not reachable
   * in the current module environment is considered visible.
   */

  ifdebug(8) {
    pips_debug(8, "After replacing global variables\n");
    dump_transformer(t_caller);
  }

  /* Callee f may have read/write effects on caller's scalar
   * integer variables thru an array and/or non-integer variables.
   */
  t_effects = effects_to_transformer(ef);
  all_args = arguments_union(transformer_arguments(t_caller),
			     transformer_arguments(t_effects));
  /*
    free_transformer(t_effects);
    gen_free_list(transformer_arguments(t_caller));
  */
  transformer_arguments(t_caller) = all_args;
  /* The relation basis must be updated too */
  MAP(ENTITY, v, {
    Psysteme sc = (Psysteme) predicate_system(transformer_relation(t_caller));
    sc_base_add_variable(sc, (Variable) v);
  }, transformer_arguments(t_effects));
    

  ifdebug(8) {
    pips_debug(8,
	       "End: after taking all scalar effects in consideration %p\n",
	       t_caller);
    dump_transformer(t_caller);
  }

  /* pips_assert("transformer t_caller is consistent", transformer_consistency_p(t_caller)); 
   */


  return t_caller;
}

transformer
transformer_add_identity(transformer tf, entity v)
{
  entity v_new = entity_to_new_value(v);
  entity v_old = entity_to_old_value(v);
  Pvecteur eq = vect_new((Variable) v_new, (Value) 1);

  vect_add_elem(&eq, (Variable) v_old, (Value) -1);
  tf = transformer_equality_add(tf, eq);
  transformer_arguments(tf) = 
    arguments_add_entity(transformer_arguments(tf), v_new);

  return tf;
}


/* transformer assigned_expression_to_transformer(entity e, expression
 * expr, list ef): returns a transformer abstracting the effect of
 * assignment e = expr when possible, transformer_undefined otherwise.
 *
 * Note: it might be better to distinguish further between e and expr
 * and to return a transformer stating that e is modified when e
 * is accepted for semantics analysis.
 *
 */
transformer 
assigned_expression_to_transformer(
    entity v,
    expression expr)
{
  transformer tf = transformer_undefined;

  pips_debug(8, "begin\n");

  if(entity_has_values_p(v)) {
    entity v_new = entity_to_new_value(v);
    entity v_old = entity_to_old_value(v);
    entity tmp = make_local_temporary_value_entity(entity_type(v));
    list tf_args = CONS(ENTITY, v, NIL);

    tf = any_expression_to_transformer(tmp, expr, transformer_undefined, TRUE);
    reset_temporary_value_counter();
    if(!transformer_undefined_p(tf)) {
      tf = transformer_value_substitute(tf, v_new, v_old);
      tf = transformer_value_substitute(tf, tmp, v_new);
      tf = transformer_temporary_value_projection(tf);
      transformer_arguments(tf) = tf_args;
    }
  }
  else {
    /* vect_rm(ve); */
    tf = transformer_undefined;
  }

  pips_debug(8, "end with tf=%p\n", tf);

  return tf;
}

transformer integer_assign_to_transformer(expression lhs,
					  expression rhs,
					  list ef) /* effects of assign */
{
  /* algorithm: if lhs and rhs are linear expressions on scalar integer
     variables, build the corresponding equation; else, use effects ef
       
     should be extended to cope with constant integer division as in
     N2 = N/2
     because it is used in real program; inequalities should be
     generated in that case 2*N2 <= N <= 2*N2+1
       
     same remark for MOD operator
       
     implementation: part of this function should be moved into
     transformer.c
  */

  transformer tf = transformer_undefined;
  normalized n = NORMALIZE_EXPRESSION(lhs);

  pips_debug(8,"begin\n");

  if(normalized_linear_p(n)) {
    Pvecteur vlhs = (Pvecteur) normalized_linear(n);
    entity e = (entity) vecteur_var(vlhs);

    if(entity_has_values_p(e) /* && integer_scalar_entity_p(e) */) {
      /* FI: the initial version was conservative because
       * only affine scalar integer assignments were processed
       * precisely. But non-affine operators and calls to user defined
       * functions can also bring some information as soon as
       * *some* integer read or write effect exists
       */
      /* check that *all* read effects are on integer scalar entities */
      /*
	if(integer_scalar_read_effects_p(ef)) {
	tf = assigned_expression_to_transformer(e, rhs, ef);
	}
      */
      /* Check that *some* read or write effects are on integer 
       * scalar entities. This is almost always true... Let's hope
       * assigned_expression_to_transformer() returns quickly for array
       * expressions used to initialize a scalar integer entity.
       */
      if(some_integer_scalar_read_or_write_effects_p(ef)) {
	tf = assigned_expression_to_transformer(e, rhs);
      }
    }
  }
  /* if some condition was not met and transformer derivation failed */
  if(tf==transformer_undefined)
    tf = effects_to_transformer(ef);

  pips_debug(6,"return tf=%lx\n", (unsigned long)tf);
  ifdebug(6) (void) print_transformer(tf);
  pips_debug(8,"end\n");
  return tf;
}

transformer any_scalar_assign_to_transformer(entity v,
					     expression rhs,
					     list ef, /* effects of assign */
					     transformer pre) /* precondition */
{
  transformer tf = transformer_undefined;

  if(entity_has_values_p(v)) {
    entity v_new = entity_to_new_value(v);
    entity v_old = entity_to_old_value(v);
    entity tmp = make_local_temporary_value_entity(entity_type(v));

    tf = any_expression_to_transformer(tmp, rhs, pre, TRUE);

    if(!transformer_undefined_p(tf)) {

      pips_debug(9, "A transformer has been obtained:\n");
      ifdebug(9) dump_transformer(tf);

      if(entity_is_argument_p(v, transformer_arguments(tf))) {
	/* Is it standard compliant? The assigned variable is modified by the rhs. */
	transformer teq = simple_equality_to_transformer(v, tmp, TRUE);
	string s = words_to_string(words_syntax(expression_syntax(rhs)));

	pips_user_warning("Variable %s in lhs is uselessly updated by the rhs '%s'\n",
			  entity_local_name(v), s);

	free(s);

	tf = transformer_combine(tf, teq);
	free_transformer(teq);
      }
      else {
	/* Take care of aliasing */
	entity v_repr = value_to_variable(v_new);

	/* tf = transformer_value_substitute(tf, v_new, v_old); */
	tf = transformer_value_substitute(tf, v_new, v_old);

	pips_debug(9,"After substitution v_new=%s -> v_old=%s\n",
	      entity_local_name(v_new), entity_local_name(v_old));

	tf = transformer_value_substitute(tf, tmp, v_new);

	pips_debug(9,"After substitution tmp=%s -> v_new=%s\n",
	      entity_local_name(tmp), entity_local_name(v_new));

	transformer_arguments(tf) = arguments_add_entity(transformer_arguments(tf), v_repr);
      }
    }
    if(!transformer_undefined_p(tf)) {
      tf = transformer_temporary_value_projection(tf);
      pips_debug(9, "After temporary value projection, tf=%p:\n", tf);
      ifdebug(9) dump_transformer(tf);
    }
    reset_temporary_value_counter();
  }

  if(transformer_undefined_p(tf))
    tf = effects_to_transformer(ef);

  return tf;
}

transformer any_assign_to_transformer(list args, /* arguments for assign */
				      list ef, /* effects of assign */
				      transformer pre) /* precondition */
{
  transformer tf = transformer_undefined;
  expression lhs = EXPRESSION(CAR(args));
  expression rhs = EXPRESSION(CAR(CDR(args)));
  syntax slhs = expression_syntax(lhs);

  pips_assert("2 args to assign", CDR(CDR(args))==NIL);

  /* The lhs must be a scalar reference to perform an interesting analysis */
  if(syntax_reference_p(slhs)) {
    reference rlhs = syntax_reference(slhs);
    if(ENDP(reference_indices(rlhs))) {
      entity v = reference_variable(rlhs);
      tf = any_scalar_assign_to_transformer(v, rhs, ef, pre); 
    }
  }

  /* if some condition was not met and transformer derivation failed */
  if(tf==transformer_undefined)
    tf = effects_to_transformer(ef);

  pips_debug(6,"return tf=%p\n", tf);
  ifdebug(6) (void) print_transformer(tf);
  pips_debug(8,"end\n");
  return tf;
}

static transformer 
instruction_to_transformer(i, e)
instruction i;
cons * e; /* effects associated to instruction i */
{
  transformer tf = transformer_undefined;
  test t;
  loop l;
  call c;
  whileloop wl;

  debug(8,"instruction_to_transformer","begin\n");

  switch(instruction_tag(i)) {
  case is_instruction_block:
    tf = block_to_transformer(instruction_block(i));
    break;
  case is_instruction_test:
    t = instruction_test(i);
    tf = test_to_transformer(t, e);
    break;
  case is_instruction_loop:
    l = instruction_loop(i);
    tf = loop_to_transformer(l);
    break;
  case is_instruction_whileloop:
    wl = instruction_whileloop(i);
    tf = whileloop_to_transformer(wl, e);
    break;
  case is_instruction_goto:
    pips_error("instruction_to_transformer",
	       "unexpected goto in semantics analysis");
    tf = transformer_identity();
    break;
  case is_instruction_call:
    c = instruction_call(i);
    tf = call_to_transformer(c, e);
    break;
  case is_instruction_unstructured:
    tf = unstructured_to_transformer(instruction_unstructured(i), e);
    break ;
  default:
    pips_error("instruction_to_transformer","unexpected tag %d\n",
	       instruction_tag(i));
  }
  debug(9,"instruction_to_transformer","resultat:\n");
  ifdebug(9) (void) print_transformer(tf);
  debug(8,"instruction_to_transformer","end\n");
  return tf;
}


transformer statement_to_transformer(s)
statement s;
{
  instruction i = statement_instruction(s);
  list e = NIL;
  transformer t;
  transformer te;

  pips_debug(8,"begin for statement %03d (%d,%d)\n",
	     statement_number(s), ORDERING_NUMBER(statement_ordering(s)), 
	     ORDERING_STATEMENT(statement_ordering(s)));

  e = load_cumulated_rw_effects_list(s);
  t = load_statement_transformer(s);

  /* it would be nicer to control warning_on_redefinition */
  if (t == transformer_undefined) {
    t = instruction_to_transformer(i, e);

    /* add array references information */
    if(get_bool_property("SEMANTICS_TRUST_ARRAY_REFERENCES")) {
      transformer_add_reference_information(t, s);
    }

    if(!transformer_consistency_p(t)) {
      int so = statement_ordering(s);
      (void) fprintf(stderr, "statement %03d (%d,%d):\n",
		     statement_number(s),
		     ORDERING_NUMBER(so), ORDERING_STATEMENT(so));
      /* (void) print_transformer(load_statement_transformer(s)); */
      (void) print_transformer(t);
      dump_transformer(t);
      pips_internal_error("Inconsistent transformer detected\n");
    }
    ifdebug(1) {
      pips_assert("Transformer is internally consistent",
		  transformer_internal_consistency_p(t));
    }
    store_statement_transformer(s, t);
  }
  else {
    user_warning("statement_to_transformer","redefinition for statement %03d (%d,%d)\n",
		 statement_number(s), ORDERING_NUMBER(statement_ordering(s)), 
		 ORDERING_STATEMENT(statement_ordering(s)));
    pips_internal_error("transformer redefinition");
  }

  ifdebug(1) {
    int so = statement_ordering(s);
    transformer stf = load_statement_transformer(s);

    (void) fprintf(stderr, "statement %03d (%d,%d), transformer %p:\n",
		   statement_number(s),
		   ORDERING_NUMBER(so), ORDERING_STATEMENT(so),
		   stf);
    (void) print_transformer(stf);
    pips_assert("same pointer", stf==t);
  }


  /* If i is a loop, the expected transformer can be more complex (see
     nga06) because the stores transformer is later used to compute the
     loop body precondition. It cannot take into account the exit
     condition. */
  if(instruction_loop_p(i)) {
    /* likely memory leak:-(. te should be allocated in both test
       branches and freed at call site but I program everything under
       the opposite assumption */
    te = refine_loop_transformer(t, instruction_loop(i));
  }
  else {
    te = t;
  }

  pips_debug(8,"end with t=%p\n", t);

  return te;
}
