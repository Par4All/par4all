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

#define IS_LOWER_BOUND 0
#define IS_UPPER_BOUND 1

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

static void 
unstructured_to_postconditions(
    transformer pre,
    transformer pre_first,
    unstructured u)
{
    cons *blocs = NIL ;
    control ct = unstructured_control(u) ;
    transformer c_pre = transformer_undefined;
    transformer post = transformer_undefined;

    debug(8,"unstructured_to_postconditions","begin\n");

    /* SHARING! Every statement gets a pointer to the same precondition!
     * I do not know if it's good or not but beware the bugs!!!
     */
    /* FI: changed to make free_transformer_mapping possible without 
     * testing sharing.
     *
     * pre and pre_first can or not be used depending on the
     * unstructured structure. They are always duplicated and
     * the caller has to take care of their de-allocation.
     */
    CONTROL_MAP(c, {
	statement st = control_statement(c) ;
	if(c==ct && ENDP(control_predecessors(c)) && statement_test_p(st)) {
	    /* special case for the first node if it has no predecessor */
	    /* and if it is a test, as it always should, at least if */
	    /* unspaghettify has been applied... */
	    /* this is pretty useless and should be generalized to the
	       DAG part of the CFG */
	    c_pre = transformer_dup(pre_first);
	    post = statement_to_postcondition(c_pre, st);
	    transformer_free(post);
	}
	else {
	    c_pre = transformer_dup(pre);
	    post = statement_to_postcondition(c_pre, st);
	    transformer_free(post);
	}
    }, ct, blocs) ;

    gen_free_list(blocs) ;

    debug(8,"unstructured_to_postconditions","end\n");
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
	   unstructured u */
	transformer pre_u = 
	    invariant_wrt_transformer(pre,tf);
	debug(8,"unstructured_to_postcondition",
	      "complex: based on transformer\n");
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
	(void) unstructured_to_postconditions(pre_u, pre, u) ;
	post = transformer_apply(transformer_dup(tf), pre);
	transformer_free(pre_u);
    }

    debug(8,"unstructured_to_postcondition","end\n");

    return post;
}

static transformer
add_loop_skip_condition(transformer pre, loop l)
{
    /* It is assumed that loop l is not entered */
    range r = loop_range(l);
    expression e_lb = range_lower(r);
    expression e_ub = range_upper(r);
    expression e_incr = range_increment(r);
    normalized n_lb = NORMALIZE_EXPRESSION(e_lb);
    normalized n_ub = NORMALIZE_EXPRESSION(e_ub);
    int incr = 0;
    int incr_lb = 0;
    int incr_ub = 0;

    debug(8,"add_loop_skip_condition","begin\n");

    /* is the loop increment numerically known? Is its sign known? */
    expression_and_precondition_to_integer_interval(e_incr, pre, &incr_lb, &incr_ub);

    if(incr_lb==incr_ub) {
	if(incr_lb==0) {
	    user_error("add_loop_skip_condition", "Illegal null loop increment\n");
	}
	else
	    incr = incr_lb;
    }
    else if(incr_lb>=1) {
	incr = 1;
    }
    else if(incr_ub<=-1) {
	incr = -1;
    }
    else
	incr = 0;

    /* incr == 0 is used below as a give-up condition */

    /* find the real upper and lower bounds */
    if(incr<0) {
	/* exchange bounds */
	n_lb = NORMALIZE_EXPRESSION(e_ub);
	n_ub = NORMALIZE_EXPRESSION(e_lb);
    }

    if(incr!=0 && normalized_linear_p(n_lb) && normalized_linear_p(n_ub)) {
	/* ub < lb, i.e. ub + lb + 1 <= 0 */
	Pvecteur v_ub = (Pvecteur) normalized_linear(n_ub);
	Pvecteur v_lb = (Pvecteur) normalized_linear(n_lb);

	if(value_mappings_compatible_vector_p(v_lb)
	   && value_mappings_compatible_vector_p(v_ub)) {
	    Pvecteur v = vect_substract(v_ub, v_lb);

	    vect_add_elem(&v, TCST, (Value) 1);
	    pre = transformer_inequality_add(pre, v);

	    ifdebug(8) {
		debug(8,"add_loop_skip_condition","Skip condition:\n");
		vect_fprint(stderr, v, external_value_name);
	    }
	}
	else {
	    debug(8,"add_loop_skip_condition","Non-analyzed variable in loop bound(s)\n");
	}
    }
    else {
	debug(8,"add_loop_skip_condition","increment sign unknown or non-affine bound\n");
    }

    debug(8,"add_loop_skip_condition","end\n");

    return pre;
}

static transformer 
add_affine_bound_conditions(transformer pre, 
			    entity index,
			    Pvecteur v_bound, 
			    bool lower_or_upper,
			    transformer tfb)
{
    Pvecteur v = vect_dup(v_bound);

    /* check that v is not affected by tfb:
     * N = 10
     * DO I = 1, N
     *   N = 1
     *   {1<=I<=N} !wrong!
     *   T(I) = 0.
     * ENDDO
     * and make sure that aliasings (I,J) and (I,X) are correctly handled
     */

    /* Achtung: value_mappings_compatible_vector_p() has a side
     * effect on its argument; it has to be evaluated before
     * the second half of the test else effects would be wrongly
     * interpreted in case of equivalences 
     */

    if(value_mappings_compatible_vector_p(v) &&
       !transformer_affect_linear_p(tfb,v)) {
	if (lower_or_upper == IS_LOWER_BOUND)
	    vect_add_elem(&v,
			  (Variable) entity_to_new_value(index), VALUE_MONE);
	else{
	    vect_chg_sgn(v);
	    vect_add_elem(&v, 
			  (Variable) entity_to_new_value(index), VALUE_ONE);
	}
	pre = transformer_inequality_add(pre, v);
    }
    else{
	vect_rm(v);
	v = VECTEUR_UNDEFINED;
    }	
    return pre;
}

static transformer 
add_index_bound_conditions(
    transformer pre,
    entity index,
    expression bound,
    int lower_or_upper,
    transformer tfb)
{
    normalized n = NORMALIZE_EXPRESSION(bound);
    /* tfb does not take into account the index incrementation */
    transformer t_iter = transformer_dup(tfb);

    /* It is assumed on entry that index has values recognized 
     * by the semantics analysis
     */
    /* pips_assert("add_index_bound_conditions", entity_has_values_p(index)); */

    transformer_arguments(t_iter) = 
	arguments_add_entity(transformer_arguments(t_iter), index);

    if(normalized_linear_p(n)) {
	Pvecteur v_bound = (Pvecteur) normalized_linear(n);
	add_affine_bound_conditions(pre, index, v_bound, lower_or_upper, t_iter);
    }

    free_transformer(t_iter);
    return(pre);
}

transformer 
add_index_range_conditions(
    transformer pre,
    entity i,
    range r,
    transformer tfb)
{
    /* if tfb is not undefined, then it is a loop;
       loop bounds can be kept as preconditions for the loop body
       if the loop increment is numerically known and if they
       are linear and if they are loop body invariant, i.e.
       indices are accepted */

    expression lb = range_lower(r);
    expression ub = range_upper(r);
    expression e_incr = range_increment(r);
    int incr = 0;
    int incr_lb = 0;
    int incr_ub = 0;

    debug(8,"add_index_range_conditions","begin\n");

    if(entity_has_values_p(i)) {

	/* is the loop increment numerically known? Is its sign known? */
	expression_and_precondition_to_integer_interval(e_incr, pre, &incr_lb, &incr_ub);

	if(incr_lb==incr_ub) {
	    if(incr_lb==0) {
		user_error("add_index_range_conditions", "Illegal null increment\n");
	    }
	    else
		incr = incr_lb;
	}
	else if(incr_lb>=1) {
	    incr = 1;
	}
	else if(incr_ub<=-1) {
	    incr = -1;
	}
	else
	    incr = 0;

	/* incr == 0 is used below as a give-up condition */


	/* find the real upper and lower bounds */
	if(incr<0) {
	    ub = range_lower(r);
	    lb = range_upper(r);
	}

	if(incr!=0) {
	    if(simple_dead_loop_p(lb, ub)) {
		transformer new_pre = transformer_empty();

		free_transformer(pre);
		pre = new_pre;
	    }
	    else {
		/* try to add the lower bound */
		add_index_bound_conditions(pre, i, lb, IS_LOWER_BOUND, tfb);

		/* try to add the upper bound */
		add_index_bound_conditions(pre, i, ub, IS_UPPER_BOUND, tfb);
	    }
	}

    }

    debug(8,"add_index_range_conditions","end\n");
    return pre;
}

static transformer 
add_good_loop_conditions(
    transformer pre,
    loop l,
    transformer tf)
{
    /* loop bounds can be kept as preconditions for the loop body
       if the loop increment is numerically known and if they
       are linear and if they are loop body invariant, i.e.
       indices are accepted */
    /* arg. tf is unused, it was replaced by tfb to correct a bug */
    statement b = loop_body(l);
    entity i = loop_index(l);
    range r = loop_range(l);
    transformer tfb = load_statement_transformer(b);

    debug(8,"add_good_loop_conditions","begin\n");

    pre = add_index_range_conditions(pre, i, r, tfb);

    debug(8,"add_good_loop_conditions","end\n");
    return(pre);
    
}


static transformer
add_loop_index_initialization(transformer pre, loop l)
{
    entity i = loop_index(l);
    expression init = range_lower(loop_range(l));
    transformer post = transformer_undefined;
    transformer t_init = transformer_undefined;
    list lef = proper_effects_of_expression(init);

    t_init = expression_to_transformer(i, init, lef);
    if(t_init==transformer_undefined)
	t_init = effects_to_transformer(lef);
    post = transformer_apply(t_init, pre);

    transformer_free(t_init);

    return post;
}

/* The exit value is known if
 *  - the loop index is an analyzed integer scalar variable
 *    real index are standard-compliant, integer index can be equivalenced to
 *    a real variable,...
 *  - the increment is affine (non-necessary assumption made: it is affine if the
 *    increment sign is known)
 *  - the increment sign is known
 *  - the body and loop initialization execution does not modify 
 *    the value of the upper bound
 *  - the upper bound is affine (?)
 *
 * Affine increments can be handled when their signs only are known. 
 *
 * For instance, for increment==k:
 *  i >= ub
 *  i-k <= ub-1
 *
 * Note the simplification when k==1.
 *
 * But, first of all, the value of the loop index in post must be
 * incremented. This changes its relationship with induction variables.
 *
 * Most tests here are redundant because this function is only called if it has
 * been proved that the loop was executed which implies that the upper and
 * lower bounds are affine, that the increment is affine and that the
 * increment sign is known.
 */
static transformer
add_loop_index_exit_value(
    transformer post, /* postcondition of the last iteration */
    loop l,           /* loop to process */
    transformer pre,  /* precondition on loop entrance */
    list lbe          /* list of loop body effects */ )
{
    entity i = loop_index(l);
    expression e_incr = range_increment(loop_range(l));
    normalized n_incr = NORMALIZE_EXPRESSION(e_incr);
    expression e_ub = range_upper(loop_range(l));
    normalized n_ub = NORMALIZE_EXPRESSION(e_ub);
    transformer t_incr = transformer_undefined;
    transformer t_body = load_statement_transformer(loop_body(l));
    list li = CONS(ENTITY, i, NIL);
    int ub_inc = 0;
    int lb_inc = 0;

    ifdebug(8) {
	debug(8, "add_loop_index_exit_value", "begin with post:\n");
	(void) print_transformer(post);
    }

    if(!entity_has_values_p(i)) {
	ifdebug(8) {
	    debug(8, "add_loop_index_exit_value", "give up because %s has no values:\n",
		  entity_local_name(i));
	    debug(8, "add_loop_index_exit_value", "end with post:\n");
	    (void) print_transformer(post);
	}
	return post;
    }

    expression_and_precondition_to_integer_interval(e_incr, pre, &lb_inc, &ub_inc);

    /* This part should be useless because post really is a loop postcondition.
     * There should be no need for an index incrementation or anything.
     * It is used to recover some of the fix-point failures:-(
     *
     * In fact, it may be useful. The loop transformer and its invariant are
     * computed without knowledge about the number of iteration, which may be
     * zero. By adding one execution of the loop body and the index incrementation,
     * we change b* into b+ which should add effective information.
     */
    if(normalized_linear_p(n_incr)) {
	Pvecteur v_incr = (Pvecteur) normalized_linear(n_incr);
	if(value_mappings_compatible_vector_p(v_incr)) {
	    t_incr = affine_increment_to_transformer(i, v_incr);
	}
	else {
	    t_incr = transformer_undefined;
	}
    }
    else {
	t_incr = transformer_undefined;
    }
    if(t_incr==transformer_undefined)
	t_incr = args_to_transformer(li);
    post = transformer_apply(t_body, post);
    post = transformer_apply(t_incr, post);
    transformer_free(t_incr);

    ifdebug(8) {
	debug(8, "add_loop_index_exit_value", "post after index incrementation:\n");
	(void) print_transformer(post);
    }

    if(normalized_linear_p(n_ub)
       && !transformer_affect_linear_p(t_body, (Pvecteur) normalized_linear(n_ub))) {
	if(lb_inc >= 1 || ub_inc <= -1) {
	    Pvecteur v_ub = (Pvecteur) normalized_linear(n_ub);
	    Pvecteur v_incr = (Pvecteur) normalized_linear(n_incr);

	    if(value_mappings_compatible_vector_p(v_ub)
	       && value_mappings_compatible_vector_p(v_incr)) {
		Pvecteur v_i = vect_new((Variable) i, (Value) 1);
		Pvecteur c1 = VECTEUR_UNDEFINED;
		Pvecteur c2 = VECTEUR_UNDEFINED;

		pips_assert("add_loop_index_exit_value", normalized_linear_p(n_incr));
		if(lb_inc==ub_inc && ABS(lb_inc)==1) {
		    /* v_i - v_incr == v_ub, regardless of the sign */
		    c1 = vect_substract(v_i, v_incr);
		    c1 = vect_cl(c1, (Value) -1, v_ub);
		    transformer_equality_add(post, c1);
		}
		else {
		    if(lb_inc>=1) {
			/* v_i - v_incr <= v_ub < v_i
			 * or:
			 * v_i - v_incr - v_ub <= 0, v_ub - v_i + 1 <= 0
			 */
			c1 = vect_substract(v_i, v_incr);
			c2 = vect_substract(v_ub, v_i);

			c1 = vect_cl(c1, (Value) -1, v_ub);
			vect_add_elem(&c2, (Variable) TCST, (Value) 1);
		    }
		    else if(ub_inc<=-1) {
			/* v_i - v_incr >= v_ub > v_i 
			 *
			 * or:
			 * - v_i + v_incr + v_ub <= 0, - v_ub + v_i + 1 <= 0
			 */
			c1 = vect_substract(v_incr, v_i);
			c2 = vect_substract(v_i, v_ub);

			c1 = vect_cl(c1, (Value) 1, v_ub);
			vect_add_elem(&c2, (Variable) TCST, (Value) 1);
		    }
		    else {
			/* should never happen! */
			pips_assert("add_loop_index_exit_value", TRUE);
		    }
		    transformer_inequality_add(post, c1);
		    transformer_inequality_add(post, c2);
		}

		ifdebug(8) {
		    debug(8, "add_loop_index_exit_value", "post with exit conditions:\n");
		    (void) print_transformer(post);
		}
	    }
	    else {
		debug(8, "add_loop_index_exit_value",
		      "post is unchanged because the increment or the upper bound"
		      " reference unanalyzed variables\n");
	    }
	}
	else {
	    debug(8, "add_loop_index_exit_value",
		  "post is unchanged because the increment sign is unknown\n");
	}
    }
    else {
	debug(8, "add_loop_index_exit_value",
	      "post is unchanged because the upper bound is not affine\n");
    }

    ifdebug(8) {
	debug(8, "add_loop_index_exit_value", "end: post:\n");
	(void) print_transformer(post);
    }

    return post;
}

bool 
simple_dead_loop_p(expression lower, expression upper)
{
    bool dead_loop_p = FALSE;
    normalized n_lower = NORMALIZE_EXPRESSION(lower);
    normalized n_upper = NORMALIZE_EXPRESSION(upper);

    if(normalized_linear_p(n_upper) && normalized_linear_p(n_lower)) {
	Pvecteur v_lower = normalized_linear(n_lower);
	Pvecteur v_upper = normalized_linear(n_upper);
	
	if(VECTEUR_NUL_P(v_lower)) {
	    if (!VECTEUR_NUL_P(v_upper)) {
		if(term_cst(v_upper)
		   && VECTEUR_NUL_P(vecteur_succ(v_upper))) {
		    dead_loop_p =  value_neg_p(vecteur_val(v_upper));
		}
	    }
	}
	else if(VECTEUR_NUL_P(v_upper)) {
	    if (!VECTEUR_NUL_P(v_lower)) {
		if(term_cst(v_lower)
		   && VECTEUR_NUL_P(vecteur_succ(v_lower))) {
		    dead_loop_p =  value_pos_p(vecteur_val(v_lower));
		}
	    }
	}
	else if(term_cst(v_upper) && term_cst(v_lower)
	   && VECTEUR_NUL_P(vecteur_succ(v_upper))
	   && VECTEUR_NUL_P(vecteur_succ(v_lower))) {
	    dead_loop_p = 
		value_gt(vecteur_val(v_lower),vecteur_val(v_upper));
	}
    }

    return dead_loop_p;
}

static transformer 
loop_to_postcondition(
    transformer pre,
    loop l,
    transformer tf)
{
    transformer post = transformer_undefined;
    statement s = loop_body(l);
    range r = loop_range(l);

    debug(8,"loop_to_postcondition","begin\n");

    if(pips_flag_p(SEMANTICS_FIX_POINT) && pips_flag_p(SEMANTICS_INEQUALITY_INVARIANT)) {
	pips_error("loop_to_postcondition","Halbwachs not implemented\n");
    }
    else {
	/* pips_error("loop_to_postcondition",
	   "Equality option not implemented\n"); */
	/* the standard version should be OK for SEMANTICS_EQUALITY_INVARIANTS... */

	/* basic cheap version: add information on loop index in pre
	   and propagate preb downwards in the loop body; compute the
	   loop postcondition independently using the loop transformer */
	/* preb = precondition for loop body; includes a lousy fix-point */
	transformer preb = transformer_dup(pre);

	/* Get rid of information related to variables modified in
	 * iterations of the loop body (including loop indices).
	 *
	 * Apparently, the loop index incrementation is not in tf...
	 * according to a test with DO I = I, N
	 * although it should be... according to the caller of this function
	 * and the display of loop transformers!
	 */
	preb = transformer_combine(preb, tf);

	/* Triolet's good loop algorithm */
	preb = add_good_loop_conditions(preb, l, tf);

	/* It might be useful to normalize preb and to detect unfeasibility.
	 * I choose not to do it because:
	 *  - it almost never happens in user code
	 *  - it's time consuming
	 *  - when it happens in automatically generated code,
	 *    dead code elimination is performed.
	 *
	 * So basically, I shift the burden from precondition computation
	 * to dead code elimination. Dead loop testing must use a strong
	 * feasibility test.
	 *
	 * If this decision is reversed, dead code elimination can be
	 * speeded-up.
	 *
	 * Note that add_good_loop_conditions() can test trivial cases
	 * automatically generated.
	 *
	 * Decision reverted: Francois Irigoin, 4 June 1997
	 */

	/* FI: this is not correct when an invariant is found; we should add one
	 * more incrementation of I (to modify the output of the invariant and
	 * express the fact that the loop bound is no more true, at least when
	 * the increment is one. 6 July 1993
	 *
	 * Note 1: this comments was wrong! See Validation/Semantics/induc1.f
	 * Note 2: but the result was wrong anyway because it assumed that at
	 * least one iteration always was performed.
	 * Note 3: This is fixed by reverting the above decision.
	 */
	if(empty_range_wrt_precondition_p(r, pre)) {
	    debug(8, "loop_to_postcondition", "The loop is never executed\n");

	    /* propagate an impossible precondition in the loop body */
	    (void) statement_to_postcondition(transformer_empty(), s);
	    /* The loop body precondition is not useful any longer */
	    free_transformer(preb);
	    post = transformer_dup(pre);
	    post = add_loop_index_initialization(post, l);
	}
	else if(non_empty_range_wrt_precondition_p(r, pre)) {
	    debug(8, "loop_to_postcondition", "The loop certainly is executed\n");

	    /* propagate preconditions in the loop body */
	    (void) statement_to_postcondition(preb, s);

	    post = transformer_apply(tf, pre);
	    /* The loop body effects should be passed as fourth argument
	     * to check that the value of the upper bound expression is
	     * not modified when the body is executed. But it is not available
	     * and it is not yet used by ad_loop_index_exit_value()!
	     */
	    post = add_loop_index_exit_value(post, l, pre, NIL);
	}
	else {
	    /* First version: OK, but could be better! */
	    /* transformer postloop = transformer_apply(tf, pre); */
	    /* pre must be copied because sc_convex_hull updates is arguments
	     * and may make them inconsistent when the two bases are merged
	     * to perform the convex hull in a common vector space
	     */
	    /*
	    transformer preloop = transformer_dup(pre);
	    debug(8, "loop_to_postcondition", "The loop may be executed or not\n");

	    post = transformer_convex_hull(postloop, preloop);
	    transformer_free(postloop);
	    transformer_free(preloop);
	    */

	    /* Second version: assume it is empty or non-empty and perform
	     * the convex hull of both
	     * (should be checked on Validation/Semantics/induc1.f)
	     */
	    transformer post_ne = transformer_dup(pre);
	    transformer post_al = transformer_apply(tf, pre);

	    debug(8, "loop_to_postcondition", "The loop may be executed or not\n");

	    /* propagate preconditions in the loop body */
	    (void) statement_to_postcondition(preb, s);

	    /* We should add (when possible) the non-entry condition in post_ne!
	     * For instance, DO I = 1, N leads to N <= 0
	     */
	    post_ne = add_loop_skip_condition(post_ne, l);

	    post_ne = add_loop_index_initialization(post_ne, l);
	    post_al = add_loop_index_exit_value(post_al, l, pre, NIL);
	    post = transformer_convex_hull(post_ne, post_al);
	    transformer_free(post_ne);
	    transformer_free(post_al);
	}

	/* FI: Cannot be freed because it was stored for statement s:
	   transformer_free(preb);
	   did I mean to free pre, which is useless since it was changed into
	   preb?
	   */
    }

    ifdebug(8) {
	(void) fprintf(stderr,"%s: %s\n","[loop_to_postcondition]",
		       "resultat post =");
	(void) print_transformer(post);
    }
    debug(8,"loop_to_postcondition","end\n");
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

    debug(8,"call_to_postcondition","begin\n");

    switch (tt = value_tag(entity_initial(e))) {
      case is_value_intrinsic:
	/* there is room for improvement because assign is now the only 
	   well handled intrinsic */
	debug(5, "call_to_postcondition", "intrinsic function %s\n",
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
	debug(5, "call_to_postcondition", "external function %s\n",
	      entity_name(e));
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
	pips_error("call_to_postcondition", 
		   "call to symbolic or constant %s\n",
		   entity_name(e));
	break;
      case is_value_unknown:
	pips_error("call_to_postcondition", "unknown function %s\n",
		   entity_name(e));
	break;
      default:
	pips_error("call_to_postcondition", "unknown tag %d\n", tt);
    }

    debug(8,"call_to_postcondition","end\n");

    return(post);
}

/* Non-convex information can be made convexe when moving postcondition downwards
 * because some of the parts introducing non-convexity may be made empty by
 * the pre-existing precondition.
 *
 * This is not true for transformers computed upwards, although conditions on
 * initial values could also be added. Let's wait for a case which would prove
 * this useful.
 *
 * Argument pre is modified and may or not be returned as newpre. It may also be
 * freed and newpre allocated. In other words, argument "pre" should not be used
 * after a call to this function.
 */
static transformer 
transformer_add_condition_information_updown(
    transformer pre,
    expression c,
    bool veracity,
    bool upwards)
{
#   define DEBUG_TRANSFORMER_ADD_CONDITION_INFORMATION_UPDOWN 7
    /* default option: no condition can be added */
    transformer newpre = pre;

    /* check first that c's effects are purely reads on integer scalar
       variable; I'm not sure I'm reusing Remi's function well... */
    cons * ef = proper_effects_of_expression(c);

    ifdebug(DEBUG_TRANSFORMER_ADD_CONDITION_INFORMATION_UPDOWN) {
	debug(DEBUG_TRANSFORMER_ADD_CONDITION_INFORMATION_UPDOWN,
	      "transformer_add_condition_information_updown", 
	      "begin upwards=%s veracity=%s c=", 
	      bool_to_string(upwards), bool_to_string(veracity));
	print_expression(c);
	(void) fprintf(stderr,"pre=");
	print_transformer(pre);
    }

    if(integer_scalar_read_effects_p(ef)) {
	syntax s = expression_syntax(c);
	if(syntax_call_p(s)) {
	    entity e = call_function(syntax_call(s));
	    /* do not factor out c1/e1 and c2/e2 initialization; they do not
	       exist for all calls */
	    if(ENTITY_RELATIONAL_OPERATOR_P(e)) {
		expression e1 = 
		    EXPRESSION(CAR(call_arguments(syntax_call(s))));
		expression e2 = 
		    EXPRESSION(CAR(CDR(call_arguments(syntax_call(s)))));
		newpre = transformer_add_relation_information(pre, e, e1, e2, 
							      veracity, upwards);
	    }
	    else if(ENTITY_AND_P(e)) {
		/* call transformer_add_condition_information recursively
		 * on both sub-expressions if veracity == TRUE; else
		 * try your best...
		 */
		if(veracity) {
		    /* FI: this is a bit confusing because of aliasing conditions.
		     * The second condition is added after the first one was analyzed
		     * and the "and" effect is enforced by side effect
		     *
		     * This is equivalent to allocating copies of pre and then computing
		     * their intersection.
		     */
		    transformer pre1 = pre;
		    transformer pre2 = pre;
		    expression c1 = 
			EXPRESSION(CAR(call_arguments(syntax_call(s))));
		    expression c2 = 
			EXPRESSION(CAR(CDR(call_arguments(syntax_call(s)))));
		    pre1 = transformer_add_condition_information_updown
			(pre1, c1, veracity, upwards);
		    pre2 = transformer_add_condition_information_updown
			(pre2, c2, veracity, upwards);
		    newpre = pre2;
		}
		else if(!upwards) {
		    /* compute !(c1&&c2) as !c1 || !c2 */
		    transformer pre1 = transformer_dup(pre);
		    transformer pre2 = pre;
		    expression c1 = 
			EXPRESSION(CAR(call_arguments(syntax_call(s))));
		    expression c2 = 
			EXPRESSION(CAR(CDR(call_arguments(syntax_call(s)))));
		    pre1 = transformer_add_condition_information_updown
			(pre1, c1, veracity, upwards);
		    pre2 = transformer_add_condition_information_updown
			(pre2, c2, veracity, upwards);
		    newpre = transformer_convex_hull(pre1, pre2);

		    ifdebug(DEBUG_TRANSFORMER_ADD_CONDITION_INFORMATION_UPDOWN) {
			debug(DEBUG_TRANSFORMER_ADD_CONDITION_INFORMATION_UPDOWN, "transformer_add_condition_information_updown", "pre1 =\n");
			(void) (void) print_transformer(pre1);
			debug(DEBUG_TRANSFORMER_ADD_CONDITION_INFORMATION_UPDOWN, "transformer_add_condition_information_updown", "pre2 =\n");
			(void) (void) print_transformer(pre2);
			debug(DEBUG_TRANSFORMER_ADD_CONDITION_INFORMATION_UPDOWN, "transformer_add_condition_information_updown", "newpre =\n");
			(void) (void) print_transformer(newpre);
		    }
		}
		else {
		    /* might be possible to add a convex hull on the initial values
		     * as in !upwards
		     */
		    ;
		}
	    }
	    else if(ENTITY_OR_P(e)) {
		/* call transformer_add_condition_information recursively
		 * on both sub-expressions if veracity == FALSE; else
		 * try to do your best...
		 *
		 * Should be fused with the .AND. case?!? Careful with veracity...
		 */
		if(!veracity) {
		    /* compute !(c1||c2) as !c1 && !c2 */
		    expression c1 = 
			EXPRESSION(CAR(call_arguments(syntax_call(s))));
		    expression c2 = 
			EXPRESSION(CAR(CDR(call_arguments(syntax_call(s)))));
		    newpre = transformer_add_condition_information_updown
			(pre, c1, veracity, upwards);
		    newpre = transformer_add_condition_information_updown
			(newpre, c2, veracity, upwards);
		}
		else if(!upwards) {
		    /* compute (c1||c2) as such */
		    transformer pre1 = transformer_dup(pre);
		    transformer pre2 = pre;
		    expression c1 = 
			EXPRESSION(CAR(call_arguments(syntax_call(s))));
		    expression c2 = 
			EXPRESSION(CAR(CDR(call_arguments(syntax_call(s)))));
		    pre1 = transformer_add_condition_information_updown
			(pre1, c1, veracity, upwards);
		    pre2 = transformer_add_condition_information_updown
			(pre2, c2, veracity, upwards);
		    newpre = transformer_convex_hull(pre1, pre2);

		    /* free_transformer(pre1); */
		    /* free_transformer(pre2); */

		    ifdebug(DEBUG_TRANSFORMER_ADD_CONDITION_INFORMATION_UPDOWN) {
			debug(DEBUG_TRANSFORMER_ADD_CONDITION_INFORMATION_UPDOWN,
			      "transformer_add_condition_information_updown", "pre1 =\n");
			(void) (void) print_transformer(pre1);
			debug(DEBUG_TRANSFORMER_ADD_CONDITION_INFORMATION_UPDOWN,
			      "transformer_add_condition_information_updown", "pre2 =\n");
			(void) print_transformer(pre2);
			debug(DEBUG_TRANSFORMER_ADD_CONDITION_INFORMATION_UPDOWN,
			      "transformer_add_condition_information_updown", "newpre =\n");
			(void) (void) print_transformer(newpre);
		    }
		}
		else {
		    /* might be possible to add a convex hull on the initial values
		     * as in !upwards
		     */
		    newpre = pre;
		}
	    }
	    else if(ENTITY_NOT_P(e)) {
		    expression c1 = 
			EXPRESSION(CAR(call_arguments(syntax_call(s))));
		    newpre = transformer_add_condition_information_updown
			(pre, c1, !veracity, upwards);
	    }
	    else if((ENTITY_TRUE_P(e) && !veracity) ||
		    (ENTITY_FALSE_P(e) && veracity)) {
		free_transformer(pre);
		newpre = transformer_empty();
	    }
	    else
		/* do not know what to do with other logical operators, for the time being! 
		 * keep pre unmodified
		 */
		newpre = pre;
	}
	else
	    pips_error("transformer_add_condition_information_updown",
		       "ill. expr. as test condition\n");
    }
    /* Help! I need a free_effects(ef) here */

    ifdebug(DEBUG_TRANSFORMER_ADD_CONDITION_INFORMATION_UPDOWN) {
	debug(DEBUG_TRANSFORMER_ADD_CONDITION_INFORMATION_UPDOWN,
	      "transformer_add_condition_information_updown", "end newpre=\n");
	print_transformer(newpre);
    }

    return newpre;
}

transformer 
transformer_add_condition_information(
    transformer pre,
    expression c,
    bool veracity)
{
    transformer post = 
	transformer_add_condition_information_updown(pre, c, veracity, TRUE);
    return post;
}

transformer
precondition_add_condition_information(
    transformer pre,
    expression c,
    bool veracity)
{
    transformer post = 
	transformer_add_condition_information_updown(pre, c, veracity, FALSE);
    return post;
}

/* Renaming of variables in v according to transformations occuring
 * later. If a variable is modified by post, its old value must
 * be used in v
 */

static void
upwards_vect_rename(Pvecteur v, transformer post)
{
    /* FI: it would probably ne more efficient to
     * scan va and vb than the argument list...
     */
    list modified_values = transformer_arguments(post);
    
    MAP(ENTITY, v_new, {
	entity v_init = new_value_to_old_value(v_new);
	
	(void) vect_variable_rename(v, (Variable) v_new,
				    (Variable) v_init);
    }, modified_values);
}

transformer 
transformer_add_relation_information(
    transformer pre,
    entity relop,
    expression e1,
    expression e2,
    bool veracity,
    bool upwards)
{
#   define DEBUG_TRANSFORMER_ADD_RELATION_INFORMATION 7
    /* default: no change */
    transformer newpre = pre;
    /* both expressions e1 and e2 must be affine */
    normalized n1 = NORMALIZE_EXPRESSION(e1);
    normalized n2 = NORMALIZE_EXPRESSION(e2);

    ifdebug(DEBUG_TRANSFORMER_ADD_RELATION_INFORMATION) {
	debug(DEBUG_TRANSFORMER_ADD_RELATION_INFORMATION,"transformer_add_relation_information", 
	      "begin upwards=%s veracity=%s relop=%s e1=", 
	      bool_to_string(upwards), bool_to_string(veracity), entity_local_name(relop));
	print_expression(e1);
	(void) fprintf(stderr,"e2=");
	print_expression(e2);
	(void) fprintf(stderr,"pre=");
	print_transformer(pre);
    }

    if(normalized_linear_p(n1) && normalized_linear_p(n2)) {
	Pvecteur v1 = (Pvecteur) normalized_linear(n1);
	Pvecteur v2 = (Pvecteur) normalized_linear(n2);

	if((ENTITY_EQUAL_P(relop) && veracity) ||
	   (ENTITY_NON_EQUAL_P(relop) && !veracity)) {
	    /* v1 - v2 == 0 */
	    Pvecteur v = vect_substract(v1, v2);
	    if(upwards) {
		upwards_vect_rename(v, pre);
	    }
	    newpre = transformer_equality_add(pre, v);
	}
	else if((ENTITY_EQUAL_P(relop) && !veracity) ||
	   (ENTITY_NON_EQUAL_P(relop) && veracity)) {
	    /* v2 - v1 + 1 <= 0 ou v1 - v2 + 1 <= 0 */
	    /* FI: I do not know if this is valid when your are moving upwards
	     * variables in v2 and v1 may have to be renamed as #init values (i.e. old values)
	     */
	    transformer prea = transformer_dup(pre);
	    transformer preb = pre;
	    Pvecteur va = vect_substract(v2, v1);
	    Pvecteur vb = vect_substract(v1, v2);
	    vect_add_elem(&va, TCST, VALUE_ONE);
	    vect_add_elem(&vb, TCST, VALUE_ONE);
	    /* FI: I think that this should be programmed (see comment above)
	     * but I'm waiting for a bug to occur... (6 July 1993)
	     *
	     * FI: Well, the bug was eventually seen:-) (8 November 1995)
	     */
	    if(upwards) {
		upwards_vect_rename(va, pre);
		upwards_vect_rename(vb, pre);
	    }
	    prea = transformer_inequality_add(prea, va);
	    preb = transformer_inequality_add(preb, vb);
	    newpre = transformer_convex_hull(prea, preb);
	    /* free_transformer(prea); */
	    /* free_transformer(preb); */
	}
	else if ((ENTITY_GREATER_THAN_P(relop) && veracity) ||
		 (ENTITY_LESS_OR_EQUAL_P(relop) && !veracity)) {
	    /* v2 - v1 + 1 <= 0 */
	    Pvecteur v = vect_substract(v2, v1);
	    vect_add_elem(&v, TCST, VALUE_ONE);
	    if(upwards) {
		upwards_vect_rename(v, pre);
	    }
	    newpre = transformer_inequality_add(pre, v);
	}
	else if ((ENTITY_GREATER_THAN_P(relop) && !veracity) ||
		 (ENTITY_LESS_OR_EQUAL_P(relop) && veracity)) {
	    /* v1 - v2 <= 0 */
	    Pvecteur v = vect_substract(v1, v2);
	    if(upwards) {
		upwards_vect_rename(v, pre);
	    }
	    newpre = transformer_inequality_add(pre, v);
	}
	else if ((ENTITY_GREATER_OR_EQUAL_P(relop) && veracity) ||
		 (ENTITY_LESS_THAN_P(relop) && !veracity)) {
	    /* v2 - v1 <= 0 */
	    Pvecteur v = vect_substract(v2, v1);
	    if(upwards) {
		upwards_vect_rename(v, pre);
	    }
	    newpre = transformer_inequality_add(pre, v);
	}
	else if ((ENTITY_GREATER_OR_EQUAL_P(relop) && !veracity) ||
		 (ENTITY_LESS_THAN_P(relop) && veracity)) {
	    /* v1 - v2 + 1 <= 0 */
	    Pvecteur v = vect_substract(v1, v2);
	    vect_add_elem(&v, TCST, VALUE_ONE);
	    if(upwards) {
		upwards_vect_rename(v, pre);
	    }
	    newpre = transformer_inequality_add(pre, v);
	}
	else
	    /* do nothing... although Malik may have tried harder! */
	    newpre = pre;
    }
    else
	/* do nothing, although MODULO and INTEGER DIVIDE could be handled */
	newpre = pre;

    ifdebug(DEBUG_TRANSFORMER_ADD_RELATION_INFORMATION) {
	debug(DEBUG_TRANSFORMER_ADD_RELATION_INFORMATION,
	      "transformer_add_relation_information", "end newpre=\n");
	print_transformer(newpre);
    }

    return newpre;
}

transformer 
data_to_precondition(m)
entity m;
{
    transformer pre = transformer_identity();
    /* cons * effects_ = 
	load_statement_cumulated_effects(
           code_statement(value_code(entity_initial(m)))); */
    /* FI: to be recoded using a hash_map on one of value_mappings' hash
       tables or the argument field of m's intraprocedural transformer;
       statement s of module m could be passed as argument */
    list effects_ = load_module_intraprocedural_effects(m);
    /* the same entity might be entered twice, once as part of a read
       effect and a second time as part of a write effect; b is used
       to keep track of them */
    Pbase b = (Pbase) VECTEUR_NUL;

    /* assume m is a module and value mappings for m are available */

    /* FI: variables v initialized in a BLOCKDATA foo are not recognized 
     * because they are known as foo:v in the symbole table and not 
     * identified in the effects since a blockdata does not read or write any 
     * variable 
     */

    pips_debug(8, "begin for %s\n", module_local_name(m));

    /* look for entities with an integer initial value that are analyzed */
    MAPL(cef,
     {entity e = reference_variable(effect_reference(EFFECT(CAR(cef))));
      value val = entity_initial(e);
      if(value_constant_p(val) && constant_int_p(value_constant(val))) {
	  int int_val = constant_int(value_constant(val));
	  if(entity_has_values_p(e)
	     && !base_contains_variable_p(b, (Variable) e)) {
	      Pvecteur v = vect_new((Variable) e, VALUE_ONE);
	      vect_add_elem(&v, TCST, int_to_value(-int_val));
	      pre = transformer_equality_add(pre, v);
	      b = vect_add_variable(b, (Variable) e);
	  }}},
	 effects_);

    base_rm(b);
    pips_assert("data_to_precondition", pre != transformer_undefined);

    ifdebug(8) {
	dump_transformer(pre);
	debug(8, "data_to_precondition", "end for %s\n", module_local_name(m));
    }

    return pre;
}

/* a simpler remake of the previous one that works in all cases. FC.
 */
transformer 
all_data_to_precondition(entity m) 
{
    transformer pre = transformer_identity();
    Pbase b = (Pbase) VECTEUR_NUL;

    pips_debug(8, "begin for %s\n", module_local_name(m));

    /* look for entities with an integer initial value. */
    MAP(ENTITY, e,
    {
	value val = entity_initial(e);
	if(value_constant_p(val) && constant_int_p(value_constant(val))) 
	{
	    int int_val = constant_int(value_constant(val));
	    if(entity_has_values_p(e)
	       && !base_contains_variable_p(b, (Variable) e)) 
	    {
		Pvecteur v = vect_new((Variable) e, VALUE_ONE);
		vect_add_elem(&v, TCST, int_to_value(-int_val));
		pre = transformer_equality_add(pre, v);
		b = vect_add_variable(b, (Variable) e);
	    }
	}
    },
	code_declarations(entity_code(m)));

    base_rm(b);
    pips_assert("some transformer", pre != transformer_undefined);

    ifdebug(8) {
	dump_transformer(pre);
	pips_debug(8, "end for %s\n", module_local_name(m));
    }

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
    call c;

    debug(8,"instruction_to_postcondition","begin pre=%x tf=%x\n", pre, tf);

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
    debug(9,"instruction_to_postcondition","resultat post:\n");
    ifdebug(9) (void) print_transformer(post);
    debug(8,"instruction_to_postcondition","post=%x end\n", post);
    return post;
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

    pips_assert("statement_to_postcondition", pre != transformer_undefined);

    ifdebug(1) {
	int so = statement_ordering(s);
	(void) fprintf(stderr, "statement %03d (%d,%d), precondition %p:\n",
		       statement_number(s), ORDERING_NUMBER(so),
		       ORDERING_STATEMENT(so), pre);
	(void) print_transformer(pre) ;
    }

    pips_assert("statement_to_postcondition", tf != transformer_undefined);
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
	Psysteme s = 
		(Psysteme) predicate_system(transformer_relation(pre));

	free_predicate(transformer_relation(pre));
	gen_free_list(transformer_arguments(pre));
	transformer_arguments(pre) = NIL;
	transformer_relation(pre) = make_predicate(sc_empty(BASE_NULLE));
    }

    if (load_statement_precondition(s) == transformer_undefined) {
	/* keep only global initial scalar integer values;
	   else, you might end up giving the same xxx#old name to
	   two different local values */
	cons * non_initial_values =
	    arguments_difference(transformer_arguments(pre),
				 get_module_global_arguments());

	MAPL(cv,
	 {entity v = ENTITY(CAR(cv));
	  ENTITY(CAR(cv)) = entity_to_old_value(v);},
	     non_initial_values);

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
	    (void) fprintf(stderr, "statement %03d (%d,%d), precondition %p end:\n",
			   statement_number(s), ORDERING_NUMBER(so),
			   ORDERING_STATEMENT(so), pre);
	    (void) print_transformer(pre);
	    pips_error("statement_to_postcondition", "Non-consistent precondition after update\n");
	}

	/* store the precondition in the ri */
	store_statement_precondition(s,
				     transformer_filter(pre,
							non_initial_values));
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
