 /* semantic analysis: processing of loops
  *
  * $Id$
  *
  * $Log: loop.c,v $
  * Revision 1.1  2001/06/21 09:03:46  irigoin
  * Initial revision
  *
  *
  *
  */
#include <stdio.h>
#include <string.h>
/* #include <stdlib.h> */

#include "genC.h"
/* #include "database.h" */
#include "linear.h"
#include "ri.h"
/*
#include "text.h"
#include "text-util.h"
*/
#include "ri-util.h"
/* #include "constants.h" */
/* #include "control.h" */
#include "effects-generic.h"
#include "effects-simple.h"

#include "misc.h"

#include "properties.h"

#include "vecteur.h"
#include "contrainte.h"
/*
#include "ray_dte.h"
#include "sommet.h"
#include "sg.h"
#include "polyedre.h"
*/

#include "transformer.h"

#include "semantics.h"

transformer add_loop_skip_condition(transformer pre, loop l)
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

    debug(8,"add_loop_skip_condition","begin with pre\n");
    ifdebug(8) {
	(void) fprintf(stderr,"%s: %s\n","[add_loop_skip_condition]",
		       "input pre =");
	(void) print_transformer(pre);
    }

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
		vect_fprint(stderr, v, (char * (*)(Variable)) external_value_name);
	    }
	}
	else {
	    debug(8,"add_loop_skip_condition","Non-analyzed variable in loop bound(s)\n");
	}
    }
    else {
	debug(8,"add_loop_skip_condition","increment sign unknown or non-affine bound\n");
    }

    debug(8,"add_loop_skip_condition","end with pre\n");
    ifdebug(8) {
	(void) fprintf(stderr,"%s: %s\n","[add_loop_skip_condition]",
		       "new pre =");
	(void) print_transformer(pre);
    }

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
	else {
	    /* incr == 0 is used below as a give-up condition */
	    incr = 0;
	}

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


transformer add_loop_index_initialization(transformer pre, loop l)
{
    entity i = loop_index(l);
    expression init = range_lower(loop_range(l));
    transformer post = transformer_undefined;
    transformer t_init = transformer_undefined;
    list lef = expression_to_proper_effects(init);

    t_init = any_assign_to_transformer(i, init, lef, pre);
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
transformer
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
    /* Do not apply an extra iteration! */
    /* post = transformer_apply(t_body, post); */
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

transformer precondition_filter_old_values(transformer pre)
{
    Psysteme sc = predicate_system(transformer_relation(pre));
    Pbase b = BASE_UNDEFINED;

    for(b = sc_base(sc); !BASE_NULLE_P(b); b = vecteur_succ(b)) {
      entity old_v = (entity) vecteur_var(b);

      if(old_value_entity_p(old_v)) {
	pips_assert("No old values are left", FALSE);
      }
    }

    return pre;
}
