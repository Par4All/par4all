/* $RCSfile: utils.c,v $ (version $Revision$)
 * $Date: 1996/06/15 18:30:12 $, 
 *
 * utilities for reductions.
 *
 * FC, June 1996.
 */

#include "local-header.h"
#include "semantics.h"

/******************************************************************** MISC */
/* ??? some arrangements with words_range to print a star in this case:-)
 */
static syntax make_star_syntax()
{
    return make_syntax(is_syntax_range, 
		       make_range(expression_undefined, 
				  expression_undefined, 
				  expression_undefined));
}

/***************************************************** REFERENCED ENTITIES */
/* returns the list of referenced variables
 */
static list /* of entity */ referenced;
static void ref_rwt(reference r)
{ referenced = gen_once(reference_variable(r), referenced);}

static list /* of entity */
referenced_variables(reference r)
{
    referenced = NIL;
    gen_recurse(r, reference_domain, gen_true, ref_rwt);
    return referenced;
}

/************************************** REMOVE A VARIABLE FROM A REDUCTION */
/* must be able to remove a modified variable from a reduction:
 * . A(I) / I -> A(*)
 * . A(B(C(I))) / C -> A(B(*))
 * . A(I+J) / I -> A(*)
 * . A(B(I)+C(I)) / I -> A(*) or A(B(*)+C(*)) ? former looks better
 */

/* the variable that must be removed is stored here
 */
static entity variable_to_remove;
/* list of expressions to be deleted (at rwt)
 */
static list /* of expression */ dead_expressions;
/* the first expression encountered which is a function call, so as 
 * to avoid "*+J" results
 */
static expression first_encountered_call;
/* stack of reference expressions (if there is no call)
 */
DEFINE_LOCAL_STACK(ref_exprs, expression)

static bool expr_flt(expression e)
{ 
    if (expression_reference_p(e)) /* REFERENCE */
	ref_exprs_push(e);
    else if (!first_encountered_call && expression_call_p(e)) /* CALL */
	first_encountered_call = e;
    return TRUE;
}

static void expr_rwt(expression e)
{
    if (expression_reference_p(e)) 
	ref_exprs_pop();
    else if (first_encountered_call==e)
	first_encountered_call = NULL;

    if (gen_in_list_p(e, dead_expressions))
    {
	free_syntax(expression_syntax(e));
	expression_syntax(e) = make_star_syntax();
    }
}

static bool ref_flt(reference r)
{
    if (reference_variable(r)==variable_to_remove)
    {
	dead_expressions = 
	    gen_once(first_encountered_call? first_encountered_call:
		     ref_exprs_head(), dead_expressions);
	return FALSE;
    }
    return TRUE;
}

void 
remove_variable_from_reduction(
    reduction red,
    entity var)
{
    variable_to_remove = var;
    dead_expressions = NIL;
    first_encountered_call = NULL;
    make_ref_exprs_stack();

    gen_multi_recurse(reduction_reference(red), 
		      expression_domain, expr_flt, expr_rwt,
		      reference_domain, ref_flt, gen_null,
		      NULL);

    gen_free_list(dead_expressions);
    free_ref_exprs_stack();
}

bool
update_reduction_under_effect(
    reduction red,
    effect eff)
{
    entity var = effect_variable(eff);
    bool updated = FALSE;
    
    /* REDUCTION is dead if the reduction variable is affected
     */
    if (entity_conflict_p(reduction_variable(red),var))
    {
	reduction_operator_tag(reduction_op(red)) = 
	    is_reduction_operator_none;
	return FALSE;
    }
    /* else */

    if (effect_read_p(eff)) return TRUE;

    /* now var is written */
    MAP(ENTITY, e, 
    {
	if (entity_conflict_p(var, e)) 
	{ 
	    updated = TRUE; 
	    remove_variable_from_reduction(red, e);
	}
    },
	reduction_dependences(red));

    if (updated)
    {
	gen_free_list(reduction_dependences(red));
	reduction_dependences(red) =
	     referenced_variables(reduction_reference(red));
    }

    return TRUE;
}

/* looks for a reduction about var in reds, and returns it.
 * tells whether it worths keeping on. It does not if there may be some 
 * conflicts with other reduced variables...
 */
static bool 
find_reduction_of_var(entity var, reductions reds, reduction *pr)
{
    MAP(REDUCTION, r,
    {
	entity red_var = reduction_variable(r);
	if (red_var==var)
	{
	    *pr = copy_reduction(r);
	    return TRUE;
	}
	else if (entity_conflict_p(red_var, var))
	    return FALSE; /* I will not combine them... */
    },
	reductions_list(reds));
    return TRUE;
}

/* merge two reductions into first so as to be compatible with both.
 * deletes the second. tells whether they where compatibles
 * quite basic at the time
 */
static bool 
merge_two_reductions(reduction first, reduction second)
{
    pips_assert("same variable", 
		reduction_variable(first)==reduction_variable(second));

    if (reduction_operator_tag(reduction_op(first))!=
	reduction_operator_tag(reduction_op(second)))
    {
	free_reduction(second);
	return FALSE;
    }

    if (!reference_equal_p(reduction_reference(first), 
			   reduction_reference(second)))
    {
	/* actually merges, very simple at the time 
	 */
	free_reference(reduction_reference(first));
	reduction_reference(first) = 
	    make_reference(reduction_variable(second), NIL);
    }

    free_reduction(second);
    return TRUE;
}

/* what to do with reduction *pr for variable var 
 * under effects le and reductions reds.
 * - *pr==NULL => no candidates found yet.
 * returns whether woth to go on.
 * conditions:
 */   
bool 
update_compatible_reduction(
    reduction *pr,
    entity var,
    list /* of effect */ le,
    reductions reds)
{
    reduction found = NULL;

    if (!find_reduction_of_var(var, reds, &found)) 
	return FALSE;

    if (found)
    {
	if (*pr) /* some reduction already available */
	    return merge_two_reductions(*pr, found);
	else 
	{
	    *pr = found;
	    return TRUE;
	}
    }
    /* else 
     * now no new reduction waas found, must check *pr against effects 
     */
    if (*pr) /* some reduction */
    {
	MAP(EFFECT, e,
	    if (!update_reduction_under_effect(*pr, e))
	    {
		free_reduction(*pr), *pr=NULL;
	        return FALSE,
	    }
	    le);
    }
    else
    {
	MAP(EFFECT, e,
	    if (entity_conflict_p(effect_variable(e), var))
	        return FALSE,
	    le);
    }
    
    return TRUE;
}

/*************************************************** CALL PROPER REDUCTION */
/* extract the proper reduction of a call (instruction) if any.
 */

/* I trust intrinsics (operations?) and summary effects...
 */
bool pure_function_p(entity f)
{
    value v = entity_initial(f);

    if (value_symbolic_p(v) || value_constant_p(v) || value_intrinsic_p(v))
	return TRUE;
    /* else */

    MAP(EFFECT, e,
    {
	if (effect_write_p(e)) return FALSE; /* a side effect!? */
	if (io_effect_entity_p(effect_variable(e))) return FALSE; /* LUNS */
    },
	load_summary_effects(f));

    return TRUE;
}

/* tells whether r is a functional reference...
 * actually I would need to recompute somehow the proper effects of obj?
 */
static bool is_functional;
static bool call_flt(call c)
{
    if (pure_function_p(call_function(c))) 
	return TRUE;
    /* else */
    is_functional = FALSE;
    gen_recurse_stop(NULL);
    return FALSE;
}
static bool
functional_object_p(gen_chunk* obj)
{
    is_functional = TRUE;
    gen_recurse(obj, call_domain, call_flt, gen_null);
    return is_functional;
}

/* returns the possible operator if it is a reduction, 
 * and the operation commutative. (- and / are not)
 */
#define OKAY(op,com) { *op_tag=op; *commutative=com; return TRUE;}

static bool 
extract_reduction_operator(
    expression rhs,
    tag *op_tag,
    bool *commutative)
{
    syntax s = expression_syntax(rhs);
    call c;
    entity f;

    if (!syntax_call_p(s)) return FALSE;
    c = syntax_call(s);
    f = call_function(c);
    
    if (ENTITY_PLUS_P(f)) 
	OKAY(is_reduction_operator_sum, TRUE)
    else if (ENTITY_MINUS_P(f))
	OKAY(is_reduction_operator_sum, FALSE)
    else if (ENTITY_MULTIPLY_P(f))
	OKAY(is_reduction_operator_prod, TRUE)
    else if (ENTITY_DIVIDE_P(f))
	OKAY(is_reduction_operator_prod, FALSE)
    else if (ENTITY_MIN_P(f) || ENTITY_MIN0_P(f))
	OKAY(is_reduction_operator_min, TRUE)
    else if (ENTITY_MAX_P(f) || ENTITY_MAX0_P(f))
	OKAY(is_reduction_operator_max, TRUE)
    else if (ENTITY_AND_P(f))
	OKAY(is_reduction_operator_and, TRUE)
    else if (ENTITY_OR_P(f))
	OKAY(is_reduction_operator_or, TRUE)
    else
	return FALSE;
}

/* looks for an equal reference in the list.
 * may be stopped at the first element if first_only.
 * the reference found is also returned.
 */
static bool 
equal_reference_in_list_p(
    reference r,
    list /* of expression */ le,
    bool first_only,
    reference *found)
{
    MAP(EXPRESSION, e,
    {
	syntax s = expression_syntax(e);
	if (syntax_reference_p(s))
	{
	    *found = syntax_reference(s);
	    if (reference_equal_p(r, *found)) return TRUE;
	}

	if (first_only) return FALSE;
    },
	le);

    return FALSE;
}

/* checks that the references are the only touched within this statement.
 * I trust the proper effects to store all references...
 */
static bool
no_other_effects_on_references(
    statement s,
    reference r1,
    reference r2)
{
    list /* of effect */ le = load_statement_proper_effects(s);
    entity var = reference_variable(r1);

    pips_assert("same variable", var==reference_variable(r2));
    
    MAP(EFFECT, e,
    {
	reference r = effect_reference(e);
	if (r!=r1 && r!=r2 && entity_conflict_p(reference_variable(r), var))
	    return FALSE;
    },
	le);

    return TRUE;
}

/* is the call in s a reduction? returns the reduction if so.
 * mallocs are avoided if nothing is found...
 * looks for v = v OP y, where y is independent of v.
 */
bool 
call_proper_reduction_p(
    statement s,    /* needed to query about proper effects */
    call c,         /* the call of interest */
    reduction *red) /* the returned reduction (if any) */
{
    list /* of expression */ le;
    expression elhs, erhs;
    reference lhs, other;
    tag op;
    bool comm;

    pips_debug(7, "call to %s (0x%x)\n", 
	       entity_name(call_function(c)), (unsigned int) s);

    if (!ENTITY_ASSIGN_P(call_function(c))) 
	return FALSE;

    le = call_arguments(c);
    pips_assert("two arguments", gen_length(le)==2);
    elhs = EXPRESSION(CAR(le));
    erhs = EXPRESSION(CAR(CDR(le)));

    pips_assert("lhs reference", syntax_reference_p(expression_syntax(elhs)));
    lhs = syntax_reference(expression_syntax(elhs));

    /* the lhs and rhs must be functionnal 
     * (same location on different evaluations)
     */
    if (!functional_object_p(lhs) || !functional_object_p(erhs))
	return FALSE;
    
    /* the operation performed must be valid for a reduction
     */
    if (!extract_reduction_operator(erhs, &op, &comm))
	return FALSE;

    /* there should be another direct reference as lhs
     * !!! syntax is a call if extract_reduction_operator returned TRUE
     */
    if (!equal_reference_in_list_p(lhs, 
	 call_arguments(syntax_call(expression_syntax(erhs))), !comm, &other))
	return FALSE;

    /* there should be no extract effects on the reduced variable
     */
    if (!no_other_effects_on_references(s, lhs, other))
	return FALSE;

    pips_debug(7, "returning a %s reduction on %s\n",
	       reduction_operator_tag_name(op), 
	       entity_name(reference_variable(lhs)));

    /* well, it is ok for a reduction now! 
     */
    *red = make_reduction(copy_reference(lhs),
			  make_reduction_operator(op, UU), 
			  referenced_variables(lhs));
    return TRUE;
}

/* end of $RCSfile: utils.c,v $
 */
