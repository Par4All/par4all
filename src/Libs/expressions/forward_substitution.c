/*
 * $Id$
 * 
 * Perform forward substitution when possible.
 *
 * The main purpose of such a transformation is to undo manual
 * optimizations targeted to some particular architecture...
 * 
 * What kind of interface should be available?
 * global? per loop? 
 *
 * basically there should be some common stuff with partial evaluation,
 * as suggested by CA, and I agree, but as I looked at the implementation
 * in the other file, I cannot really guess how to merge and parametrize
 * both stuff as one... FC. 
 *
 * This information is only implemented to forward substitution within 
 * a sequence. Things could be performed at the control graph level.
 *
 * An important issue is to only perform the substitution only if correct.
 *
 * $Log: forward_substitution.c,v $
 * Revision 1.4  1998/03/31 17:56:25  coelho
 * fixed a bug (free of maybe NULL value).
 *
 * Revision 1.3  1998/03/31 17:53:15  coelho
 * special case "x = a(i); a(j) = x+1;" is substituted before stopping.
 *
 * Revision 1.2  1998/03/31 17:17:45  coelho
 * cleaner + debug level.
 *
 * Revision 1.1  1998/03/31 16:08:25  coelho
 * Initial revision
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "genC.h"

#include "misc.h"
#include "ri.h"
#include "ri-util.h"
#include "resources.h"
#include "pipsdbm.h"
#include "properties.h"
#include "prettyprint.h"

#include "effects-generic.h"
#include "effects-simple.h"

/* structure to hold a substitution to be performed forward.
 */
typedef struct
{
    statement source; /* the statement where the definition was found. */
    entity var;       /* maybe could be a reference to allow arrays? */
    expression val;   /* the result of the substitution. */
}
    t_substitution, * p_substitution;

#define DEBUG_NAME "FORWARD_SUBSTITUTION_DEBUG_LEVEL"

/* returns whether there is other write proper effect than on var
 * or if some variable conflicting with var is read... (?) 
 * proper_effects must be available.
 */
static bool
functionnal_on(
    entity var,
    statement s)
{
    effects efs = load_proper_rw_effects(s);
    MAP(EFFECT, e, 
	if ((effect_write_p(e) && effect_variable(e)!=var) ||
	    (effect_read_p(e) && entity_conflict_p(effect_variable(e), var)))
	    return FALSE,
	effects_effects(efs));
    return TRUE;
}

/* Whether it is a candidate of a substitution operation. that is: 
 * (1) it is a call
 * (2) it is an assignment call
 * (3) the assigned variable is a scalar
 * (4) there are no side effects in the expression
 */
static p_substitution
substitution_candidate(statement s, bool only_scalar)
{
    list /* of expression */ args;
    call c;
    entity fun, var;
    syntax svar;
    instruction i = statement_instruction(s);
    p_substitution subs;

    if (!instruction_call_p(i)) return NULL; /* CALL */
    
    c = instruction_call(i);
    fun = call_function(c);

    if (!ENTITY_ASSIGN_P(fun)) return NULL; /* ASSIGN */

    args = call_arguments(c);
    pips_assert("2 args to =", gen_length(args)==2);
    svar = expression_syntax(EXPRESSION(CAR(args)));
    pips_assert("assign to a reference", syntax_reference_p(svar));
    var = reference_variable(syntax_reference(svar));

    if (only_scalar && !entity_scalar_p(var)) return NULL; /* SCALAR */

    if (!functionnal_on(var, s)) return NULL; /* NO SIDE EFFECTS */

    subs = (p_substitution) malloc(sizeof(t_substitution));

    subs->source = s;
    subs->var = var; /* ?? could be the expression for array propagation? */
    subs->val = EXPRESSION(CAR(CDR(args)));

    return subs;
}

/* x = a(i) ; a(j) = x;
 * we can substitute x but it cannot be continued.
 * just a hack for this very case at the time. 
 * maybe englobing parallel loops or dependence information could be use for 
 * a better decision? I'll think about it on request only.
 */
static bool
cool_enough_for_a_last_substitution(statement s)
{
    p_substitution x = substitution_candidate(s, FALSE);
    bool ok = (x!=NULL);
    if (x) free(x);
    return ok;
}

/* do perform the substution var -> val everywhere in s
 */
static entity the_variable;
static expression the_value;
static bool
expr_flt(expression e)
{
    syntax s = expression_syntax(e);
    reference r;
    if (!syntax_reference_p(s)) return TRUE;
    r = syntax_reference(s);
    if (reference_variable(r)==the_variable)
    {
	expression_syntax(e) = copy_syntax(expression_syntax(the_value));
	free_syntax(s);
	return FALSE;
    }
    return TRUE;
}
static void
perform_substitution(
    entity var, /* variable to be substituted */
    expression val, /* value of the substitution */
    statement s /* where to do this */)
{
    the_variable = var;
    the_value = val;
    gen_recurse(s, expression_domain, expr_flt, gen_null);
}

/* whether there are some conflicts between W cumulated in s2
 * and any proper of s1: this mean that some variable of s1 have
 * their value modified, hence the substitution cannot be carried on.
 * proper and cumulated effects must be available.
 */
static bool 
some_conflicts_between(statement s1, statement s2)
{
    effects efs1, efs2;
    efs1 = load_proper_rw_effects(s1);
    efs2 = load_cumulated_rw_effects(s2);

    MAP(EFFECT, e2,
    {
	if (effect_write_p(e2))
	{
	    entity v2 = effect_variable(e2);
	    MAP(EFFECT, e1,
		if (entity_conflict_p(effect_variable(e1), v2)) return TRUE,
		effects_effects(efs1));
	}
    },
	effects_effects(efs2));

    return FALSE;
}

/* top-down forward substitution of scalars in SEQUENCE only.
 */
static bool
seq_flt(sequence s)
{
    MAPL(ls, 
    {
	statement first = STATEMENT(CAR(ls));
	p_substitution subs = substitution_candidate(first, TRUE);
	if (subs)
	{
	    /* scan following statements and substitute while no conflicts.
	     */
	    MAP(STATEMENT, anext,
	    {
		if (some_conflicts_between(subs->source, anext))
		{
		    /* for some special case the substitution is performed.
		     */
		    if (cool_enough_for_a_last_substitution(anext))
			perform_substitution(subs->var, subs->val, anext);

		    /* now STOP propagation!
		     */
		    break; 
		}
		else
		    perform_substitution(subs->var, subs->val, anext);
	    },   
		CDR(ls));

	    free(subs);
	}
    },
        sequence_statements(s));

    return TRUE;
}

/* interface to pipsmake.
 * should have proper and cumulated effects...
 */
bool 
forward_substitute(string module_name)
{
    statement stat;

    debug_on(DEBUG_NAME);

    /* set require resources.
     */
    set_current_module_entity(local_name_to_top_level_entity(module_name));
    set_current_module_statement((statement)
        db_get_memory_resource(DBR_CODE, module_name, TRUE));
    set_proper_rw_effects((statement_effects)
	db_get_memory_resource(DBR_PROPER_EFFECTS, module_name, TRUE));
    set_cumulated_rw_effects((statement_effects)
	db_get_memory_resource(DBR_CUMULATED_EFFECTS, module_name, TRUE));

    stat = get_current_module_statement();

    /* do the job here:
     */
    gen_recurse(stat, sequence_domain, seq_flt, gen_null);

    /* return result and clean.
     */
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, stat);

    reset_cumulated_rw_effects();
    reset_proper_rw_effects();
    reset_current_module_entity();
    reset_current_module_statement();

    debug_off();

    return TRUE;
}
