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
 * Thus conversions are inserted and if none is available, the propagation
 * and substitution are not performed.
 *
 * $Log: forward_substitution.c,v $
 * Revision 1.8  1998/04/14 21:30:43  coelho
 * linear.h
 *
 * Revision 1.7  1998/04/01 13:22:02  coelho
 * initial comment updated after bug fix.
 *
 * Revision 1.6  1998/04/01 12:30:27  coelho
 * generic conversions are inserted if necessary, or the conversion is not
 * performed if no conversion function is available.
 *
 * Revision 1.5  1998/04/01 09:43:43  coelho
 * fixed a bug that was too optimistic when selectig cool statements...
 * w/w conflict have to be avoided to allow the transformation.
 *
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
#include "linear.h"
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

/* newgen-looking make/free
 */
static p_substitution 
make_substitution(statement source, entity var, expression val)
{
    p_substitution subs;
    basic bval = basic_of_expression(val), bvar = entity_basic(var);

    pips_assert("defined basics", 
		!(basic_undefined_p(bval) || basic_undefined_p(bvar)));

    subs = (p_substitution) malloc(sizeof(t_substitution));
    pips_assert("malloc ok", subs);
    subs->source = source;
    subs->var = var;
    if (basic_equal_p(bval, bvar))
	subs->val = copy_expression(val);
    else
    {
	entity conv = basic_to_generic_conversion(bvar);
	if (entity_undefined_p(conv))
	{
	    pips_user_warning("no conversion function...");
	    free(subs); 
	    return NULL;
	}
	subs->val = make_call_expression
	    (conv, CONS(EXPRESSION, copy_expression(val), NIL));
    }
    free_basic(bval);
    return subs;
}

static void 
free_substitution(p_substitution subs)
{
    if (subs) {
	free_expression(subs->val);
	free(subs);
    }
}

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

    if (!instruction_call_p(i)) return NULL; /* CALL */
    
    c = instruction_call(i);
    fun = call_function(c);

    if (!ENTITY_ASSIGN_P(fun)) return NULL; /* ASSIGN */

    ifdebug(7) {
	pips_debug(7, "considering assignment statement:\n");
	print_statement(s);
    }

    args = call_arguments(c);
    pips_assert("2 args to =", gen_length(args)==2);
    svar = expression_syntax(EXPRESSION(CAR(args)));
    pips_assert("assign to a reference", syntax_reference_p(svar));
    var = reference_variable(syntax_reference(svar));

    if (only_scalar && !entity_scalar_p(var)) return NULL; /* SCALAR */

    if (!functionnal_on(var, s)) return NULL; /* NO SIDE EFFECTS */

    return make_substitution(s, var, EXPRESSION(CAR(CDR(args))));
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
    free_substitution(x);
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
    p_substitution subs, /* substitution to perform */
    statement s /* where to do this */)
{
    ifdebug(8) {
	pips_debug(8, "\n");
	print_statement(subs->source);
	print_statement(s);
    }
    the_variable = subs->var;
    the_value = subs->val;
    /* maybe should take care of exit nodes? */
    gen_recurse(s, expression_domain, expr_flt, gen_null);
    ifdebug(8) {
	pips_debug(8, "result:\n");
	print_statement(s);
    }
}

/* whether there are some conflicts between W cumulated in s2
 * and any proper of s1: this mean that some variable of s1 have
 * their value modified, hence the substitution cannot be carried on.
 * if only_written in true, only W/W conflicts are reported.
 * proper and cumulated effects must be available.
 */
static bool 
some_conflicts_between(statement s1, statement s2, bool only_written)
{
    effects efs1, efs2;
    efs1 = load_proper_rw_effects(s1);
    efs2 = load_cumulated_rw_effects(s2);

    pips_debug(8, "looking for conflict %d/%d\n", 
	       statement_number(s1), statement_number(s2));

    MAP(EFFECT, e2,
    {
	if (effect_write_p(e2))
	{
	    entity v2 = effect_variable(e2);
	    pips_debug(9, "written variable %s\n", entity_name(v2));
	    MAP(EFFECT, e1,
		if (entity_conflict_p(effect_variable(e1), v2) &&
		    (!(only_written && effect_read_p(e1))))
		{
		    pips_debug(8, "conflict with %s\n",
			       entity_name(effect_variable(e1)));
		    return TRUE;
		},
		effects_effects(efs1));
	}
    },
	effects_effects(efs2));

    pips_debug(8, "no conflict\n");
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
		if (some_conflicts_between(subs->source, anext, FALSE))
		{
		    /* for some special case the substitution is performed.
		     */
		    if (cool_enough_for_a_last_substitution(anext) &&
			!some_conflicts_between(subs->source, anext, TRUE))
			perform_substitution(subs, anext);

		    /* now STOP propagation!
		     */
		    break; 
		}
		else
		    perform_substitution(subs, anext);
	    },   
		CDR(ls));

	    free_substitution(subs);
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
