/*
 * $Id$
 *
 * $Log: macros.c,v $
 * Revision 1.1  1997/09/18 16:01:17  coelho
 * Initial revision
 *
 *
 * Partial Fortran statement functions support by cold expansion.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "parser_private.h"
#include "ri.h"
#include "ri-util.h"

#include "misc.h"
#include "properties.h"

#include "syntax.h"


/*********************************************************** MACRO HANDLING */

typedef struct {
    call lhs;
    expression rhs;
} macro_t;

static macro_t * current_macros;
static int current_macros_size = 0;
static int current_macro_index = 0; /* next available chunk */

void
parser_init_macros_support(void)
{
    pips_debug(5, "initializing macro-expansion support stuff\n");

    current_macro_index = 0; /* ??? memory leak... */
    
    if (current_macros_size==0)
    {
	current_macros_size = 10;
	current_macros = (macro_t*)
	    malloc(sizeof(macro_t)*current_macros_size);
	pips_assert("malloc ok", current_macros);
    }
}

void
parser_close_macros_support(void)
{
    pips_debug(5, "closing macro-expansion support stuff\n");

    for (current_macro_index--; current_macro_index>=0; current_macro_index--)
    {
	call c;
	entity macro;

	free_expression(current_macros[current_macro_index].rhs);
	c = current_macros[current_macro_index].lhs;

	macro = call_function(c);
	free_call(c);

	/* what about the entity? 
	 * It might exist such a real top-level entity...
	 * what if added as a callee...
	 * the entity should be destroyed...
	 * best would be to have it as a local entity,
	 * and have the calles and top-level updates delayed.
	 */
	remove_from_called_modules(macro);
    }
}

static macro_t *
find_entity_macro(entity e)
{
    int i;
    for (i=0; i<current_macro_index; i++)
	if (same_entity_p(e, call_function(current_macros[i].lhs)))
	    return &current_macros[i];
    
    return NULL; /* not found */
}

bool
parser_entity_macro_p(entity e)
{
    return find_entity_macro(e)==NULL;
}


void 
parser_add_a_macro(call c, expression e)
{
    entity macro = call_function(c);

    pips_debug(5, "adding macro %s\n", entity_name(macro));
    pips_assert("macros support initialized", current_macros_size>0);

    if (current_macro_index>=current_macros_size) /* resize! */
    {
	current_macros_size*=2;
	current_macros = (macro_t*) 
	    realloc(current_macros, sizeof(macro_t)*current_macros_size);
	pips_assert("realloc ok", current_macros);
    }
    
    pips_assert("macro not already defined", 
		find_entity_macro(macro) == NULL);

    current_macros[current_macro_index].lhs = c;
    current_macros[current_macro_index].rhs = e;
    current_macro_index++;
}


/****************************************************** MACRO SUBSTITUTION */

static expression s_init, s_repl;

static void
expr_rwt(expression e)
{
    if (expression_equal_p(s_init, e)) 
    {
	free_syntax(expression_syntax(e));
	expression_syntax(e) = copy_syntax(expression_syntax(s_repl));
    }
}

/* substitutes occurences of initial by replacement in tree
 */
static void
substitute_expression_in_expression(
    expression tree,
    expression initial,
    expression replacement)
{
    ifdebug(8) {
	pips_debug(8, "tree/initial/replacement\n");
	print_expression(tree);
	print_expression(initial);
	print_expression(replacement);
    }

    s_init = initial;
    s_repl = replacement;

    gen_recurse(tree, expression_domain, gen_true, expr_rwt);
}


void
parser_macro_expansion(expression e)
{
    macro_t * def;
    call c, lhs;
    entity macro;
    expression rhs;
    list /* of expression */ lactuals, lformals;

    if (!expression_call_p(e)) return;
    
    c = syntax_call(expression_syntax(e));
    macro = call_function(c);
    lactuals = call_arguments(c);

    /* get the macro definition. */
    def = find_entity_macro(macro);

    if (def==NULL) {
	pips_debug(5, "no macro definition for %s\n", entity_name(macro));
	return;
    }

    lhs = def->lhs;
    rhs = copy_expression(def->rhs); /* duplicated, for latter subs. */

    pips_assert("right macro function", macro == call_function(lhs));
    
    lformals = call_arguments(lhs);

    pips_assert("same # args", gen_length(lactuals)==gen_length(lformals));

    /* replace each formal by its actual.
     */
    for (; !ENDP(lactuals); POP(lactuals), POP(lformals))
	substitute_expression_in_expression
	    (rhs, EXPRESSION(CAR(lformals)), EXPRESSION(CAR(lactuals)));

    /* it is important to keep the same expression, for gen_recurse use.
     */
    free_syntax(expression_syntax(e));
    expression_syntax(e) = expression_syntax(rhs);
    expression_syntax(rhs) = syntax_undefined;
    free(rhs);
}


void 
parser_substitute_all_macros(statement s)
{
    if (get_bool_property("PARSER_EXPAND_STATEMENT_FUNCTIONS"))
	gen_recurse(s, expression_domain, gen_true, parser_macro_expansion);

    /* free all stuff? */
}
