/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

  This file is part of PIPS.

  PIPS is free software: you can redistribute it and/or modify it
  under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  any later version.

  PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.

  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with PIPS.  If not, see <http://www.gnu.org/licenses/>.

*/
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
/*
 * Partial Fortran statement functions support by cold expansion.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "parser_private.h"
#include "linear.h"
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

void parser_init_macros_support(void)
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

void parser_close_macros_support(void)
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

static macro_t * find_entity_macro(entity e)
{
    int i;
    for (i=0; i<current_macro_index; i++)
	if (same_entity_p(e, call_function(current_macros[i].lhs)))
	    return &current_macros[i];
    
    return NULL; /* not found */
}

bool parser_entity_macro_p(entity e)
{
    return find_entity_macro(e)==NULL;
}

void parser_add_a_macro(call c, expression e)
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
    
    if (find_entity_macro(macro) != NULL) {
      pips_user_warning("Macro \"%s\" is not yet defined.\n", 
			entity_name(macro));
      ParserError("parser_add_a_macro",
		  "It may be an undeclared array.\n");
    }

    /* expand macros in the macro! It is ok, because
     * referenced macros must appear in preceding lines (F77 15-5, line 3-5).
     */
    parser_substitute_all_macros_in_expression(e);

    /* store the result.
     */
    current_macros[current_macro_index].lhs = c;
    current_macros[current_macro_index].rhs = e;
    current_macro_index++;
}


/* is there a call to some untrusted function?
 */
static bool some_call;

static bool call_flt(call c)
{
    value v = entity_initial(call_function(c));
    if (value_intrinsic_p(v) || value_constant_p(v) || value_symbolic_p(v))
	return true;
    /* else untrusted!
     */
    some_call = true;
    gen_recurse_stop(NULL);
    return false;
}

static bool untrusted_call_p(expression e)
{
    some_call = false;
    gen_recurse(e, call_domain, call_flt, gen_null);
    return some_call;
}



/****************************************************** MACRO SUBSTITUTION */

/* must take care not to substitute in an inserted expression
 */
static expression s_init = expression_undefined, s_repl = expression_undefined;
static list /* of expression */ already_subs = NIL;

static bool expr_flt(expression e)
{
    return !gen_in_list_p(e, already_subs);
}

static void expr_rwt(expression e)
{
    if (expression_equal_p(s_init, e)) 
    {
	free_syntax(expression_syntax(e));
	expression_syntax(e) = copy_syntax(expression_syntax(s_repl));
	already_subs = CONS(EXPRESSION, e, already_subs);
    }
}

/* substitutes occurences of initial by replacement in tree
 */
static void substitute_expression_in_expression(
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

    gen_recurse(tree, expression_domain, expr_flt, expr_rwt);

    s_init = expression_undefined;
    s_repl = expression_undefined;
}

void reset_substitute_expression_in_expression(void)
{
    gen_free_list(already_subs); 
    already_subs = NIL;
}

void parser_macro_expansion(expression e)
{
    bool warned = false;
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

    pips_assert("same #args", gen_length(lactuals)==gen_length(lformals));

    reset_substitute_expression_in_expression();

    /* replace each formal by its actual.
     */
    for (; !ENDP(lactuals); POP(lactuals), POP(lformals))
    {
	expression actu, form;

	form = EXPRESSION(CAR(lformals)); /* MUST be a simple reference */
	pips_assert("dummy arg ok", 
		    expression_reference_p(form) &&
	  ENDP(reference_indices(syntax_reference(expression_syntax(form)))));

	/* if the replacement is a constant, or a reference without
	 * calls to external functions, it should be safe 
	 */
	actu = EXPRESSION(CAR(lactuals));

	if (!warned && untrusted_call_p(actu)) {
	    pips_user_warning("maybe non safe substitution of macro %s!\n",
			      module_local_name(macro));
	    warned = true;
	}

	substitute_expression_in_expression(rhs, form, actu);
    }

    reset_substitute_expression_in_expression();

    /* it is important to keep the same expression, for gen_recurse use.
     */
    free_syntax(expression_syntax(e));
    expression_syntax(e) = expression_syntax(rhs);
    expression_syntax(rhs) = syntax_undefined;
    free(rhs);
}


void parser_substitute_all_macros(statement s)
{
    if (current_macro_index>0 &&
        get_bool_property("PARSER_EXPAND_STATEMENT_FUNCTIONS"))
	gen_recurse(s, expression_domain, gen_true, parser_macro_expansion);
}


void parser_substitute_all_macros_in_expression(expression e)
{
  parser_substitute_all_macros((statement)e);
}
