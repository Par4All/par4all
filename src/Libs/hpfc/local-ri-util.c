/* $Id$
 * $Log: local-ri-util.c,v $
 * Revision 1.16  1997/10/30 17:09:53  coelho
 * nope.
 *
 * Revision 1.15  1997/03/20 10:20:58  coelho
 * RCS headers.
 *
 *
 * HPFC (c) Fabien Coelho May 1993
 */

#include "defines-local.h"

type type_variable_dup(t)
type t;
{
    if(type_variable_p(t))
    {
	variable v = type_variable(t);

	return MakeTypeVariable(variable_basic(v),
				ldimensions_dup(variable_dimensions(v)));
    }
    else
	return t; /* !!! means sharing */
}

/*  library functions...
 */
static string fortran_library[] =
{ 
  "TIME",
  (string) NULL
} ;

bool fortran_library_entity_p(e)
entity e;
{
    string *s, name=entity_local_name(e);

    if (!top_level_entity_p(e)) return(FALSE);
    for (s=fortran_library; *s!=(string) NULL; s++)
	if (same_string_p(*s, name)) return TRUE;

    return FALSE; /* else not found */
}

reference expression_to_reference(e)
expression e;
{
    syntax s = expression_syntax(e);
    message_assert("reference", syntax_reference_p(s));
    return syntax_reference(s);
}

/* just returns the entity of an expression... 
 */
entity expression_to_entity(e)
expression e;
{
    syntax s = expression_syntax(e);
    
    switch (syntax_tag(s))
    {
    case is_syntax_call:
	return call_function(syntax_call(s));
    case is_syntax_reference:
	return reference_variable(syntax_reference(s));
    case is_syntax_range:
    default: 
	pips_internal_error("unexpected syntax tag: %d\n", syntax_tag(s));
	return entity_undefined; 
    }
}

list expression_list_to_entity_list(l)
list /* of expressions */ l;
{
    list /* of entities */ n = NIL;
    MAP(EXPRESSION, e, n = CONS(ENTITY, expression_to_entity(e), n), l);
    return gen_nreverse(n);		 
}

list entity_list_to_expression_list(l)
list /* of entities */ l;
{
    list /* of expressions */ n = NIL;
    MAP(EXPRESSION, e, n = CONS(EXPRESSION, entity_to_expression(e), n), l);
    return gen_nreverse(n);
}

/* that is all
 */
