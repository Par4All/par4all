/* polynome_ri.c */

/* This file gathers some functions interfacing
 * polynomial library and the RI.
 *
 * The "Variable" type used by polynomials is
 * casted to "entity", the "Value" type, to int.
 *
 * char *variable_name(Variable var)
 *      return the complete name of the entity var
 *
 * char *variable_local_name(Variable var)
 *      return the abbreviated, local name of var
 *
 * bool is_inferior_var(Variable var1, var2)
 *      return TRUE if the complete name of var1
 *      is lexicographically before var2's one.
 *
 * Variable name_to_variable(char *name)
 *      inverse function of variable_name.
 *      name must be the complete name of the variable.
 */
/* Modif:
  -- entity_local_name is replaced by module_local_name. LZ 230993
*/

#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include <string.h>

#include "linear.h"

#include "genC.h"
#include "ri.h"
#include "complexity_ri.h"
#include "ri-util.h"
#include "misc.h"
#include "matrice.h"
#include "complexity.h"

char *variable_name(var)
Variable var;
{
    string s = (string) malloc(10);

    if (var == TCST) 
	s = strdup(TCST_NAME);
    /* FI: no longer useful */
    /*
    else if (var == UNKNOWN_VARIABLE) 
	s = strdup(UNKNOWN_VARIABLE_NAME);
    else if (var == UNKNOWN_RANGE) 
	s = strdup(UNKNOWN_RANGE_NAME);
	*/
    else if (var == (Variable) chunk_undefined) 
	pips_error("variable_name", "unexpected var == chunk_undefined.\n");
    else 
	s = strdup(entity_name((entity) var));

    return (s);
}

char *variable_local_name(var)
Variable var;
{
    string s = NULL;

    if (var == TCST) 
	s = strdup(TCST_NAME);
    /*
    else if (var == UNKNOWN_VARIABLE) 
	s = strdup(UNKNOWN_VARIABLE_NAME);
    else if (var == UNKNOWN_RANGE) 
	s = strdup(UNKNOWN_RANGE_NAME);
	*/
    else if (var == (Variable) chunk_undefined) 
	pips_error("variable_local_name",
		   "unexpected var == chunk_undefined.\n");
    else 
	s = strdup(entity_local_name((entity) var));

    return (s);
}

bool is_inferior_var(var1, var2)
Variable var1, var2;
{
    bool is_inferior = TRUE;
    
    if (var1 == TCST)
	is_inferior = TRUE;
    else if(var2 == TCST)
	is_inferior = FALSE;
    else
	is_inferior = (strcmp(variable_local_name(var1), 
			     variable_local_name(var2)) <= 0 );

    return is_inferior; 
}

bool is_inferior_varval(Pvecteur varval1, Pvecteur varval2)
{
    bool is_inferior = TRUE;
    
    if (term_cst(varval1))
	is_inferior = TRUE;
    else if(term_cst(varval2))
	is_inferior = FALSE;
    else
	is_inferior = (strcmp(variable_local_name(vecteur_var(varval1)), 
			      variable_local_name(vecteur_var(varval2)))
		       <= 0 );

    return is_inferior; 
}

bool is_inferior_pvarval(Pvecteur * pvarval1, Pvecteur * pvarval2)
{
    bool is_inferior = TRUE;
    
    if (term_cst(*pvarval1))
	is_inferior = FALSE;
    else if(term_cst(*pvarval2))
	is_inferior = TRUE;
    else
	is_inferior = (strcmp(variable_local_name(vecteur_var(*pvarval1)), 
			      variable_local_name(vecteur_var(*pvarval2)))
		       > 0 );

    return is_inferior; 
}

Variable name_to_variable(name)
char *name;
{
    if (strcmp(name, TCST_NAME) == 0) 
	return(TCST);
    /*
    else if (strcmp(name, UNKNOWN_VARIABLE_NAME) == 0) 
	return (UNKNOWN_VARIABLE);
    else if (strcmp(name, UNKNOWN_RANGE_NAME) == 0) 
	return (UNKNOWN_RANGE);
	*/
    else {
	entity e = gen_find_tabulated(name, entity_domain);
	if (e != entity_undefined) 
	    return ((Variable) e);
	else {
	    user_warning("name_to_variable",
			 "entity '%s' not found:return chunk_undefined\n", name);
	    return((Variable) chunk_undefined);
	}
    }
}

Variable local_name_to_variable(name)
char *name;
{
    string s = make_entity_fullname("TOP-LEVEL", name);

    if (strcmp(name, TCST_NAME)==0) 
	return(TCST);
    /*
    else if (strcmp(name, UNKNOWN_VARIABLE_NAME) == 0) 
	return (UNKNOWN_VARIABLE);
    else if (strcmp(name, UNKNOWN_RANGE_NAME) == 0) 
	return (UNKNOWN_RANGE);
	*/
    else {
	entity e = gen_find_tabulated(s, entity_domain);
	if (e != entity_undefined) 
	    return ((Variable) e);
	else {
	    user_warning("local_name_to_variable",
			 "entity '%s' not found:return chunk_undefined\n", name);
	    return((Variable) chunk_undefined);
	}
    }
}
