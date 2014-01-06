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
 *      return true if the complete name of var1
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
//#include <stdlib.h>
#include <string.h>

#include "linear.h"

#include "genC.h"
#include "ri.h"
#include "effects.h"
#include "complexity_ri.h"
#include "ri-util.h"
#include "effects-util.h"
#include "misc.h"
#include "matrice.h"
#include "preprocessor.h"
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
	pips_internal_error("unexpected var == chunk_undefined.");
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
	pips_internal_error("unexpected var == chunk_undefined.");
    else {
      // s = strdup(entity_local_name((entity) var));
	s = strdup(entity_minimal_name((entity) var));
    }

    return (s);
}

bool is_inferior_var(var1, var2)
Variable var1, var2;
{
    bool is_inferior = true;
    
    if (var1 == TCST)
	is_inferior = true;
    else if(var2 == TCST)
	is_inferior = false;
    else
	is_inferior = (strcmp(variable_local_name(var1), 
			     variable_local_name(var2)) <= 0 );

    return is_inferior; 
}

int is_inferior_varval(Pvecteur varval1, Pvecteur varval2)
{
  int is_inferior;

  if (term_cst(varval1))
    is_inferior = 1;
  else if(term_cst(varval2))
    is_inferior = -1;
  else
    is_inferior = - strcmp(variable_local_name(vecteur_var(varval1)),
			   variable_local_name(vecteur_var(varval2)));

  return is_inferior;
}

int is_inferior_pvarval(Pvecteur * pvarval1, Pvecteur * pvarval2)
{
  int is_inferior;

  if (term_cst(*pvarval1))
    is_inferior = -1;
  else if(term_cst(*pvarval2))
    is_inferior = 1;
  else {
    // FI: should be entity_user_name...
    is_inferior = strcmp(variable_local_name(vecteur_var(*pvarval1)),
			 variable_local_name(vecteur_var(*pvarval2)));
  }

  // FI: Make sure that you do not return 0 for two different entities
  if(is_inferior==0 && vecteur_var(*pvarval1)!=vecteur_var(*pvarval2)) {
    is_inferior = strcmp(entity_name((entity)vecteur_var(*pvarval1)),
			 entity_name((entity)vecteur_var(*pvarval2)));
  }

  return is_inferior;
}

Variable name_to_variable(char * name)
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
