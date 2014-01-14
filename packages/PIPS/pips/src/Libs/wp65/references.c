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
 * Function of manipulation of reference lists
 * 
  * Corinne Ancourt
  */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "linear.h"

#include "genC.h"
#include "ri.h"
#include "effects.h"
#include "misc.h"
#include "ri-util.h"
#include "effects-util.h"
#include "constants.h"
#include "ri.h"
#include "effects.h"

#include "dg.h"
typedef dg_arc_label arc_label;
typedef dg_vertex_label vertex_label;
#include "graph.h"

#include "matrice.h"
#include "tiling.h"
#include "database.h"
#include "text.h"
#include "text-util.h"
#include "resources.h"

#include "wp65.h"

bool ref_in_list_p(list lt,reference r)
{
    return (false);
}

void update_map(statement_mapping m, statement st, reference r) 
{
    list lt;

    lt = (list) GET_STATEMENT_MAPPING(m,st);
    if (lt != (list) HASH_UNDEFINED_VALUE) { 
	if (!ref_in_list_p(lt,r)) {
	    lt =  gen_nconc(lt, CONS(REFERENCE,r, NIL));
	    ifdebug(9) 
		(void) fprintf(stderr,
			       "ajout de la ref: %s au statement %"PRIdPTR"\n",
			       entity_local_name(reference_variable(r)), 
			       statement_number(st));    
	}
    }
    else {
	lt= CONS(REFERENCE,r, NIL);
	ifdebug(9) 
	    (void) fprintf (stderr,
		"ajout de la ref: %s au statement %"PRIdPTR"\n",
		entity_local_name(reference_variable(r)), 
		statement_number(st));   
    }
    SET_STATEMENT_MAPPING(m,st,lt);
}

/* This function gives the list of operands belonging to Expression e
*/
list expression_to_operand_list(e, lr)
expression e;
list lr;
{
    syntax s = expression_syntax(e);
    switch(syntax_tag(s)) {
    case is_syntax_reference:
	lr = gen_nconc(lr, CONS(REFERENCE, syntax_reference(s), NIL));
		break;
    case is_syntax_range:
	lr = expression_to_operand_list(range_lower(syntax_range(s)), lr);
	lr = expression_to_operand_list(range_upper(syntax_range(s)), lr);
	lr = expression_to_operand_list(range_increment(syntax_range(s)),
					  lr);
	break;
    case is_syntax_call:
	MAPL(ce, {
	    expression e = EXPRESSION(CAR(ce));
	    lr = expression_to_operand_list(e, lr);
	    },
	     call_arguments(syntax_call(s)));
	break;
    default:
	(void) fprintf(stderr, 
		       "expression_to_operand_list - unexpected syntax\n");
    }

    return lr;
}

/* This function tests whether at least one  array indice of 
 * Reference r  belongs to  List lwr or not
 */

bool reference_in_list_p(reference r,list lwr)
{ 
    list lref2;
    bool result = false;
    for (lref2 = lwr; 
	 lref2 != NULL && !result; 
	 result = result || reference_equal_p(r,REFERENCE(CAR(lref2))),
	 lref2 = CDR(lref2)) ;
    return(result);
}
bool array_indice_in_list_p(reference r,list lwr)
{ 
 
    list lr = NIL;
    bool result = false;
    list lref1;
    MAPL(ce,{	 expression e = EXPRESSION(CAR(ce));
		 lr = expression_to_reference_list(e, lr);
	     },
	 reference_indices(r));
    for (lref1 = lr; 
	 lref1 != NIL && !result;
	 result = result || reference_in_list_p(REFERENCE(CAR(lref1)),
						lwr),
	 lref1 = CDR(lref1));
   
    return result;
}

/* This function add Reference r to List l, if r doesn't belong to l
*/
void reference_list_update(list *l, reference r)
{
    list lref1 =*l ;
    if (*l != NIL)  {
	for(; lref1!= NIL && !(reference_equal_p(REFERENCE(CAR(lref1)),r)) ; 
	    lref1 = CDR(lref1));
	if (lref1 == NIL) 
	    *l= gen_nconc(*l, CONS(REFERENCE,r,NIL));
    }
    else *l =  CONS(REFERENCE,r,NIL);
}

/* This function adds all the references of l2 to l1 if they don't appear in l1
*/
void reference_list_add(list *l1,list *l2)
{
    list lref2 =*l2 ;
    for(; lref2!= NIL; 
	reference_list_update(l1, REFERENCE(CAR(lref2))),
	lref2 = CDR(lref2));
}


/* This function prints the references belonging to l
*/ 
void reference_list_print(list l)
{
    list lref;
    for (lref = l; 
	 lref != NIL;
	 (void) fprintf(stderr, 
		       "%s,",
		       entity_local_name(reference_variable(REFERENCE(CAR(lref))))), 
	 lref = CDR(lref));
    (void) fprintf(stderr,"\n");
}


void reference_scalar_defined_p(reference r)
{
    assert(!reference_undefined_p(r) 
	   && r!=NULL && reference_variable(r)!=NULL 
	   && reference_indices(r) == NIL);
}
/* This function adds the reference r to List l, if the reference_variable(r) 
 * doesn't belong to l
*/
void variable_list_update(list *l, reference r)
{
    list lvar1 =*l ;
    if (*l != NIL)  {
	for(; 
	    lvar1!= NIL 
	    && strcmp(entity_local_name(reference_variable(REFERENCE(CAR(lvar1)))),
		      entity_local_name(reference_variable(r))); 
	    lvar1 = CDR(lvar1));
	if (lvar1 == NIL) 
	    *l= gen_nconc(*l, CONS(REFERENCE,r,NIL));
    }
    else *l =  CONS(REFERENCE,r,NIL);
}

/* This function adds all the references of l2 to l1 if they don't appear in l1
*/
void variable_list_add(list *l1,list *l2)
{
    list lvar2 =*l2 ;
    for(; lvar2!= NIL; 
	variable_list_update(l1,REFERENCE(CAR(lvar2))),
	lvar2 = CDR(lvar2));
}




void  concat_data_list(list * l,list * lr, statement st,statement_mapping map, bool perfect_nested_loop)
{
    instruction inst; 
    list  lt = (list) GET_STATEMENT_MAPPING(map,st);
 
    if (lt != (list) HASH_UNDEFINED_VALUE)  {
	variable_list_add(l, &lt);
	reference_list_add(lr,&lt);
    }
    inst = statement_instruction(st);
    switch(instruction_tag(inst)) {
    case is_instruction_block:{
	cons * b;
	b = instruction_block(inst);
	if (list_of_calls_p(b))
	    concat_data_list(l,lr,STATEMENT(CAR(b)),map,perfect_nested_loop);
	else
	    MAPL(st, { 
		concat_data_list(l,lr,STATEMENT(CAR(st)), map,
				 perfect_nested_loop);
		   } , b);
	break;
    }
   
    case is_instruction_loop: {
	concat_data_list(l,lr,loop_body(instruction_loop(inst)),map,
			 perfect_nested_loop);
	break;} 
    default:  return;
    }
}


