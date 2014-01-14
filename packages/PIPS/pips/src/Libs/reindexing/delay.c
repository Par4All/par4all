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
/************************************************************************/
/* Name     : delay.c
 * Package  : reindexing
 * Author   : Alexis Platonoff
 * Date     : March 1995
 * Historic :
 *
 * Documents: SOON
 * Comments : This file contains the functions dealing with the dealy.
 */

/* Ansi includes 	*/
#include <stdio.h>

/* Newgen includes 	*/
#include "genC.h"

/* C3 includes 		*/
#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "ray_dte.h"
#include "sommet.h"
#include "sg.h"
#include "sc.h"
#include "polyedre.h"
#include "union.h"
#include "matrice.h"
#include "matrix.h"
#include "sparse_sc.h"

/* Pips includes 	*/
#include "boolean.h"
#include "ri.h"
#include "constants.h"
#include "ri-util.h"
#include "misc.h"
#include "complexity_ri.h"
#include "database.h"
#include "graph.h"
#include "dg.h"
#include "paf_ri.h"
#include "parser_private.h"
#include "property.h"
#include "reduction.h"
#include "text.h"
#include "text-util.h"
#include "tiling.h"
#include "text-util.h"
#include "pipsdbm.h"
#include "resources.h"
#include "static_controlize.h"
#include "paf-util.h"
#include "pip.h"
#include "array_dfg.h"
#include "prgm_mapping.h"
#include "conversion.h"
#include "scheduling.h"
#include "reindexing.h"

/* External variables */
extern hash_table delay_table;

/* Local defines */
typedef dfg_vertex_label vertex_label;
typedef dfg_arc_label arc_label;


/*=======================================================================*/
/* void rewrite_nothing(chunk *) {return;}: rewrite nothing,
 * incredible,no?
 * 
 * AC 94/07/25 
 * could use gen_null instead. FC.
 */

static void rewrite_nothing_call(call c) {return;}
static void rewrite_nothing_ref(reference r) {return;}


/*=======================================================================*/
/* bool reference_filter(r): filter on the reference r.
 * 
 * AC 94/07/28
 */

static bool reference_filter(r)
reference    r;
{
  entity       e = reference_variable(r);
  list         lexp = NIL, lexp2 = NIL;
  int          n, d;
  expression   exp, exp2;
  call         ca;

  /* first we get the number of the instruction */
  n = get_number_of_ins(e);
  
  /* get the delay in the hash table */
  d = (int)hash_get(delay_table, (char *)n);

  if (get_debug_level() > 6) {
    fprintf(stderr,"\nOld ref : ");
    print_reference(r);
    fprintf(stderr, "\n n = %d", n);
    fprintf(stderr, "\n d = %d", d);
  }   
    
  /* process the instruction only if the reference is an array. */
  if(reference_indices(r) != NIL) {
    if ((d > 0) && (d != INFINITY)) {
      exp = EXPRESSION(CAR(reference_indices(r)));
      lexp = CDR(reference_indices(r));
      /* build the modulo expression */
      lexp2 = CONS(EXPRESSION, int_to_expression(d+1), NIL);
      lexp2 = CONS(EXPRESSION, exp, lexp2);
      ca = make_call(entity_intrinsic(MODULO_OPERATOR_NAME), lexp2);
      exp2 = make_expression(make_syntax(is_syntax_call, ca),
			     normalized_undefined);
      reference_indices(r) = CONS(EXPRESSION, exp2, lexp);
    }
    else if (d == 0) {
      lexp = CDR(reference_indices(r));
      exp = int_to_expression(0);
      reference_indices(r) = CONS(EXPRESSION, exp, lexp);
    }
  }
  if (get_debug_level() > 6) {
    fprintf(stderr,"\nNew ref : ");
    print_reference(r);}

  return(false);
}

/*=======================================================================*/
/* bool assignation_filter(c): tests if the call is an assignation. This
 * is the filter of the function gen_recurse(). In case of a call we do not
 * want to go down so the bool is always set to false. We test too if
 * the assignation is an instruction and in that case, we treat each 
 * reference by calling the function gen_recurse again.
 * 
 * AC 94/07/25
 */

static bool assignation_filter(c)
call        c;
{
  list        lexp;
  expression  exp;

  if (ENTITY_ASSIGN_P(call_function(c)))
  {
    lexp = call_arguments(c);
    exp = EXPRESSION(CAR(lexp));
    
    /* first, test if the expression is an array that is an instruction */
    if (array_ref_exp_p(exp))
    { gen_recurse(c, reference_domain, reference_filter,
		  rewrite_nothing_ref); }
  }
  
  return(false);
}

/*=======================================================================*/
/* list add_delay_information(t, sl): go through the list of new statement 
 * and replace each first time dimension by its expression modulo the value 
 * of the delay.
 * 
 * AC 94/07/06
 */

list add_delay_information(t, sl)

 hash_table   t;
 list         sl;
{
 instruction  ins;

 ins = make_instruction_block(sl);

 gen_recurse(ins, call_domain, assignation_filter, rewrite_nothing_call);

 return(instruction_block(ins));
}


/*=======================================================================*/
/* void fprint_delay(fp, t): print the hash_table t
 *
 * AC 94/07/01 
 */

void fprint_delay(fp, g, t)
FILE        *fp;
graph       g;
hash_table  t;
{
  list        vl;

  for (vl = graph_vertices(g); !ENDP(vl); POP(vl)) {
    int         cn, del;
    vertex      cv;

    cv = VERTEX(CAR(vl));
    cn = dfg_vertex_label_statement(vertex_vertex_label(cv));
    del = (int)hash_get(delay_table, (char *)cn);
    fprintf(fp,"\nInstruction n. %d \t=> delai = %d", cn, del);
  }
}
