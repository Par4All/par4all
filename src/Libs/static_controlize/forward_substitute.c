/*

  $Id$

  Copyright 1989-2009 MINES ParisTech

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
/* Name     :   forward_substitute.c
 * Package  : 	static_controlize 
 * Author   :   Arnauld LESERVOT
 * Date     :	27 04 93
 * Modified :
 * Documents:	"Implementation du Data Flow Graph dans Pips"
 * Comments :	Some usefull functions to forward substitute an
 * 		expression in a statement scope. The core of this
 * 		file is forward_substitute_in_exp. 
 */

/* Ansi includes 	*/
#include <stdio.h>
#include <string.h>

/* Newgen includes	*/
#include "genC.h"

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "ray_dte.h"
#include "sommet.h"
#include "sg.h"
#include "sc.h"
#include "polyedre.h"
#include "matrix.h"

/* Pips includes	*/
#include "graph.h"
#include "database.h"
#include "ri.h"
#include "paf_ri.h"
#include "ri-util.h"
#include "constants.h"
#include "misc.h"
#include "control.h"
#include "text-util.h"
#include "static_controlize.h"
#include "pipsdbm.h"
#include "resources.h"
#include "paf-util.h"

/* Global variables:
 * replaced by a variable given as an argument "fst" : */
/* extern	hash_table Gforward_substitute_table ;*/


/*==================================================================*/
/* range forward_substitute_in_range((range*) pr, hash_table fst) AL 05/93
 * Forward-substitutes in a range all expressions in the global variable
 * Gforward_substitute_table.
 */
range	forward_substitute_in_range(pr, fst)
range* 	pr;
hash_table fst; /* forward substitute table */
{

	debug( 5, "forward_substitute_in_range", "begin\n");
	debug( 7, "forward_substitute_in_range", "forwarding in range_lower\n");
	forward_substitute_in_exp(&range_lower( *pr), fst); 
	debug( 7, "forward_substitute_in_range", "forwarding in range_upper\n");
	forward_substitute_in_exp(&range_upper( *pr), fst); 
	debug(7, "forward_substitute_in_range", 
			"forwarding in range_increment\n");
	forward_substitute_in_exp(&range_increment( *pr), fst);
	debug( 5, "forward_substitute_in_range", "end\n");
	return( *pr );
}

/*==================================================================*/
/* call forward_substitute_in_call((call*) pc, hash_table fs) AL 05/93
 * Forward-substitutes in a call all expressions in the global variable
 * Gforward_substitute_table.
 */
call	forward_substitute_in_call(pc, fst)
call	*pc;
hash_table fst; /* forward substitute table */
{
	list 	the_list;

	if hash_table_empty_p(fst) return( *pc );
	debug( 5, "forward_substitute_in_call", "call in : %s\n",
			entity_name( call_function( *pc )) );
	the_list = (list) call_arguments( *pc );
	if ( the_list != NIL ) forward_substitute_in_list( &the_list, fst );
	debug( 5, "forward_substitute_in_call", "end\n" );
	return( *pc );
}
	
/*==================================================================*/
/* loop forward_substitute_in_loop((loop*) pl , hash_table fs) AL 05/93
 * Forward-substitutes in a loop all expressions in the global variable
 * Gforward_substitute_table.
 */
loop	forward_substitute_in_loop(pl, fst)
loop 	*pl;
hash_table fst; /* forward substitute table */
{
	range	the_range;

	if hash_table_empty_p(fst) return( *pl );
	debug( 5, "forward_substitute_in_loop", "begin\n" );
	if ( hash_get( fst, (char*) loop_index(*pl))
		!= HASH_UNDEFINED_VALUE )
		pips_error( "forward_substitution_in_loop",
			"Redefinition of an index loop !\n" );
	the_range = loop_range( *pl );
	forward_substitute_in_range( &the_range, fst );
	debug( 5, "forward_substitute_in_loop", "end\n" );
	return( *pl );
}

/*==================================================================*/ 
/* list	forward_substitute_in_list((list*)  pl , hash_table fs)	AL 05/93
 * Forward-substitutes in a list all expressions in the 
 * global variable Gforward_substitute_table.
 */
list	forward_substitute_in_list(pl, fst)
list	*pl;
hash_table fst; /* forward substitute table */
{
	debug( 5, "forward_substitute_in_list", "begin\n" );
	for(; !ENDP( *pl ); POP( *pl ) ) {
		forward_substitute_in_exp(&(EXPRESSION(CAR(*pl))), fst);
	}
	debug( 5, "forward_substitute_in_list", "end\n" );
	return( *pl );
}


/*==================================================================*/
/* expression forward_substitute_in_exp((expression*) pexp , hash_table
 * fs) AL 05/93
 *
 * Forward-substitutes in an expression all expressions in
 * the global variable Gforward_substitute_table.
 */
expression forward_substitute_in_exp(pexp, fst)
expression *pexp;
hash_table fst; /* forward substitute table */
{
	expression	exp1;
	syntax		synt;
	reference	ref;
	list		indice;
	normalized	nor;

	if hash_table_empty_p(fst) return( *pexp );
	debug( 5, "forward_substitute_in_exp", 
		"  in exp : %s\n", ((*pexp == expression_undefined)?
		"expression undefined":words_to_string(words_expression( *pexp ))));

	if ( expression_normalized( *pexp ) != normalized_undefined )
					unnormalize_expression( *pexp );
	synt	= expression_syntax( *pexp );
	switch( syntax_tag( synt ) )
	{
	  case is_syntax_range	 :
	  {
	    forward_substitute_in_range(&syntax_range( synt ), fst);
	    break ;
	  }
	    case is_syntax_call	 :
	    {
	      forward_substitute_in_call(&syntax_call( synt ), fst);
	      break ;
	    }
	    case is_syntax_reference :
	    {
	      debug( 5, "forward_substitute_in_exp",
		    "forwarding in reference : begin \n");
	      ref = syntax_reference( synt );
	      indice  = reference_indices( ref );
	      if (forward_substitute_in_list(&indice, fst) !=
		  NIL) {
		debug( 5, "forward_substitute_in_reference",
		      "end\n");
		break;
	      }
	      exp1 = (expression) hash_get(fst,
					   (char*) reference_variable(ref));
	      if ( exp1 != expression_undefined )
		*pexp =  expression_dup( exp1 );
	      debug( 5, "forward_substitute_in_exp",
		    "forwarding in reference : end\n" );
	      break ;
	    }
	    default	: pips_error( "forward_substitute_in_exp",
				     "Bad expression tag" );
	  }
	unnormalize_expression( *pexp );
	nor = NORMALIZE_EXPRESSION( *pexp );
	if ( normalized_tag( nor ) == is_normalized_linear ) {
		expression exp2;
		exp2 = Pvecteur_to_expression( normalized_linear( nor ) );
		if (exp2 != expression_undefined) *pexp = exp2;
	}
	debug(5, "forward_substitute_in_exp", "  return exp : %s\n",
			words_to_string(words_expression( *pexp )) );
	return( *pexp );
}

/*==================================================================*/
