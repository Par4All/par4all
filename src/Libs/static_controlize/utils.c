/*

  $Id$

  Copyright 1989-2010 MINES ParisTech

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
/* Name     :	utils.c
 * Package  :	static_controlize.c
 * Author   :	Arnauld LESERVOT
 * Date     :	27/04/93
 * Modified :	
 * Documents:	"Implementation du Data Flow Graph dans Pips"
 * Comments :	
 */

/* Ansi includes 	*/
#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Newgen includes	*/
#include "genC.h"
#include "boolean.h"

/* C3 includes		*/
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
#include "ri.h"
/* Types arc_label and vertex_label must be defined although they are
   not used */
typedef void * arc_label;
typedef void * vertex_label;
#include "graph.h"
#include "paf_ri.h"
#include "database.h"
#include "parser_private.h"
#include "ri-util.h"
#include "constants.h"
#include "misc.h"
#include "text.h"
#include "text-util.h"
#include "paf-util.h"
#include "effects-generic.h"
#include "alias-classes.h"
#include "static_controlize.h"

/* Global variables 	*/
/* extern	list		Gscalar_written_forward; */
extern 	list		Gstructure_parameters;


extern list assigned_var;
#define entity_assigned_by_array_p(ent) (gen_find_eq(ent, assigned_var) != chunk_undefined)
/*
extern  list		Genclosing_loops;
extern  list		Genclosing_tests;
extern  hash_table	Gforward_substitute_table; */


/*=======================================================================*/
/* list stco_same_loops( in_map, in_s, in_s2 )			AL 25/10/93
 * Input  : A static_control mapping on statements and 2 statements.
 * Output : A list of loops common to in_s  and in_s2.
 */
list stco_same_loops( in_map, in_s, in_s2 )
statement_mapping	in_map;
statement 		in_s, in_s2;
{
        list            l, l2, ret_l = NIL;
        static_control  sc, sc2;

        pips_debug(7,"begin\n");
        sc = (static_control) GET_STATEMENT_MAPPING( in_map, in_s );
        sc2= (static_control) GET_STATEMENT_MAPPING( in_map, in_s2 );
        l = static_control_loops( sc );
        l2 = static_control_loops( sc2 );
        for(; !ENDP(l); POP(l)) {
		list l3 = l2;
                loop lo = LOOP(CAR( l ));
                int in = statement_ordering(loop_body( lo ));
                for(; !ENDP(l3); POP(l3) ) {
                        loop ll = LOOP(CAR( l3 ));
                        int in2 = statement_ordering(loop_body( ll ));
                        if( in == in2 ) ADD_ELEMENT_TO_LIST(ret_l, LOOP, ll);
                }
        }
        pips_debug(7, "end \n");

        return( ret_l );
}

/*=================================================================*/
/* int stco_renumber_code( (statement) in_st, (int) in_ct )	AL 25/10/93
 * Input  : A statement in_st and a begin count number in_ct.
 * Output : Renumber statement_number in an textual order and return
 *		the last number attributed.

 * I've removed many renumbering since it is done by a deeper call to
 * stco_renumber_code() and renumbering sequences kill an assert in
 * the prettyprinter...

 */
int
stco_renumber_code(statement in_st,
		   int in_ct)
{
  int		count;
  instruction	inst;
  
  pips_debug(7, "begin\n");
  count = in_ct;
  inst = statement_instruction( in_st );
  
  /* Renumber all the statement but the sequence: */
  if (instruction_tag(inst) != is_instruction_block)
    statement_number(in_st) = count++;
  
  switch(instruction_tag(inst)) {
  case is_instruction_block : {
    MAPL( stmt_ptr, {
	statement st = STATEMENT(CAR( stmt_ptr ));
	/*
	  statement_number( st ) = count++;
	*/
	count = stco_renumber_code( st, count );
      }, instruction_block( inst ) );
    break;
  }
  case is_instruction_test : {
    test t = instruction_test(inst);
    statement tt, tf;
    
    tt = test_true( t );
    tf = test_false( t );
    /*
      statement_number( tt ) = count++;
      statement_number( tf ) = count++;
    */
    count = stco_renumber_code( tt, count );
    count = stco_renumber_code( tf, count );
    break;
  }
  case is_instruction_loop : {
    statement lb = loop_body( instruction_loop( inst ) );
    /*
      statement_number( lb ) = count++;
    */
    count = stco_renumber_code( lb, count );
    break;
  }
  case is_instruction_forloop :
    {
      break;
    }
  case  is_instruction_whileloop :
    {
      break;
    }
  case is_instruction_call : {
    break;
  }
  case is_instruction_goto : {
    statement gs = instruction_goto( inst );
    /*
      statement_number( gs ) = count++;
    */
    count = stco_renumber_code( gs, count );
    break;
  }
  case is_instruction_expression :
    {
      break;
    }
  case is_instruction_unstructured : {
    list blocs = NIL;
    unstructured u = instruction_unstructured( inst );
    
    control_map_get_blocs(unstructured_control(u), &blocs ) ;
    blocs = gen_nreverse( blocs ) ;
    MAPL( ctl_ptr,  {
	statement stmt = control_statement(
					   CONTROL(CAR( ctl_ptr )));
	/*
	  statement_number( stmt ) = count++;
	*/
	count = stco_renumber_code( stmt, count );
      }, blocs);
    gen_free_list(blocs);
    break;
  }
  default : pips_error("stco_renumber_code",
		       "Bad instruction tag");
  }
  
  pips_debug(7, "return count : %d\n", count);
  return( count );
}

/*=================================================================*/
/* expression sc_opposite_exp_of_conjunction(expression exp, (list) *ell)
 * AL 08/19/93 Input : exp a conjunction of linear expressions Output :
 * simplified value of NOT( exp )
 */
expression sc_opposite_exp_of_conjunction(exp, ell)
expression exp;
list *ell;
{
	expression ret_exp;

        pips_debug(9, "begin\n");
        if ( (exp == expression_undefined) ||
                (syntax_tag(expression_syntax( exp )) != is_syntax_call) ) {
		ret_exp = expression_undefined;
        }
	else if (splc_positive_relation_p(exp, ell)) {
		expression 	exp1, exp2;
		exp1 = EXPRESSION(CAR( call_arguments(
				syntax_call(expression_syntax( exp )))));
		exp2 = MakeBinaryCall( ENTITY_GE,
			   make_op_exp( MINUS_OPERATOR_NAME,
				int_to_expression(-1), exp1 ),
			   int_to_expression(0) );
		ret_exp = exp2;
	}
	else if(ENTITY_AND_P(call_function(
			syntax_call(expression_syntax( exp ))))) {
		list		args = NIL;
		expression 	r_exp, l_exp;

		args = call_arguments(syntax_call(expression_syntax( exp )));
		l_exp = EXPRESSION(CAR( args ));
		r_exp = EXPRESSION(CAR(CDR( args )));
		ret_exp = MakeBinaryCall( ENTITY_OR,
				sc_opposite_exp_of_conjunction(l_exp, ell),
				sc_opposite_exp_of_conjunction(r_exp, ell));
			
	}
	else ret_exp = expression_undefined;

	pips_debug(9, "end\n");
	return( ret_exp );
}


/*=================================================================*/
/* bool splc_positive_relation_p((expression) exp, list *ell) AL 04/93
 * Returns true if exp is an affine form of structural parameters and of
 * loop-counters.
 */
bool splc_positive_relation_p(exp, ell)
expression exp;
list *ell;
{
	syntax s = expression_syntax( exp );
	call c;
	list args;

	pips_debug(7, "exp : %s\n",
		   words_to_string( words_expression( exp, NIL ) ));
	if (syntax_tag(s) != is_syntax_call) return( false );
	c = syntax_call( s );
	if (!ENTITY_GREATER_OR_EQUAL_P(call_function( c ))) return( false );
	args = call_arguments( c );
	return(splc_linear_expression_p(EXPRESSION(CAR(args)), ell) &&
	       expression_equal_integer_p( EXPRESSION(CAR(CDR(args))), 0 ) );
}

/*=================================================================*/
/* list ndf_normalized_test( (expression) exp, (list) *ell) AL 04/93
 * Returns a list of positive linear forms from an input expression which
 * is a logical combinaison of affine forms of structural parameters and
 * of loop counters.
 */
list ndf_normalized_test(expression exp, list *ell)
{
	list 		args, ret_list = NIL;
	entity		fun;
	expression 	arg1, arg2, exp2, exp3;

	pips_debug(7, "doing\n");
	if ( (exp == expression_undefined) ||
		(syntax_tag(expression_syntax( exp )) != is_syntax_call) ) {
		return  NIL;
	}
	pips_debug(7, "input exp : %s\n",
		   words_to_string(words_expression( exp, NIL )) );

	fun = call_function(syntax_call(expression_syntax( exp )));
	args = call_arguments(syntax_call(expression_syntax( exp )));
	if(ENDP(args)) return NIL;

	if (splc_positive_relation_p(exp, ell))  {
		ADD_ELEMENT_TO_LIST( ret_list, EXPRESSION, exp );
		return( ret_list );
	}
	if (ENTITY_NOT_P( fun )) {
		arg1 = EXPRESSION(CAR(args));
		if (splc_positive_relation_p(arg1, ell)) {
			exp3 = EXPRESSION(CAR( call_arguments(
				 syntax_call(expression_syntax( arg1 )))));
			exp2 = MakeBinaryCall( ENTITY_GE,
				 make_op_exp( MINUS_OPERATOR_NAME,
				   int_to_expression(-1),
				   exp3 ),
				 int_to_expression(0) );
			ADD_ELEMENT_TO_LIST( ret_list, EXPRESSION, exp2 );
			return( ret_list );
		}
		else {
			expression exp2 = expression_undefined;

			MAPL( exp_ptr, {
			   exp3 = EXPRESSION(CAR( exp_ptr ));
			   if (exp2 == expression_undefined)
			     exp2 = sc_opposite_exp_of_conjunction(exp3, ell);
			   else
			     exp2 = MakeBinaryCall(ENTITY_AND,
				     sc_opposite_exp_of_conjunction(exp3, ell),
						   copy_expression( exp2 ) );
			}, ndf_normalized_test(arg1, ell));
			ret_list = ndf_normalized_test(exp2, ell);
			return( ret_list );
		}
	}

	arg1 = EXPRESSION(CAR(args));
	arg2 = EXPRESSION(CAR(CDR(args)));

	/*  Redon FND propagation : see document. */
	if (ENTITY_OR_P( fun )) {
	   expression exp3;
	
	   MAPL( exp_ptr, {
	      exp3 = EXPRESSION(CAR( exp_ptr ));
	      ADD_ELEMENT_TO_LIST( ret_list, EXPRESSION, exp3 );
	      }, ndf_normalized_test(arg1, ell));
	   MAPL( exp_ptr, {
	      exp3 = EXPRESSION(CAR( exp_ptr ));
	      ADD_ELEMENT_TO_LIST( ret_list, EXPRESSION, exp3 );
	      }, ndf_normalized_test(arg2, ell));
	   return( ret_list );
	}
	
	/*
	%%%%%%%%%%%%%
	*/

	/* We propagate an Exclusive Normal Disjunctive Form */
	/* if (ENTITY_OR_P( fun )) {
	 * list		not_l1, not_l2, l1, l2;
	 *
	 *l1 = ndf_normalized_test(arg1, ell);
	 *l2 = ndf_normalized_test( arg2, ell);
	 *not_l1 = ndf_normalized_test( MakeUnaryCall(ENTITY_NOT, arg1), ell );
	 *not_l2 = ndf_normalized_test( MakeUnaryCall(ENTITY_NOT, arg2), ell );
	 *
	 * MAPL( exp_ptr, {
	 * expression S1 = EXPRESSION(CAR( exp_ptr ));
	 *MAPL( exp_ptr2, {
	 *expression T1 = EXPRESSION(CAR( exp_ptr ));
	 *ADD_ELEMENT_TO_LIST( ret_list,  EXPRESSION,
	 *MakeBinaryCall( ENTITY_AND, S1, T1 ) );
	 *}, l2);
	 *}, not_l1 );
	 *
	 *MAPL( exp_ptr, {
	 *expression S1 = EXPRESSION(CAR( exp_ptr ));
	 *		MAPL( exp_ptr2, {
	 *			expression T1 = EXPRESSION(CAR( exp_ptr ));
	 *			ADD_ELEMENT_TO_LIST( ret_list,  EXPRESSION,
	 *				MakeBinaryCall( ENTITY_AND, S1, T1 ) );
	 *		}, l2);
	 *	}, l1 );
 	 *
	 *	MAPL( exp_ptr, {
	 *		expression S1 = EXPRESSION(CAR( exp_ptr ));
	 *		MAPL( exp_ptr2, {
	 *			expression T1 = EXPRESSION(CAR( exp_ptr ));
	 *			ADD_ELEMENT_TO_LIST( ret_list,  EXPRESSION,
	 *				MakeBinaryCall( ENTITY_AND, S1, T1 ) );
	 *		}, not_l2);
	 *	}, l1 );
	 *
	 *	return( ret_list );
	 *}
	 */
	else if (ENTITY_AND_P( fun )) {
		expression 	exp4, exp5, exp6;
		list		l1, l2;
		
		l1 = ndf_normalized_test(arg1, ell);
		l2 = ndf_normalized_test(arg2, ell);
		MAPL( exp_ptr, {
			exp4 = EXPRESSION(CAR( exp_ptr ));
			MAPL( ep, {
			   exp5 = EXPRESSION(CAR( ep ));
			   exp6 = MakeBinaryCall( ENTITY_AND,
					exp4, exp5 );
			   ADD_ELEMENT_TO_LIST(ret_list, EXPRESSION, exp6);
			}, l2 );
		}, l1 );
		return( ret_list );
	}
	else return NIL;
}
	

/*=================================================================*/
/* expression normalize_test_leaves((expression) exp, (list) *ell) AL 04/93
 * If exp is linear in structurals and loop-counters, it returns the same
 * expression with linear positive forms
 */
expression normalize_test_leaves(exp, ell)
expression exp;
list *ell;
{
	syntax		s 	 = expression_syntax( exp );
	entity		fun  	 = call_function(syntax_call( s ));
	list		args 	 = call_arguments(syntax_call( s ));
	list		new_args = NIL;
	expression	ret_exp  = expression_undefined , ret_exp1  = expression_undefined, ret_exp2  = expression_undefined;
	expression 	e, ne, arg1, arg2, arg3, arg4;

	if (exp == expression_undefined || args == NIL) return( exp );
	pips_debug(7, "exp : %s\n",
		   words_to_string(words_expression( exp, NIL )) );
	if (syntax_tag( s ) != is_syntax_call) return( ret_exp );

	if (ENTITY_NOT_P( fun )) {
		expression exp1 =
		  normalize_test_leaves(EXPRESSION(CAR(args)), ell);

		/* We return expression_undefined if we can not normalize */
		if (exp1 == expression_undefined) return(expression_undefined);

		ret_exp = MakeUnaryCall(ENTITY_NOT, exp1 );
		pips_debug(7, "returning : %s\n",
			   words_to_string(words_expression( ret_exp, NIL )) );
		return( ret_exp );
	}

	arg1 = EXPRESSION(CAR(args));
	arg2 = EXPRESSION(CAR(CDR(args)));
	arg3 = copy_expression( arg1 );
	arg4 = copy_expression( arg2 );
	if (ENTITY_EQUIV_P( fun )) {
	  ret_exp1 = MakeBinaryCall( ENTITY_OR,
				     MakeUnaryCall( ENTITY_NOT, arg1),
				     arg2 );
	  ret_exp2 = MakeBinaryCall( ENTITY_OR,
				     MakeUnaryCall( ENTITY_NOT, arg4),
				     arg3 ) ;
	  exp = MakeBinaryCall( ENTITY_AND, ret_exp1, ret_exp2 );
	}
	else if (ENTITY_NON_EQUIV_P( fun )) {
	  MakeBinaryCall( ENTITY_AND,
			  MakeUnaryCall( ENTITY_NOT, arg2),
			  arg1 );
	  MakeBinaryCall( ENTITY_AND,
			  MakeUnaryCall( ENTITY_NOT, arg3),
			  arg4 ) ;
	  exp = MakeBinaryCall( ENTITY_OR, ret_exp1, ret_exp2);
	}
	
	s = expression_syntax( exp );
	fun = call_function(syntax_call( s ));
	args = call_arguments(syntax_call( s ));
	arg1 = EXPRESSION(CAR(args));
	arg2 = EXPRESSION(CAR(CDR(args)));
	arg3 = copy_expression( arg1 );
	arg4 = copy_expression( arg2 );
	if (ENTITY_STRICT_LOGICAL_OPERATOR_P( fun )) {
	  MAPL( exp_ptr, {
	      e = EXPRESSION(CAR( exp_ptr ));
	      ne = normalize_test_leaves(e, ell);
	      if (ne == expression_undefined) return(ne);
	      ADD_ELEMENT_TO_LIST(new_args, EXPRESSION,
				  copy_expression( ne ));
	    }, args);
	  call_arguments(syntax_call( s )) = new_args;
	  ret_exp = copy_expression( exp );
	  pips_debug(7, "returning : %s\n",
		     words_to_string(words_expression( ret_exp, NIL )) );
	  return( ret_exp );
	}
	else if (	ENTITY_RELATIONAL_OPERATOR_P( fun ) &&
			(!splc_linear_expression_p(arg1, ell) ||
			 !splc_linear_expression_p(arg2, ell)) ) {
	  
	  pips_debug(7, "returning : %s\n",
		     "expression_undefined" );
	  return( expression_undefined );
	}
	if (ENTITY_LESS_THAN_P( fun )) {
	  ret_exp = MakeBinaryCall( ENTITY_GE,
				    make_op_exp( MINUS_OPERATOR_NAME,
						 make_op_exp(
							     MINUS_OPERATOR_NAME,
							     arg2, arg1 ),
						 int_to_expression(1) ),
				    int_to_expression(0) );
	}
	else if (ENTITY_LESS_OR_EQUAL_P( fun )) {
	  ret_exp = MakeBinaryCall( ENTITY_GE,
				    make_op_exp( MINUS_OPERATOR_NAME,
						 arg2, arg1 ),
				    int_to_expression(0) );
	}
	else if (ENTITY_GREATER_THAN_P( fun )) {
	  ret_exp = MakeBinaryCall( ENTITY_GE,
				    make_op_exp( MINUS_OPERATOR_NAME,
						 make_op_exp(
							     MINUS_OPERATOR_NAME,
							     arg1, arg2 ),
						 int_to_expression(1) ),
				    int_to_expression(0) );
	}
	else if (ENTITY_GREATER_OR_EQUAL_P( fun )) {
	  ret_exp = MakeBinaryCall( (entity) ENTITY_GE,
				    make_op_exp( MINUS_OPERATOR_NAME,
						 arg1, arg2 ),
				    int_to_expression(0) );
	}
	else if (ENTITY_EQUAL_P( fun )) {
	  ret_exp1 = MakeBinaryCall( ENTITY_GE,
				     make_op_exp( MINUS_OPERATOR_NAME,
						  arg1, arg2 ),
				     int_to_expression(0) );
	  ret_exp2 = MakeBinaryCall( ENTITY_GE,
				     make_op_exp( MINUS_OPERATOR_NAME,
						  arg4, arg3 ),
				     int_to_expression(0) ) ;
	   ret_exp = MakeBinaryCall( ENTITY_AND, ret_exp1, ret_exp2);
	}
	else if (ENTITY_NON_EQUAL_P( fun )) {
	  ret_exp1 = MakeBinaryCall( ENTITY_GE,
                                make_op_exp( MINUS_OPERATOR_NAME,
					     make_op_exp( MINUS_OPERATOR_NAME,
                                        arg1, arg2 ),
				   int_to_expression(1) ),
				 int_to_expression(0) );
	  ret_exp2 =  MakeBinaryCall( ENTITY_GE,
                                make_op_exp( MINUS_OPERATOR_NAME,
				   make_op_exp( MINUS_OPERATOR_NAME,
                                        arg4, arg3 ),
				   int_to_expression(1) ),
				      int_to_expression(0) );
	  ret_exp = MakeBinaryCall( ENTITY_OR, ret_exp1, ret_exp2);
	}
	else ret_exp = expression_undefined;

	pips_debug(7, "returning : %s\n",
		((ret_exp == expression_undefined)?"expression_undefined":\
		 words_to_string(words_expression( ret_exp, NIL ))) );
	return( ret_exp );
}
			

/*=================================================================*/
/* expression sc_conditional( (expression) exp, (list) *ell ) AL 04/93 If
 * exp is linear in structurals and loop-counters, it returns the same
 * expression with a normal disjunctive form.
 */
expression sc_conditional(expression exp, list *ell)
{
	expression 	e, ret_exp = expression_undefined;
	syntax		s = expression_syntax( exp );
	list		ndf_list;

	pips_debug(7, "exp : %s\n",
		   words_to_string(words_expression(exp, NIL)));

	if ( syntax_tag(s) != is_syntax_call ) return( ret_exp );
	e = normalize_test_leaves(exp, ell);
	ndf_list = ndf_normalized_test(e, ell);
	if (ndf_list != NIL) {
		ret_exp = EXPRESSION(CAR( ndf_list ));
		ndf_list = CDR( ndf_list );
		MAPL( exp_ptr,{
			e = EXPRESSION(CAR( exp_ptr ));
			ret_exp = MakeBinaryCall( ENTITY_OR,
					copy_expression( ret_exp ), e );
		}, ndf_list );
	}

	pips_debug(7, "returning : %s\n",
		((ret_exp == expression_undefined)?"expression_undefined":
		 words_to_string(words_expression( ret_exp, NIL ))) );
	return( ret_exp );
}			
				
	
	
/*=================================================================*/
/* list loops_to_indices((loop) l )				AL 04/93
 * Returns indices of the loop -list l.
 */
list loops_to_indices(list l)
{
	list rl = NIL;
	loop lo;

	pips_debug(7, "doing\n");
	if (l == NIL) return(NIL);
	MAPL( loop_ptr, {
		lo = LOOP(CAR( loop_ptr ));
		ADD_ELEMENT_TO_LIST( rl, ENTITY, loop_index( lo ) );
	}, l);
	return rl;
}

/*=================================================================*/
/* bool splc_linear_expression_p((expression) exp) 		AL 04/93
 * Returns true if exp is linear in structural parameters and loop counters.
 */
bool splc_linear_expression_p(expression exp, list *ell)
{
  Pvecteur     vect;
  bool         ONLY_SPLC;

  pips_debug(7, "exp : %s\n",
	     words_to_string(words_expression(exp, NIL)));

  if(normalized_tag(NORMALIZE_EXPRESSION(exp)) == is_normalized_complex)
    ONLY_SPLC = false;
  else
  {
    vect = (Pvecteur) normalized_linear(expression_normalized(exp));
    ONLY_SPLC = true;

    for(; !VECTEUR_NUL_P(vect) && ONLY_SPLC ; vect = vect->succ)
    {
      entity var = (entity) vect->var;
      bool assigned_by_array = entity_assigned_by_array_p(var);
      if( ! term_cst(vect) )
	if(!(ENTITY_SP_P(var) || (gen_find_eq(var,loops_to_indices(*ell)) != chunk_undefined)) || assigned_by_array == true)
	  ONLY_SPLC = false;
    }
  }
  unnormalize_expression(exp);
  pips_debug(7,	"  result : %s\n", (ONLY_SPLC?"TRUE":"FALSE") );
  return(ONLY_SPLC);
}


/*=================================================================*/
/* bool splc_linear_expression_list_p((list) l)		AL 04/93
 * Returns true if all expressions exp are structural parameters
 * and loop counters linear functions.
 */
bool splc_linear_expression_list_p(list l, list * ell)
{
	bool		bo = true;
	expression	exp;

	pips_debug( 7, "doing \n");
	MAPL( exp_ptr, {
		exp = EXPRESSION(CAR( exp_ptr ));
		bo = bo && splc_linear_expression_p(exp, ell);
		}, l );
	return( bo );
}

/*=================================================================*/
/* bool splc_linear_access_to_arrays_p((list) l, (list) *ell) AL 04/93
 * Returns true if all expressions exp are structural parameters and loop
 * counters linear functions.
 */
bool splc_linear_access_to_arrays_p(list l, list * ell)
{
	bool		bo, ret_bo = true;
	expression	exp;
	syntax		s;
	tag		t;

	pips_debug(7, "doing\n");
	if (l == NIL) return(true);
	MAPL( exp_ptr, {
	  exp = EXPRESSION(CAR( exp_ptr ));
	  s   = expression_syntax( exp );
	  t   = syntax_tag( s );
	  if (t == is_syntax_call)
	    bo =
	      splc_linear_access_to_arrays_p(call_arguments(syntax_call(s)),
					     ell);
	  else if (t == is_syntax_reference)
	    bo =
	      splc_linear_expression_list_p(reference_indices(syntax_reference(s)),
					    ell);
		else bo = false;
		ret_bo = ret_bo && bo;
	}, l );

	return( ret_bo );
}

/*=================================================================*/
/* char* print_structurals( (list) l )				AL 04/93
 * Prints structural parameters.
 */
string print_structurals(list l)
{
	return(strdup( words_to_string(words_entity_list( l )) ));
}

/*=================================================================*/
/* list sc_list_of_exp_dup( (list) l )				AL 04/93
 * Duplicates a list of expressions.
 */
list sc_list_of_exp_dup(list l)
{
	list ret_list = NIL;

	pips_debug(9, "begin\n");
	for(; !ENDP( l ); POP( l ) ) {
		expression exp;

		exp = EXPRESSION(CAR( l ));
		ADD_ELEMENT_TO_LIST( ret_list, EXPRESSION, copy_expression(exp) );
	}

	pips_debug(9, "end\n");
	return( ret_list );
}

/*=================================================================*/
/* list sc_list_of_entity_dup( (list) l )			AL 04/93
 * Duplicates a list of entities.
 */
list sc_list_of_entity_dup(list l)
{
	list rl = NIL;

	pips_debug( 7, "doing\n");
	if ( l == NIL ) return( NIL );
	MAPL( ent_ptr, {
		entity ent = ENTITY(CAR( ent_ptr ));
		ADD_ELEMENT_TO_LIST( rl, ENTITY, ent );
	}, l );
	return( rl );
}

/*=================================================================*/
/* list sc_list_of_loop_dup( (list) l )				AL 04/93
 * Duplicates a list of loops. See Newgen gen_copy_seq()?
 */
list sc_list_of_loop_dup(list l)
{
	list rl = NIL;

	pips_debug( 7, "doing\n");
	if ( l == NIL ) return(NIL);
	MAPL( loop_ptr, {
		loop lo = LOOP(CAR( loop_ptr ));
		ADD_ELEMENT_TO_LIST( rl, LOOP, lo );
	}, l );
	return( rl );
}

/*=================================================================*/
/* list sc_loop_dup( (list) l )					AL 04/93
 * Duplicates a loop with sharing of the loop_body and sharing of the
 * loop locals.
 */
loop sc_loop_dup(loop l)
{
	loop new_loop;

	pips_debug( 7, "doing\n");
	new_loop = make_loop(loop_index(l), copy_range(loop_range(l)),
			loop_body(l), loop_label(l), loop_execution(l),
			loop_locals(l));

	return(new_loop);
}

/*=================================================================*/
/* list make_undefined_list( )					AL 04/93
 * Duplicates a list of 2 undefined statements.
 *
 * FI: this is no longer possible. List elements must be
 * defined. Maybe empty/nop statements could be used instead?
 */
list make_undefined_list()
{
	list the_list = NIL;

	pips_debug(7, "doing\n");
	/*
	ADD_ELEMENT_TO_LIST( the_list, STATEMENT, statement_undefined);
	ADD_ELEMENT_TO_LIST( the_list, STATEMENT, statement_undefined);
	*/
	ADD_ELEMENT_TO_LIST( the_list, STATEMENT,
			     make_continue_statement(entity_empty_label()));
	ADD_ELEMENT_TO_LIST( the_list, STATEMENT,
			     make_continue_statement(entity_empty_label()));
	return the_list;
}


/*=================================================================*/
/* int in_forward_defined( (entity) ent )			AL 30/08/93
 * Returns the number of entities ent in the list Gscalar_written_forward.
 */
int in_forward_defined(entity ent, list *swfl)
{
    cons *pc;
    int  ret_int = 0;

    pips_debug(9, "doing \n");
    for (pc = *swfl; pc != NIL; pc = pc->cdr ) {
        if ((void *) ent == CAR(pc).p)
                ret_int++;
    }

    pips_debug(9, "returns : %d\n", ret_int);
    return( ret_int );
}

/*=================================================================*/
/* bool in_forward_defined_p( (entity) ent )			AL 04/93
 * Returns true if ent is in global variable Gscalar_written_forward.
 */
bool in_forward_defined_p(entity ent, list * swfl)
{
	entity ch;

	pips_debug(7, "doing \n");
	ch = (entity) gen_find_eq(ent, *swfl );
	pips_debug(9, "scalar written_forward = %s\n",
			print_structurals(*swfl) );
	return( ch != entity_undefined );
}

/*=================================================================*/
/* void verify_structural_parameters( (list) l )		AL 04/93
 * Updates the global variable Gstructure_parameters.
 * 'l' is a list of entities, which are structural-parameters candidates.
 * An entity will be a structural parameter if it is a candidate and if it
 * is not written forward.
 */
void verify_structural_parameters(list the_list, list *swfl)
{
	pips_debug(7,"doing\n");
	MAPL( el_ptr,
		{
		entity ent = ENTITY(CAR( el_ptr ));
		if (   gen_find_eq( ent, *swfl )
		    == chunk_undefined )
			ADD_ELEMENT_TO_LIST( Gstructure_parameters,
					     ENTITY,
					     ent );
		},
	      the_list);
	pips_debug(7, "list of structurals : %s\n",
		   print_structurals(Gstructure_parameters) );
}

/*=================================================================*/
/* list sc_entity_to_formal_integer_parameters((entity) f)
 * This is a strict copy of entity_to_formal_integer_parameters
 * from semantics/interprocedural.c . This function is copied
 * here to keep locality of local functions to each pass.
 *
 * FI: Well, but it was modified to handle all integer scalar variables
 * and a bug was added because a COMMON has a RAM storage (OK, this could
 * be discussed). I add a test on type_variable_p()
 */
list sc_entity_to_formal_integer_parameters(f)
entity f;
{
    list formals_or_ram_integer = NIL;
    list decl = NIL;

    pips_assert("sc_entity_to_formal_integer_parameters",entity_module_p(f));

    decl = code_declarations(entity_code(f));
    MAPL(ce, {entity e = ENTITY(CAR(ce));
	      storage sto = entity_storage(e);
	      type t = entity_type(e);

              if( type_variable_p(t)
		  && (storage_formal_p(sto) || storage_ram_p(sto))
		  && entity_integer_scalar_p(e))
                  formals_or_ram_integer = CONS(ENTITY, e,
						formals_or_ram_integer);},
         decl);

    return formals_or_ram_integer;
}

/*=================================================================*/
/* bool sp_linear_expression_p( (expression) exp)
 * Returns true if the expression is a linear combinaison of
 * structural parameters.
 */
bool sp_linear_expression_p(expression exp)
{
   Pvecteur     vect;
   bool         ONLY_SP;

   pips_debug(7, "exp : %s\n",
	      words_to_string(words_expression(exp, NIL)));

   if(normalized_tag(NORMALIZE_EXPRESSION(exp)) == is_normalized_complex)
        ONLY_SP = false;
   else
   {
        vect = (Pvecteur) normalized_linear(expression_normalized(exp));
        ONLY_SP = true;

        for(; !VECTEUR_NUL_P(vect) && ONLY_SP ; vect = vect->succ)
        {
                entity var = (entity) vect->var;

                if( ! term_cst(vect) )
                if( ! (ENTITY_SP_P(var)) )
                        ONLY_SP = false;
        }
   }
   unnormalize_expression(exp);
   pips_debug(7,
		 "  result : %s\n", (ONLY_SP?"TRUE":"FALSE") );
   return(ONLY_SP);
}

/*=================================================================*/
/* bool splc_feautrier_expression_p( (expression) exp )
 * Returns true if exp quasi affine form in structural parameters
 * and in surrounding loop-counters.
 */
bool splc_feautrier_expression_p(expression exp, list * ell)
{
	bool b = false;
	syntax s = expression_syntax( exp );

	pips_debug(7, "exp : %s \n",
		((exp == expression_undefined)?"expression_undefined":
		 words_to_string( words_expression( exp, NIL ) ) ));

	if (splc_linear_expression_p(exp, ell)) return( true );
	if ( syntax_tag( s ) == is_syntax_call ) {
		call c;
		list args;
		expression exp1, exp2;

		c = syntax_call( s );
		if (ENTITY_DIVIDE_P(call_function( c ))) {
			args = call_arguments( c );
			exp1 = EXPRESSION(CAR( args ));
			exp2 = EXPRESSION(CAR( CDR(args) ));
			b    = splc_feautrier_expression_p(exp1, ell)
			       && expression_constant_p( exp2 );
		}
	}
	pips_debug(7, "returning : %s\n",
			(b?"TRUE":"FALSE") );
	return( b );
}

/*=================================================================*/
/* bool sp_feautrier_expression_p( (expression) exp)
 * Returns true if exp quasi affine form.
 */
bool sp_feautrier_expression_p(expression exp)
{
	bool b = false;
	syntax s = expression_syntax( exp );

	pips_debug(7, "exp : %s \n",
		((exp == expression_undefined)?"expression_undefined":
		 words_to_string( words_expression( exp, NIL ) ) ));

	if (sp_linear_expression_p( exp )) return( true );
	if ( syntax_tag( s ) == is_syntax_call ) {
		call c;
		list args;
		expression exp1, exp2;

		c = syntax_call( s );
		if (ENTITY_DIVIDE_P(call_function( c ))) {
			args = call_arguments( c );
			exp1 = EXPRESSION(CAR( args ));
			exp2 = EXPRESSION(CAR( CDR(args) ));
			b    = sp_feautrier_expression_p( exp1 )
			  && expression_constant_p( exp2 );
		}
	}
	pips_debug(7, "returning : %s\n",
			(b?"TRUE":"FALSE") );
	return( b );
}

/*=================================================================*/
/* entity  sp_feautrier_scalar_assign_call( (call) c )
 * Returns the left-hand-side entity if it is an assignement of
 * a linear combinaison of structural parameters.
 */
entity sp_feautrier_scalar_assign_call(call c)
{
	entity 		ent, ret_ent = entity_undefined;
	expression 	rhs;

	pips_debug(7, "doing\n");
	if ((ent = scalar_assign_call(c)) != entity_undefined) {
		rhs = EXPRESSION(CAR(CDR(call_arguments(c))));
		if (sp_feautrier_expression_p( rhs )) ret_ent = ent;
	}
	pips_debug(7,
		"returning : %s \n",
		((ret_ent == entity_undefined)?"entity_undefined":
		entity_name( ret_ent )) );
	return( ret_ent );
}
		

/*=================================================================*/
/* bool get_sp_of_call_p( (call) c, fst) AL 04/93 Updates the global
 * variables Gstructure_parameters and Gforward_substitute_table according
 * to the type of call.  Returns true if the call has to be modified
 * (redefinition of a structural parameter), false in all the other cases.
 *
 * AP, sep 95 : Gforward_substitute_table is no longer a global variable,
 * we pass it as an argument.
 */
bool get_sp_of_call_p( c, fst, swfl)
call c;
hash_table fst; /* forward substitute table */
list *swfl;
{
   entity	lhs_ent, ent;
   bool		ret_bool = false;

   pips_debug(7, "begin\n");
   /*FI: not too sure about the false parameter... */
   pips_debug(9, "input call : %s \n",
	      words_to_string(words_regular_call( c, false, NIL )));
   pips_debug(9, "struct param. before : %s \n",
			print_structurals( Gstructure_parameters ));

   if (ENTITY_READ_P( call_function(c) )) {
        list the_arg = call_arguments( c );

        MAPL( exp_ptr,
	      {
	  expression exp = EXPRESSION(CAR( exp_ptr ));
	  ent = expression_int_scalar( (expression) exp );
	  if ((ent != entity_undefined)  && !in_forward_defined_p(ent, swfl))
		ADD_ELEMENT_TO_LIST( Gstructure_parameters, ENTITY, ent );
          },
	      the_arg);
   }

   if (    ((lhs_ent = sp_feautrier_scalar_assign_call(c)) != entity_undefined)
	&& (in_forward_defined(lhs_ent, swfl) <= 1) )	{

	expression nsp_exp;
	entity     nsp_ent;

	if ( !ENTITY_SP_P( lhs_ent ) ) {
		ADD_ELEMENT_TO_LIST( Gstructure_parameters, ENTITY, lhs_ent );
	}
	else {
	        nsp_ent = make_nsp_entity();
		nsp_exp = make_entity_expression( nsp_ent, NIL );
		hash_put(fst, (char*) lhs_ent, (char*) nsp_exp  );
		ADD_ELEMENT_TO_LIST( Gstructure_parameters, ENTITY, nsp_ent );
		ret_bool = true;
	}
   }

   pips_debug(9, "struct param. after  : %s \n",
			print_structurals( Gstructure_parameters ));
   pips_debug(9, "call has to be modified : %s \n",
		((ret_bool == true)?"TRUE":"FALSE") );
   pips_debug(7, "end\n");
   return( ret_bool );
}


/* rewriting of forward_substitute_in_exp, in_loop,...
 *
 * FI: what was the initial semantics? Substitute the left hand sides
 * only?
 *
 * see transformations/induction_substitution.c
 *
 * It's a simple case because the loop indices cannot be written
 * within the loop bounds
*/

static void substitute_variable_by_expression(expression e, hash_table fst)
{
  syntax s = expression_syntax(e);
  if(syntax_reference_p(s)) {
    entity v = reference_variable(syntax_reference(s));
    expression ne = hash_get(fst, (void *) v);

    if(!expression_undefined_p(ne)) {
      /* FI: let's take care of memory leaks later...
       *
       * The hash_table could contain syntax objects...
       */
      expression_syntax(e) = expression_syntax(ne);
    }
  }
}

void forward_substitute_in_anyloop(void *pl, hash_table fst)
{
  /* Would it be better to loop over with HASH_MAP or to check every
     reference? */
  gen_context_multi_recurse(pl, fst,
			    expression_domain, gen_true, substitute_variable_by_expression,
			    NULL);
}

void forward_substitute_in_exp(expression * pe, hash_table fst)
{
  /* Would it be better to loop over with HASH_MAP or to check every
     reference? */
  gen_context_multi_recurse(*pe, fst,
			    expression_domain, gen_true, substitute_variable_by_expression,
			    NULL);
}

void forward_substitute_in_call(call * pc, hash_table fst)
{
  pips_assert("call c is consistent_p\n", call_consistent_p(*pc));

  /* Would it be better to loop over with HASH_MAP or to check every
     reference? */
  gen_context_multi_recurse(*pc, fst,
			    expression_domain, gen_true, substitute_variable_by_expression,
			    NULL);
}


/* bool normalizable_loop_p(loop l)
 * Returns true if "l" has a constant step.
 */
bool normalizable_loop_p(loop l)
{
  pips_debug(7, "doing\n");
  return(expression_constant_p(range_increment(loop_range(l))));
}


/* Code retrieved from revision 14476,
   transformations/loop_normalize.c */

/* Code to be retrieved: I suppose you need a constant increment? */
bool normalizable_loop_p_retrieved(loop l)
{
  bool ok = false;
  entity i = loop_index(l);
  range r = loop_range(l);

  ok = normalizable_and_linear_loop_p(i, r);

  return ok;
}

/* State if an expression OR part of that expression corresponds to one
   of the entity of the list l.
   This function was specifically designed in order to find if an expression
   is using one of the variable/entity listed in l.
   The function considers that the list only contains scalar variables.
*/
 bool is_expression_in_list (expression exp, list l) {
   if (l == NIL)
     return false;
   syntax s = expression_syntax(exp);
   switch (syntax_tag(s)) {
     // If the expression is an array we go look among the indices by calling
     // the function on these indices
   case is_syntax_reference :
     {
       reference ref = syntax_reference(s);
       if (reference_indices(ref) != NIL) {
	 FOREACH(expression, e, reference_indices(ref)) {
	   if(is_expression_in_list(e, l))
	     return true;
	 }
       }
       else {
	 entity ent = reference_variable(ref);
	 if (gen_find_eq(ent, l) != chunk_undefined)
	   return true;
       }
     }
     break;
     // Same principle, we go look among the arguments of the call
   case is_syntax_call :
     {
       call c = syntax_call(s);
       FOREACH(expression, e, call_arguments(c)) {
	 if(is_expression_in_list(e, l))
	   return true;
       }
     }
     break;
   case is_syntax_cast :
     {
       cast ca = syntax_cast(s);
       entity ent = expression_to_entity(cast_expression(ca));
       if (gen_find_eq(ent, l) != chunk_undefined) {
	 return true;
       }
     }
     break;
     // We call the function on every component of the range object
   case is_syntax_range :
     {
       range ra = syntax_range(s);
       return is_expression_in_list(range_lower(ra), l) || is_expression_in_list(range_upper(ra), l) || is_expression_in_list(range_increment(ra), l);
     }
     break;
   case is_syntax_subscript :
     {
       subscript sub = syntax_subscript(s);
       bool isinArray = is_expression_in_list(subscript_array(sub), l);
       if (isinArray)
	 return true;
       FOREACH(expression, e, subscript_indices(sub)) {
	 if (is_expression_in_list(e, l)) {
	   return true;
	 }
       }
     }
     break;
   case is_syntax_application :
     {
       application app = syntax_application(s);
       bool isinFunction = is_expression_in_list(application_function(app), l);
       if (isinFunction)
	 return true;
       FOREACH(expression, e, application_arguments(app)) {
	 if (is_expression_in_list(e, l)) {
	   return true;
	 }
       }
     }
     break;
   case is_syntax_va_arg :
     {
       list sil = syntax_va_arg(s);
       FOREACH(sizeofexpression, soe, sil) {
	 if (is_expression_in_list(sizeofexpression_expression(soe), l))
	   return true;
       }
     }
     break;
   case is_syntax_sizeofexpression :
     {
       sizeofexpression si = syntax_sizeofexpression(s);
       if (is_expression_in_list(sizeofexpression_expression(si), l))
	 return true;
     }
     break;
   default :
     return false;
   }
   return false;
 }

/* Allows the static_controlize phase to keep and update a list containing
   all the variables of the program assigned directly or indirectly by an array
*/
bool get_reference_assignments (statement s, list* l) {
  // we only compute assignments
  if (!assignment_statement_p(s)) {
    return false;
  }
  instruction inst = statement_instruction(s);
  // May not be necessary
  if (!instruction_call_p(inst)) {
    return false;
  }
  // Being an assignment, logically this call has only two arguments
  call c = instruction_call(inst);
  list args = call_arguments(c);
  // Left member of the assignment
  expression left = gen_car(args);
  entity eleft = expression_to_entity(left);
  // Right member of the assignment
  expression right = gen_car(CDR(args));
  // If the assigned variable is not in the list
  if (gen_find_eq(eleft, *l) == chunk_undefined) {
    // Test if the assignment is static control
    entity e = sp_feautrier_scalar_assign_call(c);
    syntax s = expression_syntax(left);
    // We look for the right member in the list in order to know if the variable read has not been assigned by an array
    bool isRightArrayAccess = is_expression_in_list(right, *l);
    // If the left member is a scalar and if the right member is not static control or if it is a variable previously assigned
    // by an array, then we add the left member to the list
    if ((syntax_reference_p(s) && reference_indices(syntax_reference(s)) == NIL) && (entity_undefined_p(e) || isRightArrayAccess == true)) {
      *l = gen_cons(eleft, *l);    
    }
  }
  return true;
}
