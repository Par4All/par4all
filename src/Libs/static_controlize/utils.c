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
#include <malloc.h>
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
#include "graph.h"
#include "paf_ri.h"
#include "database.h"
#include "parser_private.h"
#include "syntax.h"
#include "ri-util.h"
#include "constants.h"
#include "misc.h"
#include "control.h"
#include "text.h" 
#include "text-util.h"
#include "paf-util.h"
#include "static_controlize.h"

/* Global variables 	*/
/* extern	list		Gscalar_written_forward; */
extern 	list		Gstructure_parameters;
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

        debug(7, "stco_same_loops","begin\n");
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
        debug(7, "stco_same_loops", "end \n");

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

        debug(7, "stco_renumber_code", "begin\n");
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

        debug(7, "stco_renumber_code", "return count : %d\n", count);
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

        debug(9, "sc_opposite_exp_of_conjunction", "begin\n");
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
				make_integer_constant_expression(-1), exp1 ),
			   make_integer_constant_expression(0) );
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

	debug(9, "sc_opposite_exp_of_conjunction", "end\n");
	return( ret_exp );
}


/*=================================================================*/
/* bool splc_positive_relation_p((expression) exp, list *ell) AL 04/93
 * Returns TRUE if exp is an affine form of structural parameters and of
 * loop-counters.
 */
bool splc_positive_relation_p(exp, ell)
expression exp;
list *ell;
{
	syntax s = expression_syntax( exp );
	call c;
	list args;

	debug(7, "splc_positive_relation_p", "exp : %s\n",
			words_to_string( words_expression( exp ) ));
	if (syntax_tag(s) != is_syntax_call) return( FALSE );
	c = syntax_call( s );
	if (!ENTITY_GREATER_OR_EQUAL_P(call_function( c ))) return( FALSE );
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
list 	ndf_normalized_test(exp, ell)
expression exp;
list *ell;
{
	list 		args, ret_list = NIL;
	entity		fun;
	expression 	arg1, arg2, exp2, exp3;

	debug(7, "ndf_normalized_test", "doing\n");
	if ( (exp == expression_undefined) ||
		(syntax_tag(expression_syntax( exp )) != is_syntax_call) ) {
		return( list_undefined );
	}
	debug(7, "ndf_normalized_test", "input exp : %s\n",
			words_to_string(words_expression( exp )) );

	fun = call_function(syntax_call(expression_syntax( exp )));
	args = call_arguments(syntax_call(expression_syntax( exp )));
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
				   make_integer_constant_expression(-1),
				   exp3 ),
				 make_integer_constant_expression(0) );
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
						   expression_dup( exp2 ) );
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
	else return( list_undefined );
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
	expression	ret_exp  = expression_undefined;
	expression 	e, ne, arg1, arg2;

	if (exp == expression_undefined) return( exp );
	debug(7, "normalize_test_leaves", "exp : %s\n",
			words_to_string(words_expression( exp )) );
	if (syntax_tag( s ) != is_syntax_call) return( ret_exp );

	if (ENTITY_NOT_P( fun )) {
		expression exp1 =
		  normalize_test_leaves(EXPRESSION(CAR(args)), ell);

		/* We return expression_undefined if we can not normalize */
		if (exp1 == expression_undefined) return(expression_undefined);
		
		ret_exp = MakeUnaryCall(ENTITY_NOT, exp1 );
		debug(7, "normalize_test_leaves", "returning : %s\n",
			words_to_string(words_expression( ret_exp )) );
		return( ret_exp );
	}

	arg1 = EXPRESSION(CAR(args));
	arg2 = EXPRESSION(CAR(CDR(args)));
	if (ENTITY_EQUIV_P( fun )) {
		exp = MakeBinaryCall( ENTITY_AND,
			MakeBinaryCall( ENTITY_OR,
				MakeUnaryCall( ENTITY_NOT, arg1),
				arg2 ),
			MakeBinaryCall( ENTITY_OR,
				MakeUnaryCall( ENTITY_NOT, arg2),
				arg1 ) );
	}
	else if (ENTITY_NON_EQUIV_P( fun )) {
		exp = MakeBinaryCall( ENTITY_OR,
			MakeBinaryCall( ENTITY_AND,
				MakeUnaryCall( ENTITY_NOT, arg2),
				arg1 ),
			MakeBinaryCall( ENTITY_AND,
				MakeUnaryCall( ENTITY_NOT, arg1),
				arg2 ) );
	}

	s = expression_syntax( exp );
	fun = call_function(syntax_call( s ));
	args = call_arguments(syntax_call( s ));
	arg1 = EXPRESSION(CAR(args));
	arg2 = EXPRESSION(CAR(CDR(args)));
	if (ENTITY_STRICT_LOGICAL_OPERATOR_P( fun )) {
		MAPL( exp_ptr, {
			e = EXPRESSION(CAR( exp_ptr ));
			ne = normalize_test_leaves(e, ell);
			if (ne == expression_undefined) return(ne);
			ADD_ELEMENT_TO_LIST(new_args, EXPRESSION, 
					expression_dup( ne ));
			}, args);
		call_arguments(syntax_call( s )) = new_args;
		ret_exp = expression_dup( exp );
		debug(7, "normalize_test_leaves", "returning : %s\n",
			words_to_string(words_expression( ret_exp )) );
		return( ret_exp );
	}
	else if (	ENTITY_RELATIONAL_OPERATOR_P( fun ) &&
		(!splc_linear_expression_p(arg1, ell) || 
		 !splc_linear_expression_p(arg2, ell)) ) {

		debug(7, "normalize_test_leaves", "returning : %s\n",
					"expression_undefined" );
		return( expression_undefined );
	}

	if (ENTITY_LESS_THAN_P( fun )) {
		ret_exp = MakeBinaryCall( ENTITY_GE,
				make_op_exp( MINUS_OPERATOR_NAME,
					make_op_exp( 
						MINUS_OPERATOR_NAME,
						arg2, arg1 ),
					make_integer_constant_expression(1) ),
				make_integer_constant_expression(0) );
	}
	else if (ENTITY_LESS_OR_EQUAL_P( fun )) {
                ret_exp = MakeBinaryCall( ENTITY_GE,
				make_op_exp( MINUS_OPERATOR_NAME,
					arg2, arg1 ),
				make_integer_constant_expression(0) );
	}
	else if (ENTITY_GREATER_THAN_P( fun )) {
                ret_exp = MakeBinaryCall( ENTITY_GE,
				make_op_exp( MINUS_OPERATOR_NAME,
					make_op_exp( 
						MINUS_OPERATOR_NAME,
						arg1, arg2 ),
					make_integer_constant_expression(1) ),
				make_integer_constant_expression(0) );
	}
	else if (ENTITY_GREATER_OR_EQUAL_P( fun )) {
		ret_exp = MakeBinaryCall( (entity) ENTITY_GE,
				make_op_exp( MINUS_OPERATOR_NAME,
					arg1, arg2 ),
				make_integer_constant_expression(0) );
	}
	else if (ENTITY_EQUAL_P( fun )) {
                ret_exp = MakeBinaryCall( ENTITY_AND,
			    MakeBinaryCall( ENTITY_GE,
				make_op_exp( MINUS_OPERATOR_NAME,
					arg1, arg2 ),
				make_integer_constant_expression(0) ), 
			    MakeBinaryCall( ENTITY_GE, 
				make_op_exp( MINUS_OPERATOR_NAME,
					arg2, arg1 ),
				make_integer_constant_expression(0) ) );
	}
	else if (ENTITY_NON_EQUAL_P( fun )) {
                ret_exp = MakeBinaryCall( ENTITY_OR,
                            MakeBinaryCall( ENTITY_GE,
                                make_op_exp( MINUS_OPERATOR_NAME,
				   make_op_exp( MINUS_OPERATOR_NAME,
                                        arg1, arg2 ),
				   make_integer_constant_expression(1) ),
                                make_integer_constant_expression(0) ),
                           MakeBinaryCall( ENTITY_GE, 
                                make_op_exp( MINUS_OPERATOR_NAME,
				   make_op_exp( MINUS_OPERATOR_NAME,
                                        arg2, arg1 ),
				   make_integer_constant_expression(1) ),
                                make_integer_constant_expression(0) ) );
	}
	else ret_exp = expression_undefined;

	debug(7, "normalize_test_leaves", "returning : %s\n",
		((ret_exp == expression_undefined)?"expression_undefined":\
			words_to_string(words_expression( ret_exp ))) );
	return( ret_exp );
}
			

/*=================================================================*/
/* expression sc_conditional( (expression) exp, (list) *ell ) AL 04/93 If
 * exp is linear in structurals and loop-counters, it returns the same
 * expression with a normal disjunctive form.
 */
expression sc_conditional(exp, ell)
expression exp;
list *ell;
{
	expression 	e, ret_exp = expression_undefined;
	syntax		s = expression_syntax( exp );
	list		ndf_list;

	debug(7, "sc_conditional", "exp : %s\n", 
				words_to_string(words_expression(exp)));

	if ( syntax_tag(s) != is_syntax_call ) return( ret_exp );
	e = normalize_test_leaves(exp, ell);
	ndf_list = ndf_normalized_test(e, ell);
	if (ndf_list != list_undefined) {
		ret_exp = EXPRESSION(CAR( ndf_list ));
		ndf_list = CDR( ndf_list );
		MAPL( exp_ptr,{
			e = EXPRESSION(CAR( exp_ptr ));
			ret_exp = MakeBinaryCall( ENTITY_OR,
					expression_dup( ret_exp ), e );
		}, ndf_list );
	}

	debug(7, "sc_conditional", "returning : %s\n",
		((ret_exp == expression_undefined)?"expression_undefined":
		words_to_string(words_expression( ret_exp ))) );
	return( ret_exp );
}			
				
	
	
/*=================================================================*/
/* list loops_to_indices((loop) l )				AL 04/93
 * Returns indices of the loop -list l.
 */
list loops_to_indices( l )
list l;
{
	list rl = NIL;
	loop lo;

	debug(7, "loops_to_indices", "doing\n");
	if (l == NIL) return(NIL);
	MAPL( loop_ptr, {
		lo = LOOP(CAR( loop_ptr ));
		ADD_ELEMENT_TO_LIST( rl, ENTITY, loop_index( lo ) );
	}, l);
	return rl;
}

/*=================================================================*/
/* bool splc_linear_expression_p((expression) exp) 		AL 04/93
 * Returns TRUE if exp is linear in structural parameters and loop counters.
 */
bool splc_linear_expression_p(exp, ell)
expression exp;
list *ell;
{
  Pvecteur     vect;
  bool         ONLY_SPLC;

  debug(7, "splc_linear_expression_p", "exp : %s\n",
	words_to_string(words_expression(exp)));

  if(normalized_tag(NORMALIZE_EXPRESSION(exp)) == is_normalized_complex)
    ONLY_SPLC = FALSE;
  else
  {
    vect = (Pvecteur) normalized_linear(expression_normalized(exp));
    ONLY_SPLC = TRUE;

    for(; !VECTEUR_NUL_P(vect) && ONLY_SPLC ; vect = vect->succ)
    {
      entity var = (entity) vect->var;

      if( ! term_cst(vect) )
	if(!(ENTITY_SP_P(var) ||
           (gen_find_eq(var,loops_to_indices(*ell)) != chunk_undefined)))
	  ONLY_SPLC = FALSE;
    }
  }
  unnormalize_expression(exp);
  debug(7, "splc_linear_expression_p",
	"  result : %s\n", (ONLY_SPLC?"TRUE":"FALSE") );
  return(ONLY_SPLC);
}


/*=================================================================*/
/* bool splc_linear_expression_list_p((list) l) 		AL 04/93
 * Returns TRUE if all expressions exp are structural parameters 
 * and loop counters linear functions.
 */
bool splc_linear_expression_list_p(l, ell)
list l, *ell;
{
	bool		bo = TRUE;
	expression 	exp;

	debug( 7, "splc_linear_expression_list_p", "doing \n");
	MAPL( exp_ptr, {
		exp = EXPRESSION(CAR( exp_ptr ));
		bo = bo && splc_linear_expression_p(exp, ell);
		}, l );
	return( bo );
}

/*=================================================================*/
/* bool splc_linear_access_to_arrays_p((list) l, (list) *ell) AL 04/93
 * Returns TRUE if all expressions exp are structural parameters and loop
 * counters linear functions.
 */
bool splc_linear_access_to_arrays_p(l, ell)
list l, *ell;
{
	bool 		bo, ret_bo = TRUE;
	expression 	exp;
	syntax		s;
	tag		t;

	debug(7, "splc_linear_access_to_arrays_p", "doing\n");
	if (l == NIL) return(TRUE);
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
		else bo = FALSE;
		ret_bo = ret_bo && bo;
	}, l );
	
	return( ret_bo );
}

/*=================================================================*/
/* char* print_structurals( (list) l )				AL 04/93
 * Prints structural parameters.
 */
char* print_structurals( l )
list l;
{
	return(strdup( words_to_string(words_entity_list( l )) ));
}

/*=================================================================*/
/* list sc_list_of_exp_dup( (list) l )				AL 04/93
 * Duplicates a list of expressions.
 */
list sc_list_of_exp_dup( l )
list l;
{
	list ret_list = NIL;

	debug(9, "sc_list_of_exp_dup", "begin\n");
	for(; !ENDP( l ); POP( l ) ) {
		expression exp;

		exp = EXPRESSION(CAR( l ));
		ADD_ELEMENT_TO_LIST( ret_list, EXPRESSION, expression_dup(exp) );
	}

	debug(9, "sc_list_of_exp_dup", "end\n");
	return( ret_list );
}
	
/*=================================================================*/
/* list sc_list_of_entity_dup( (list) l )			AL 04/93
 * Duplicates a list of entities.
 */
list sc_list_of_entity_dup( l )
list l;
{
	list rl = NIL;

	debug( 7, "sc_list_of_entity_dup", "doing\n");
	if ( l == NIL ) return( NIL );
	MAPL( ent_ptr, {
		entity ent = ENTITY(CAR( ent_ptr ));
		ADD_ELEMENT_TO_LIST( rl, ENTITY, ent );
	}, l );
	return( rl );
}
 
/*=================================================================*/
/* list sc_list_of_loop_dup( (list) l )				AL 04/93
 * Duplicates a list of loops.
 */
list sc_list_of_loop_dup( l )
list l;
{
	list rl = NIL;

	debug( 7, "sc_list_of_loop_dup", "doing\n");
	if ( l == NIL ) return(NIL);
	MAPL( loop_ptr, {
		loop lo = LOOP(CAR( loop_ptr ));
		ADD_ELEMENT_TO_LIST( rl, LOOP, lo );
	}, l );
	return( rl );
}
	
/*=================================================================*/
/* list sc_loop_dup( (list) l )					AL 04/93
 * Duplicates a loop.
 */
loop sc_loop_dup( l )
loop l;
{
	loop new_loop;

	debug( 7, "sc_loop_dup", "doing\n");
	new_loop = make_loop(loop_index(l), range_dup(loop_range(l)), 
			loop_body(l), loop_label(l), loop_execution(l),
			loop_locals(l));

	return(new_loop);

}
	
/*=================================================================*/
/* list make_undefined_list( )					AL 04/93
 * Duplicates a list of 2 undefined statements.
 */
list make_undefined_list()
{
	list the_list = NIL;

	debug(7, "make_undefined_list", "doing\n");
	ADD_ELEMENT_TO_LIST( the_list, STATEMENT, statement_undefined);
	ADD_ELEMENT_TO_LIST( the_list, STATEMENT, statement_undefined);
	return( the_list );
}


/*=================================================================*/
/* int in_forward_defined( (entity) ent ) 			AL 30/08/93
 * Returns the number of entities ent in the list Gscalar_written_forward.
 */
int in_forward_defined( ent, swfl)
entity ent;
list *swfl;
{
    cons *pc;
    int  ret_int = 0;

    debug(9, "in_forward_defined", "doing \n");
    for (pc = *swfl; pc != NIL; pc = pc->cdr ) {
        if ((chunk*) ent == CAR(pc).p)
                ret_int++;
    }

    debug(9, "in_forward_defined", "returns : %d\n", ret_int);
    return( ret_int );
}

/*=================================================================*/
/* bool in_forward_defined_p( (entity) ent )			AL 04/93
 * Returns TRUE if ent is in global variable Gscalar_written_forward.
 */
bool in_forward_defined_p( ent, swfl)
entity ent;
list *swfl;
{
	chunk* ch;

	debug( 7, "in_forward_defined_p", "doing \n");  
	ch = gen_find_eq( ent, *swfl );
	debug( 9, "in_forward_defined_p", "scalar written_forward = %s\n",
			print_structurals(*swfl) );
	return( ch != chunk_undefined );
}
	
/*=================================================================*/
/* bool undefined_statement_list_p( (list) l )			AL 04/93
 * Returns TRUE if l is made of 2 undefined_statement.
 */
bool undefined_statement_list_p( l ) 
list l;
{
	bool 		local_bool;
	statement 	first, second;

	debug(7, "undefined_statement_list_p","doing\n");
	if ( (l == NIL) || (gen_length(l) != 2) )
		return( FALSE );

	first = STATEMENT(CAR( l ));
	second = STATEMENT(CAR(CDR( l )));
	local_bool = ( first == statement_undefined ) 
		     && ( second == statement_undefined );
	return( local_bool );
}

/*=================================================================*/
/* void verify_structural_parameters( (list) l )		AL 04/93
 * Updates the global variable Gstructure_parameters.
 * 'l' is a list of entities, which are structural-parameters candidates.
 * An entity will be a structural parameter if it is a candidate and if it
 * is not written forward.
 */
void verify_structural_parameters( the_list, swfl)
list the_list;
list *swfl;
{
	debug(7, "verify_structural_parameters","doing\n");
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
	debug(7, "verify_structural_parameters","list of structurals : %s\n",
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
    list decl = list_undefined;

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
/* entity scalar_assign_call((call) c) 
 * Detects if the call is an assignement
 * and if the value assigned is a scalar. If it is so, it
 * returns this scalar.
 */
entity scalar_assign_call( c )
call c;
{
   entity ent = entity_undefined;

   debug( 7, "scalar_assign_call", "doing \n");
   if (ENTITY_ASSIGN_P(call_function(c)))
        {
        expression lhs;

        lhs = EXPRESSION(CAR(call_arguments(c)));
	ent = expression_int_scalar( lhs );
	}
   debug( 7, "scalar_assign_call", "returning : %s \n",
	((ent == entity_undefined)?"entity_undefined":
				entity_name(ent)) );
   return( ent );
}

	
/*=================================================================*/
/* scalar_written_in_call((call) the_call) 
 * Detects and puts a scalar written in an assignement call,
 * in the global list Gscalar_written_forward if Genclosing_loops
 * or Genclosing_tests are not empty.
 */
void scalar_written_in_call( the_call, ell, etl, swfl)
call the_call;
list *ell, *etl, *swfl;
{
   entity ent;

   debug( 7, "scalar_written_in_call", "doing\n");
   if (    ((ent = scalar_assign_call(the_call)) != entity_undefined)
        && ( (*ell != NIL) || (*etl != NIL) )
	&& entity_integer_scalar_p( ent ) )

	ADD_ELEMENT_TO_LIST(*swfl, ENTITY, ent);
}

/*=================================================================*/
/* entity  expression_int_scalar((expression) exp)
 * Returns the scalar entity if this expression is a scalar.
 */
entity expression_int_scalar( exp )
expression exp;
{
        syntax  s = expression_syntax( exp );
        tag     t = syntax_tag( s );
        entity 	ent = entity_undefined;

	debug( 7, "expression_int_scalar", "doing \n");
        switch( t ) {
                case is_syntax_reference: {
			entity local;
                        local = reference_variable(syntax_reference(s));
                        if (entity_integer_scalar_p(local)) ent = local;
                        break;
                }
                default: break;
        }
	debug( 7, "expression_int_scalar",
		 "returning : %s\n", 
		 ((ent == entity_undefined)?"entity_undefined":
			entity_local_name( ent )) );
        return( ent );
}

/*=================================================================*/
/* bool sp_linear_expression_p( (expression) exp)
 * Returns TRUE if the expression is a linear combinaison of 
 * structural parameters.
 */
bool sp_linear_expression_p( exp )
expression exp;
{
   Pvecteur     vect;
   bool         ONLY_SP;

   debug(7, "sp_linear_expression_p", "exp : %s\n",
        	words_to_string(words_expression(exp)));

   if(normalized_tag(NORMALIZE_EXPRESSION(exp)) == is_normalized_complex)
        ONLY_SP = FALSE;
   else
   {
        vect = (Pvecteur) normalized_linear(expression_normalized(exp));
        ONLY_SP = TRUE;

        for(; !VECTEUR_NUL_P(vect) && ONLY_SP ; vect = vect->succ)
        {
                entity var = (entity) vect->var;

                if( ! term_cst(vect) )
                if( ! (ENTITY_SP_P(var)) )
                        ONLY_SP = FALSE;
        }
   }
   unnormalize_expression(exp);
   debug(7, "sp_linear_expression_p",
		 "  result : %s\n", (ONLY_SP?"TRUE":"FALSE") );
   return(ONLY_SP);
}

/*=================================================================*/
/* bool splc_feautrier_expression_p( (expression) exp )
 * Returns TRUE if exp quasi affine form in structural parameters
 * and in surrounding loop-counters.
 */
bool splc_feautrier_expression_p(exp, ell)
expression exp;
list *ell;
{
	bool b = FALSE;
	syntax s = expression_syntax( exp );

	debug( 7, "splc_feautrier_expression_p", "exp : %s \n",
		((exp == expression_undefined)?"expression_undefined":
			words_to_string( words_expression( exp ) ) ));

	if (splc_linear_expression_p(exp, ell)) return( TRUE );
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
	debug(7, "splc_feautrier_expression_p", "returning : %s\n",
			(b?"TRUE":"FALSE") );
	return( b );	
}

/*=================================================================*/
/* bool sp_feautrier_expression_p( (expression) exp)
 * Returns TRUE if exp quasi affine form.
 */
bool sp_feautrier_expression_p( exp )
expression exp;
{
	bool b = FALSE;
	syntax s = expression_syntax( exp );

	debug( 7, "sp_feautrier_expression_p", "exp : %s \n",
		((exp == expression_undefined)?"expression_undefined":
			words_to_string( words_expression( exp ) ) ));

	if (sp_linear_expression_p( exp )) return( TRUE );
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
	debug(7, "sp_feautrier_expression_p", "returning : %s\n",
			(b?"TRUE":"FALSE") );
	return( b );	
}

/*=================================================================*/
/* entity  sp_feautrier_scalar_assign_call( (call) c )
 * Returns the left-hand-side entity if it is an assignement of 
 * a linear combinaison of structural parameters.
 */
entity sp_feautrier_scalar_assign_call( c )
call c;
{
	entity 		ent, ret_ent = entity_undefined;
	expression 	rhs;

	debug(7, "sp_feautrier_scalar_assign_call", "doing\n");
	if ((ent = scalar_assign_call(c)) != entity_undefined) {
		rhs = EXPRESSION(CAR(CDR(call_arguments(c))));
		if (sp_feautrier_expression_p( rhs )) ret_ent = ent;
	}
	debug( 7, "sp_feautrier_scalar_assign_call",
		"returning : %s \n", 
		((ret_ent == entity_undefined)?"entity_undefined":
		entity_name( ret_ent )) ); 
	return( ret_ent );
}
		

/*=================================================================*/
/* bool get_sp_of_call_p( (call) c, fst) AL 04/93 Updates the global
 * variables Gstructure_parameters and Gforward_substitute_table according
 * to the type of call.  Returns TRUE if the call has to be modified
 * (redefinition of a structural parameter), FALSE in all the other cases.
 *
 * AP, sep 95 : Gforward_substitute_table is no longer a global variable,
 * we pass it as an argument.
 */
bool get_sp_of_call_p( c, fst, swfl)
call c;
hash_table fst; /* forward substitute table */
list *swfl;
{
   entity 	lhs_ent, ent;
   bool		ret_bool = FALSE;

   debug( 7, "get_sp_of_call", "begin\n");
   debug(9, "get_sp_of_call", "input call : %s \n",
			words_to_string(words_regular_call( c )));
   debug(9, "get_sp_of_call", "struct param. before : %s \n",
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
		ret_bool = TRUE;
	}
   }

   debug(9, "get_sp_of_call", "struct param. after  : %s \n",
			print_structurals( Gstructure_parameters ));
   debug(9, "get_sp_of_call", "call has to be modified : %s \n",
		((ret_bool == TRUE)?"TRUE":"FALSE") );
   debug( 7, "get_sp_of_call", "end\n");
   return( ret_bool );
}

/*=================================================================*/
/* bool normalizable_loop_p(loop l)
 * Returns TRUE if "l" has a constant step.
 */
bool normalizable_loop_p(l)
loop l;
{
debug( 7, "normalizable_loop_p", "doing\n");
return(expression_constant_p(range_increment(loop_range(l))));
}
 

/*=================================================================*/
/* bool normal_loop_p( loop l ) returns TRUE if "l" 's step is egal to 1
 */
bool normal_loop_p( l )
loop l ;
{
	expression ri;
	entity ent;

	debug( 7, "normal_loop_p", "doing\n");
	ri = range_increment(loop_range(l));
	if (!expression_constant_p( ri )) return( FALSE );
	ent = reference_variable(syntax_reference(expression_syntax( ri )));
	return( strcmp(entity_local_name(ent), "1") == 0 );
}


/*=================================================================*/
/* expression make_max_exp(entity ent, expression exp1, expression exp2)
 * computes MAX( exp1, exp2 ) if exp1 and exp2 are constant expressions.
 * If it is not the case, it returns MAX( exp1, exp2 )
 */
expression make_max_exp( ent, exp1, exp2 )
entity 		ent;
expression 	exp1, exp2;
{
	expression rexp;

	debug( 7, "make_max_exp", "doing MAX( %s, %s ) \n",
		words_to_string(words_expression( exp1 )),
		words_to_string(words_expression( exp2 )) );
	if (expression_constant_p( exp1 ) && expression_constant_p( exp2 )) {
		int val1 = expression_to_int( exp1 );
		int val2 = expression_to_int( exp2 );
		if (val1 > val2) rexp = make_integer_constant_expression(val1);
		else rexp = make_integer_constant_expression( val2 );
	}
	else rexp = MakeBinaryCall( ent, exp1, exp2 );

	return rexp ;
}
	

/*=================================================================*/
/* entity make_nlc_entity(int *Gcount_nlc):
 *
 * Returns a new entity. Its local name is "NLC#", where '#' represents
 * the value of "Gcount_nlc". This variable counts the number of NLCs
 * variables.
 *
 * These entities have a special full name. The first part of it is the
 * concatenation of the define constant STATIC_CONTROLIZE_MODULE_NAME and
 * the local name of the current module.
 *
 * The type ("basic") of these variables is INTEGER.
 *
 * These variables are local to the current module, so they have a
 * "storage_ram" with DYNAMIC "area".
 *
 * NLC means Normalized Loop Counter.
 */
entity make_nlc_entity(Gcount_nlc)
int *Gcount_nlc;
{
	entity 	new_ent, mod_ent;
	char 	*name, *num;
	entity  dynamic_area;
	ram	new_dynamic_ram;
	

	debug( 7, "make_nlc_entity", "doing\n");
	(*Gcount_nlc)++;
    num=i2a(*Gcount_nlc);

	mod_ent = get_current_module_entity();

	name = strdup(concatenate(STATIC_CONTROLIZE_MODULE_NAME,
                          entity_local_name(mod_ent),
                          MODULE_SEP_STRING, NLC_PREFIX, num, (char *) NULL));
    free(num);

	new_ent = make_entity(name,
                      make_type(is_type_variable,
                                make_variable(make_basic(is_basic_int, 4),
                                              NIL)),
                      make_storage(is_storage_ram, ram_undefined),
                      make_value(is_value_unknown, UU));

	dynamic_area = FindOrCreateEntity( module_local_name(mod_ent),
                                  DYNAMIC_AREA_LOCAL_NAME);

	new_dynamic_ram = make_ram(mod_ent,
                           dynamic_area,
                           CurrentOffsetOfArea(dynamic_area, new_ent),
                           NIL);

	storage_ram(entity_storage(new_ent)) = new_dynamic_ram;

	return(new_ent);
}

/*=================================================================*/
/* entity  make_nsp_entity()
 * Makes a new NSP (for New Structural Parameter) .
 */
entity make_nsp_entity()
{
	extern  int Gcount_nsp;
	entity  new_ent, mod_ent;
	char    *name, *num;
	entity  dynamic_area;
	ram	new_dynamic_ram;

	debug( 7, "make_nsp_entity", "doing\n");
	Gcount_nsp++;
    num=i2a(Gcount_nsp);

	mod_ent = get_current_module_entity();

	name = strdup(concatenate(STATIC_CONTROLIZE_MODULE_NAME,
                          entity_local_name(mod_ent),
                          MODULE_SEP_STRING, NSP_PREFIX, num, (char *) NULL));
    free(num);

        new_ent = make_entity(name,
                      make_type(is_type_variable,
                                make_variable(make_basic(is_basic_int, 4),
                                              NIL)),
                      make_storage(is_storage_ram, ram_undefined),
                      make_value(is_value_unknown, UU));

        dynamic_area = FindOrCreateEntity( module_local_name(mod_ent),
                                  DYNAMIC_AREA_LOCAL_NAME);

        new_dynamic_ram = make_ram(mod_ent,
                           dynamic_area,
                           CurrentOffsetOfArea(dynamic_area, new_ent),
                           NIL);

        storage_ram(entity_storage(new_ent)) = new_dynamic_ram;

	return new_ent;
}

/*=================================================================*/
/* entity  make_nub_entity()
 * Makes a new NUB (for New Upper Bound) .
 */
entity make_nub_entity()
{
	extern  int Gcount_nub;
	entity  new_ent, mod_ent;
	char    *name, *num;
	entity	dynamic_area;
	ram	new_dynamic_ram;


	debug( 7, "make_nub_entity", "doing\n");
	Gcount_nub++;
    num=i2a(Gcount_nub);

	mod_ent = get_current_module_entity();

	name = strdup(concatenate(STATIC_CONTROLIZE_MODULE_NAME,
                          entity_local_name(mod_ent),
                          MODULE_SEP_STRING, NUB_PREFIX, num, (char *) NULL));

        new_ent = make_entity(name,
                      make_type(is_type_variable,
                                make_variable(make_basic(is_basic_int, 4),
                                              NIL)),
                      make_storage(is_storage_ram, ram_undefined),
                      make_value(is_value_unknown, UU));

        dynamic_area = FindOrCreateEntity( module_local_name(mod_ent),
                                  DYNAMIC_AREA_LOCAL_NAME);

        new_dynamic_ram = make_ram(mod_ent,
                           dynamic_area,
                           CurrentOffsetOfArea(dynamic_area, new_ent),
                           NIL);

        storage_ram(entity_storage(new_ent)) = new_dynamic_ram;

	return new_ent;
}

/*=================================================================*/
/* entity current_module(entity mod): returns the current module entity,
 * that is the entity of the module in which we are working currently.
 * If the entity "mod" is undefined, it returns the static entity already known;
 * Else, the static entity is updated to the entity "mod".
 */
entity current_module(mod)
entity mod;
{
    static entity current_mod;

    debug( 7, "current_module", "doing\n");
    if (mod != entity_undefined) {
	pips_assert("current_module_entity", entity_module_p(mod));
	current_mod = mod;
    }
    return(current_mod);
}

/*=================================================================*/
