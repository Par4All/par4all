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
/* Prettyprint one FORTRAN 90 loop as an array expression.

   Pierre Jouvelot

   For one level only loop, with one assignment as body. Replaces
   occurences of the index variable by ranges in expressions. Ranges are
   prettyprinted as triplet when they occur as subscript expressions and
   as vectors with implicit DO otherwise. If the replacement cannot occur,
   for instance because subscript expressions are coupled, the loop is
   printed as a loop.

   There are/were memory leaks here since a new expression is constructed.

*/

#include <stdio.h>

#include "linear.h"

#include "genC.h"
#include "text.h"
#include "text-util.h"
#include "ri.h"
#include "misc.h"

#include "ri-util.h"

/* FI: To keep track of vectorized subexpressions (a guess...) */
static set vectors;

/* Entity f is supposed to be a binary operator, not always commutative
   but meaningful for a range and a scalar or for two ranges. Boolean
   argument left means that range r appears on the left of f, i.e. as
   first operator.

   No sharing is created between the new expression e and arguments r, lw,
   up or in.
  */

expression update_range(f, r, lw, up, in, left)
entity f ;
range r ;
expression lw, up, in ;
bool left;
{
    range new_r = copy_range(r);
    intptr_t val = 0;
    expression new_up = copy_expression(up);
    expression new_lw = copy_expression(lw);
    expression new_in = copy_expression(in);
    expression e = expression_undefined;

    pips_assert("new_r is consistent before update", range_consistent_p(new_r));

    if(left) {
	range_lower( new_r ) = MakeBinaryCall(f, range_lower(new_r), new_lw);
	range_upper( new_r ) = MakeBinaryCall(f, range_upper(new_r), new_up);
    }
    else {
	range_lower( new_r ) = MakeBinaryCall(f, new_lw, range_lower(new_r));
	range_upper( new_r ) = MakeBinaryCall(f, new_up, range_upper(new_r));
    }

    pips_assert("new_r is consistent", range_consistent_p(new_r));
	
    if( strcmp( entity_local_name( f ), MULTIPLY_OPERATOR_NAME ) == 0 ) {
	/* expression "in" must be integer constant 1 */
	pips_assert("This is a scalar value", lw==up);
	free_expression(range_increment( new_r ));
	range_increment( new_r ) = 
		MakeBinaryCall(f, new_in, copy_expression(range_lower(new_r))) ;
    }
	
    if( !left && (strcmp( entity_local_name( f ), MINUS_OPERATOR_NAME ) == 0) ) {
	entity um = entity_intrinsic(UNARY_MINUS_OPERATOR_NAME);

	range_increment( new_r ) = 
		MakeUnaryCall(um, range_increment(new_r)) ;
    }

    if(expression_integer_value(range_lower(new_r), &val)) {
	free_expression(range_lower(new_r));
	range_lower(new_r) = int_to_expression(val);
    }

    if(expression_integer_value(range_upper(new_r), &val)) {
	free_expression(range_upper(new_r));
	range_upper(new_r) = int_to_expression(val);
    }

    if(expression_integer_value(range_increment(new_r), &val)) {
	free_expression(range_increment(new_r));
	range_increment(new_r) = int_to_expression(val);
    }

    pips_assert("new_r is consistent after simplification", range_consistent_p(new_r));

    e = make_expression(make_syntax(is_syntax_range, new_r),
			normalized_undefined);

    return e;
}

/* Only one call site for expand_call(). All args have been newly
   allocated there and are re-used here to build new_e, Or they are freed
   without forgetting the references thru set "vectors". There is no
   sharing between e and new_e. */
	
static
expression expand_call( e, f, args )
expression e ;
entity f ;
list args ;
{
    expression new_e = expression_undefined;
    bool vector_op = 
	(strcmp( entity_local_name( f ), PLUS_OPERATOR_NAME ) == 0 ||
	 strcmp( entity_local_name( f ), MINUS_OPERATOR_NAME ) == 0 ||
	 /* strcmp( entity_local_name( f ), DIVIDE_OPERATOR_NAME ) == 0 || */
	 strcmp( entity_local_name( f ), MULTIPLY_OPERATOR_NAME ) == 0 ) ;

    if( !vector_op ) {
	/* FI: Sharing thru args? Yes, but see above. */
	bool vectorp = false;
	new_e = make_expression(make_syntax(is_syntax_call,
					    make_call( f, args )),
				normalized_undefined);
	MAP(EXPRESSION, arg, {
	    vectorp |= set_belong_p( vectors, (char *)arg );
	}, args);

	if(vectorp)
	    set_add_element( vectors, vectors, (char *)new_e ) ;
    }
    else {
	expression lhs = EXPRESSION(CAR(args)) ;
	expression rhs = EXPRESSION(CAR(CDR(args))) ;
	syntax ls = expression_syntax( lhs ) ;
	syntax rs = expression_syntax( rhs ) ;

	if(set_belong_p( vectors, (char *)lhs ) &&
	   set_belong_p( vectors, (char *)rhs )) {
	    if( syntax_range_p( ls ) && syntax_range_p( rs )) {
		range rl = syntax_range( ls ) ;
		range rr = syntax_range( rs ) ;

		new_e = update_range(f, rl, 
				     range_lower(rr), range_upper(rr),
				     range_increment(rr), true) ;
	    }
	    else {
		new_e = MakeBinaryCall( f, lhs, rhs ) ;
	    }
	    set_add_element( vectors, vectors, (char *)new_e ) ;
	}
	else if( set_belong_p( vectors, (char *)lhs )) {
	    if( syntax_range_p( ls )) {
		range rl = syntax_range( ls ) ;

		new_e = update_range(f, rl, rhs, rhs, int_to_expression(1), true) ;
	    }
	    else {
		new_e = MakeBinaryCall( f, lhs, rhs ) ;
	    }
	    set_add_element( vectors, vectors, (char *)new_e ) ;
	}
	else if( set_belong_p( vectors, (char *)rhs )) {
	    if( syntax_range_p( rs )) {
		range rr = syntax_range( rs ) ;

		new_e = update_range(f, rr, lhs, lhs, int_to_expression(1), false) ;
	    }
	    else {
		new_e = MakeBinaryCall( f, lhs, rhs ) ;
	    }
	    set_add_element(vectors, vectors, (char *) new_e);
	}
	else {
	    /* No sharing between e and new_e */
	    new_e = copy_expression(e);
	}
    }
    return new_e;
}

/* A reference cannot always be expanded. Subscript expression coupling as
   in A(I,I) prevent expansion and an undefined expression is returned. Non-affine expressions such as A(I**2)
   cannot be transformed into triplets but can be tranformed into implicit
   DO vectors.

   Arguments s, e, i and r should not be shared with the returned expression.  */

expression expand_reference( s, e, i, r )
syntax s ;
expression e ;
entity i ;
range r ;
{
    reference rf = syntax_reference(s) ;
    expression new_e = expression_undefined;

    if( same_entity_p( reference_variable( rf ), i )) {
	/* expand occurence of loop index */
	syntax new_s = make_syntax(is_syntax_range,copy_range(r)) ;
	new_e = make_expression(new_s, normalized_undefined) ;
	set_add_element( vectors, vectors, (char *) new_e ) ;
    }
    else {
	/* expand 1 subscript expression or fail or leave unexpanded */
	int dim = 0 ;
	cons *new_args = NIL ;
	reference new_r ;

	MAPL( args, {
	    expression arg = EXPRESSION( CAR( args )) ;

	    new_e = expand_expression( arg, i, r ) ;

	    if(expression_undefined_p(new_e))
		return new_e;

	    if(set_belong_p( vectors, (char *)new_e ))
		dim++;

	    new_args = CONS(EXPRESSION, new_e, new_args ) ;
	}, reference_indices( rf )) ;

	if( dim==1 ) {
	    new_r = make_reference(reference_variable(rf),
				   gen_nreverse(new_args)) ;
	    new_e = make_expression(make_syntax(is_syntax_reference,new_r),
				    normalized_undefined) ;
	    set_add_element( vectors, vectors, (char *)new_e ) ;
	}
	else if(dim > 1) {
	    /* If dim is greater than 1, subscript expressions are coupled
	     * as in A(I,I+1).
	     */
	    /* new_args should be freed */
	    new_e = expression_undefined;
	}
	else {
	    /* Just the spine or more? */
	    gen_free_list(new_args);
	    new_e = copy_expression(e);
	}
    }
    return new_e;
}


/* Expression with a non-expandable sub-expression, e.g. a non-expandable
   reference, cannot be expanded.

   Arguments are not (should not be) shared with the returned expression.
 */

expression expand_expression( e, i, r )
expression e ;
entity i ;
range r ;
{
    syntax s = expression_syntax( e ) ;
    expression new_e = expression_undefined;

    switch(syntax_tag( s )) 
    {
    case is_syntax_reference:
	new_e = expand_reference( s, e, i, r);
	break;
    case is_syntax_call: {
	call c = syntax_call( s ) ;
	cons *new_args = NIL ;
	entity f = call_function( c ) ;
	int dim = 0 ;
	
	MAPL( args, {
	    expression arg = EXPRESSION( CAR( args )) ;
	    
	    new_e = expand_expression( arg, i, r ) ;

	    if(expression_undefined_p(new_e)) {
		return new_e;
	    }

	    /* FI: I do not know why dim is computed. Old cut-and-paste
               from expand_reference()? */
	    if(set_belong_p( vectors, (char *)new_e )) {
		dim++;
	    }

	    new_args = CONS(EXPRESSION, new_e, new_args ) ;
	}, call_arguments( c )) ;

	new_e = expand_call(e, f, gen_nreverse(new_args));
	break;
    }
    case is_syntax_range:
	pips_internal_error("Range expansion not implemented" ) ;
    default:
	pips_internal_error("unexpected syntax tag (%d)",
		   syntax_tag(s));
    }

    return new_e;
}

/* The tests necessary to check the underlying assumptions have been
   performed in text_loop(): b is either an assignment or a sequence
   with only one statement which is an assignment. */
static statement body_to_assignment_statement(statement b)
{
  instruction bi = statement_instruction(b);
  statement s = statement_undefined;

  if(instruction_sequence_p(bi)) {
    list sl = sequence_statements(instruction_sequence(bi));
    s = STATEMENT(CAR(sl));
  }
  else /* because of the input assumptions */
    s = b;

  return s;
}

/* Generate range subscript for simple loop with only one assignment. */
text text_loop_90(entity module, const char* label, int margin, loop obj, int n)
{
    /* text_loop_90() only is called if the loop is parallel and if its
     * body is a unique assignment statement or a list containing a
     * unique assignment.
     */
  statement as = body_to_assignment_statement( loop_body( obj )) ;
  instruction i = statement_instruction(as);
    entity idx = loop_index( obj ) ;
    range r = loop_range( obj ) ;

    expression lhs =
	EXPRESSION( CAR( call_arguments( instruction_call( i )))) ;
    expression rhs =
	EXPRESSION( CAR( CDR( call_arguments( instruction_call( i ))))) ;
    expression new_lhs = expression_undefined;
    expression new_rhs = expression_undefined;
    text t = text_undefined;

    pips_assert("Loop obj is consistent", loop_consistent_p(obj));

    vectors = set_make( set_pointer ) ;

    new_lhs = expand_expression( lhs, idx, r ) ;
    new_rhs = expand_expression( rhs, idx, r ) ;

    pips_assert("new_lhs is consistent", expression_consistent_p(new_lhs));
    pips_assert("new_rhs is consistent", expression_consistent_p(new_rhs));

    set_free(vectors);

    if(!expression_undefined_p(new_lhs) && !expression_undefined_p(new_rhs)) {
	statement new_s = make_assign_statement( new_lhs, new_rhs );

	statement_number(new_s) = statement_number(as);
	/* statement_ordering must be initialized too to avoid a
           prettyprinter warning */
	statement_ordering(new_s) = statement_ordering(as);
	statement_comments(new_s) = statement_comments(as);
	t = text_statement(module, margin, new_s, NIL);
	/* FI: Although new_s has been converted to text, it cannot
	   always be freed. I do not know which part of new_s is
	   reused in the result of text_statement() or somewhere
	   else... Found with valgrind and validation case
	   Prettyprint/aa01.tpips */
	//free_statement(new_s);
    }
    else {
	/* No legal vector form has been found */
	free_expression(new_lhs);
	free_expression(new_rhs);
	t = text_loop_default(module, label, margin, obj, n, NIL);
    }

    pips_assert("Loop obj still is consistent", loop_consistent_p(obj));

    return t;
}
