/* Prettyprinter for FORTRAN 90 loops.

   There are memory leaks here since a new expression is constructed.

*/

#include <stdio.h>

#include "linear.h"

#include "genC.h"
#include "text.h"
#include "text-util.h"
#include "ri.h"
#include "misc.h"

#include "ri-util.h"

static set vectors ;

/* f is supposed to be a binary operator, not always commutative. left
   means that range r appears on the left of f, i.e. as first operator. */

expression update_range(f, r, lw, up, in, left)
entity f ;
range r ;
expression lw, up, in ;
bool left;
{
    range new_r = make_range(range_lower(r),
			     range_upper(r),
			     range_increment(r)) ;

    if(left) {
	range_lower( new_r ) = MakeBinaryCall(f, range_lower(r), lw);
	range_upper( new_r ) = MakeBinaryCall(f, range_upper(r), up);
    }
    else {
	range_lower( new_r ) = MakeBinaryCall(f, lw, range_lower(r));
	range_upper( new_r ) = MakeBinaryCall(f, up, range_upper(r));
    }
	
    if( strcmp( entity_local_name( f ), MULTIPLY_OPERATOR_NAME ) == 0 ) {
	/* expression "in" must be integer constant 1 */
	pips_assert("This is a scalar value", lw==up);
	range_increment( new_r ) = 
		MakeBinaryCall(f, in, range_lower(r)) ;
    }
	
    if( !left && (strcmp( entity_local_name( f ), MINUS_OPERATOR_NAME ) == 0) ) {
	entity um = entity_intrinsic(UNARY_MINUS_OPERATOR_NAME);

	range_increment( new_r ) = 
		MakeUnaryCall(um, range_increment(r)) ;
    }

    return( make_expression(make_syntax(is_syntax_range,new_r),
			    normalized_undefined) ) ;
}
	
expression expand_call( e, f, args )
expression e ;
entity f ;
list args ;
{
    expression new_e ;
    bool vector_op = 
	(strcmp( entity_local_name( f ), PLUS_OPERATOR_NAME ) == 0 ||
	 strcmp( entity_local_name( f ), MINUS_OPERATOR_NAME ) == 0 ||
	 /* strcmp( entity_local_name( f ), DIVIDE_OPERATOR_NAME ) == 0 || */
	 strcmp( entity_local_name( f ), MULTIPLY_OPERATOR_NAME ) == 0 ) ;

    if( !vector_op ) {
	return( make_expression(make_syntax(is_syntax_call,
					    make_call( f, args )),
				normalized_undefined)) ;
    }
    {
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
				     range_increment(rr), TRUE) ;
	    }
	    else {
		new_e = MakeBinaryCall( f, lhs, rhs ) ;
	    }
	    set_add_element( vectors, vectors, (char *)new_e ) ;
	    return( new_e ) ;
	}
	else if( set_belong_p( vectors, (char *)lhs )) {
	    if( syntax_range_p( ls )) {
		range rl = syntax_range( ls ) ;

		new_e = update_range(f, rl, rhs, rhs, make_expression_1(), TRUE) ;
	    }
	    else {
		new_e = MakeBinaryCall( f, lhs, rhs ) ;
	    }
	    set_add_element( vectors, vectors, (char *)new_e ) ;
	    return( new_e ) ;
	}
	else if( set_belong_p( vectors, (char *)rhs )) {
	    if( syntax_range_p( rs )) {
		range rr = syntax_range( rs ) ;

		new_e = update_range(f, rr, lhs, lhs, make_expression_1(), FALSE) ;
	    }
	    else {
		new_e = MakeBinaryCall( f, lhs, rhs ) ;
	    }
	    set_add_element( vectors, vectors, (char *)new_e ) ;
	    return( new_e ) ;
	}
	return( e ) ;
    }
}

/* A reference cannot always be expanded. Subscript expression coupling as
   in A(I,I) prevent coupling. Non-affine expressions such as A(I**2)
   cannot be transformed into ranges. */

expression expand_reference( s, e, i, r )
syntax s ;
expression e ;
entity i ;
range r ;
{
    reference rf = syntax_reference( s ) ;
    expression new_e;
    syntax new_s ;

    if( same_entity_p( reference_variable( rf ), i )) {
	new_s = make_syntax(is_syntax_range,r) ;
	new_e = make_expression(new_s, normalized_undefined) ;
	set_add_element( vectors, vectors, (char *) new_e ) ;
	return( new_e ) ;
    }
    else {
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
	    return( new_e ) ;
	}
	else if(dim > 1) {
	    /* If dim is greater than 1, subscript expressions are coupled
	     * as in A(I,I+1).
	     */
	    return expression_undefined;
	}
	else {
	    gen_free_list( new_args ) ;
	    return( e ) ;
	}
    }
}


/* Expression with a non-expandable sub-expression, e.g. a non-expandable
   reference, cannot be expanded */

expression expand_expression( e, i, r )
expression e ;
entity i ;
range r ;
{
    syntax s = expression_syntax( e ) ;

    switch(syntax_tag( s )) 
    {
    case is_syntax_reference:
	return( expand_reference( s, e, i, r )) ;
    case is_syntax_call: {
	call c = syntax_call( s ) ;
	expression new_e ;
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
	
	return( expand_call(e,f,gen_nreverse( new_args ))) ;
    }
    case is_syntax_range:
	pips_error("expand_expression", 
		   "Range expansion not implemented\n" ) ;
    default:
	pips_error("expand_expression", "unexpected syntax tag (%d)\n",
		   syntax_tag(s));
    }

    return(expression_undefined); /* just to avoid a gcc warning */
}

text text_loop_90(module, label, margin, obj, n)
entity module;
string label;
int margin;
loop obj;
int n ;
{
    /* text_loop_90() only is called if the loop is parallel and if its
     * body is a unique assignment statement.
     */
    instruction i = statement_instruction( loop_body( obj )) ;
    entity idx = loop_index( obj ) ;
    range r = loop_range( obj ) ;
    expression lhs =
	EXPRESSION( CAR( call_arguments( instruction_call( i )))) ;
    expression rhs = 
	EXPRESSION( CAR( CDR( call_arguments( instruction_call( i ))))) ;

    expression new_lhs = expression_undefined;
    expression new_rhs = expression_undefined;
    
    vectors = set_make( set_pointer ) ;

    new_lhs = expand_expression( lhs, idx, r ) ;
    new_rhs = expand_expression( rhs, idx, r ) ;
    
    set_free( vectors ) ;

    if(!expression_undefined_p(new_lhs) && !expression_undefined_p(new_rhs)) {
	statement new_s = make_assign_statement( new_lhs, new_rhs );

	/*
	statement_number(new_s) = statement_number(loop_body(obj));
	statement_comments(new_s) = statement_comments(loop_body(obj));
	*/
	return( text_statement(module, margin, new_s)) ;
    }
    else {
	return( text_loop(module, label, margin, obj, n));
    }

}
