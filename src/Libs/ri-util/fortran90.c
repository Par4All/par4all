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

expression update_range( f, r, lw, up, in ) 
entity f ;
range r ;
expression lw, up, in ;
{
    range new_r = make_range(range_lower(r),
			     range_upper(r),
			     range_increment(r)) ;
    range_lower( new_r ) = MakeBinaryCall(f, lw, range_lower(r)) ;
    range_upper( new_r ) = MakeBinaryCall(f, up, range_upper(r)) ;
	
    if( strcmp( entity_local_name( f ), MULTIPLY_OPERATOR_NAME ) == 0 ) {
	range_increment( new_r ) = 
		MakeBinaryCall(f, in, range_increment(r)) ;
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
	     strcmp( entity_local_name( f ), DIVIDE_OPERATOR_NAME ) == 0 ||
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
				 range_increment(rr)) ;
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

	    new_e = update_range(f, rl, rhs, rhs, make_expression_1()) ;
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

	    new_e = update_range(f, rr, lhs, lhs, make_expression_1()) ;
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
	bool vectorp = FALSE ;
	cons *new_args = NIL ;
	reference new_r ;

	MAPL( args, {
	    expression arg = EXPRESSION( CAR( args )) ;

	    new_e = expand_expression( arg, i, r ) ;
	    vectorp |= set_belong_p( vectors, (char *)new_e ) ;
	    new_args = CONS(EXPRESSION, new_e, new_args ) ;
	}, reference_indices( rf )) ;

	if( vectorp ) {
	    new_r = make_reference(reference_variable(rf),
				   gen_nreverse(new_args)) ;
	    new_e = make_expression(make_syntax(is_syntax_reference,new_r),
				    normalized_undefined) ;
	    set_add_element( vectors, vectors, (char *)new_e ) ;
	    return( new_e ) ;
	}
	else {
	    gen_free_list( new_args ) ;
	    return( e ) ;
	}
    }
}

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
	bool vectorp = FALSE ;
	
	MAPL( args, {
	    expression arg = EXPRESSION( CAR( args )) ;
	    
	    new_e = expand_expression( arg, i, r ) ;
	    vectorp |= set_belong_p( vectors, (char *)new_e ) ;
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
    instruction i = statement_instruction( loop_body( obj )) ;
    entity idx = loop_index( obj ) ;
    range r = loop_range( obj ) ;
    expression lhs =
	    EXPRESSION( CAR( call_arguments( instruction_call( i )))) ;
    expression rhs = 
	    EXPRESSION( CAR( CDR( call_arguments( instruction_call( i ))))) ;
    
    vectors = set_make( set_pointer ) ;
{
    expression new_lhs = expand_expression( lhs, idx, r ) ;
    expression new_rhs = expand_expression( rhs, idx, r ) ;
    
    set_free( vectors ) ;
    return( text_statement(module, margin, 
			   make_assign_statement( new_lhs, new_rhs ))) ;
}
}

