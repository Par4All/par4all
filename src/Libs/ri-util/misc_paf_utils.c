/*
 * $Id$
 */
#include <stdio.h>

#include "linear.h"
#include "genC.h"
#include "text.h"
#include "ri.h"

#include "misc.h"
#include "text-util.h"
#include "ri-util.h"

#define STATIC_CONTROLIZE_MODULE_NAME 	"STATCON"
#define NLC_PREFIX 			"NLC"
#define NSP_PREFIX 			"NSP"
#define NUB_PREFIX			"NUB"

/* list base_to_list(Pbase v): translates a Pbase into a list of entities, in
 * the same order.
 */
list base_to_list(Pbase b)
{
 list l = NIL;

 for( ; b != NULL; b = b->succ)
   l = gen_nconc(l, CONS(ENTITY, (entity) b->var, NIL));

 return(l);
}

/* Pbase list_to_base(list l): returns the Pbase that contains the variables
 * of list "l", of entities, in the same order. 
 */
Pbase list_to_base(l)
list l;
{
 Pbase new_b = NULL;
 list el_l;

 for(el_l = l ; el_l != NIL; el_l = CDR(el_l))
   vect_add_elem((Pvecteur *) &new_b, (char *) ENTITY(CAR(el_l)), VALUE_ONE);

 new_b = base_reversal(new_b);
 return(new_b);
}

/* void fprint_entity_list(FILE *fp,list l): prints a list of entities on
 * file fp.
 */
void fprint_entity_list(FILE *fp, list l)
{
  for( ; l != NIL; l = CDR(l))
    fprintf(fp, "%s, ", entity_local_name(ENTITY(CAR(l))));
}

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
	num = (char*) malloc(32);
	(void) sprintf(num, "%d", *Gcount_nlc);

	mod_ent = get_current_module_entity();

	name = strdup(concatenate(STATIC_CONTROLIZE_MODULE_NAME,
                          entity_local_name(mod_ent),
                          MODULE_SEP_STRING, NLC_PREFIX, num, (char *) NULL));

	new_ent = make_entity(name,
                      make_type(is_type_variable,
                          make_variable(make_basic(is_basic_int, UUINT(4)),
                                              NIL,NIL)),
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

int Gcount_nsp;

int Gcount_nub;

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
	num = (char*) malloc(32);
	(void) sprintf(num, "%d", Gcount_nsp);

	mod_ent = get_current_module_entity();

	name = strdup(concatenate(STATIC_CONTROLIZE_MODULE_NAME,
                          entity_local_name(mod_ent),
                          MODULE_SEP_STRING, NSP_PREFIX, num, (char *) NULL));

        new_ent = make_entity(name,
                      make_type(is_type_variable,
                            make_variable(make_basic(is_basic_int, UUINT(4)),
                                              NIL,NIL)),
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
	num = (char*) malloc(32);
	(void) sprintf(num, "%d", Gcount_nub);

	mod_ent = get_current_module_entity();

	name = strdup(concatenate(STATIC_CONTROLIZE_MODULE_NAME,
                          entity_local_name(mod_ent),
                          MODULE_SEP_STRING, NUB_PREFIX, num, (char *) NULL));

        new_ent = make_entity(name,
                      make_type(is_type_variable,
                        make_variable(make_basic(is_basic_int, UUINT(4)),
                                              NIL,NIL)),
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

#define ADD_ELEMENT_TO_LIST( _list, _type, _element) \
    (_list = gen_nconc( _list, CONS( _type, _element, NIL)))

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
		forward_substitute_in_exp((expression*)(REFCAR(*pl)), fst);
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

/* void fprint_list_of_exp(FILE *fp, list exp_l): prints in the file "fp"
 * the list of expression "exp_l". We separate the expressions with a colon
 * (","). We do not end the print with a line feed.
 */
void fprint_list_of_exp(fp, exp_l)
FILE *fp;
list exp_l;
{
 list aux_l;
 expression exp;

 for(aux_l = exp_l; aux_l != NIL; aux_l = CDR(aux_l))
   {
    exp = EXPRESSION(CAR(aux_l));
    fprintf(fp,"%s", words_to_string(words_expression(exp)));
    if(CDR(aux_l) != NIL)
       fprintf(fp,",");
   }
}

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
