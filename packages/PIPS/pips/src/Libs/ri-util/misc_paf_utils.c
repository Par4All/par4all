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

// To have asprintf:
#include <stdio.h>

#include "linear.h"
#include "genC.h"
#include "text.h"
#include "ri.h"

#include "misc.h"
#include "text-util.h"
#include "ri-util.h"
#include "syntax.h"

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
	       words_to_string(words_expression(exp1, NIL)),
	       words_to_string(words_expression(exp2, NIL)) );
	if (expression_constant_p( exp1 ) && expression_constant_p( exp2 )) {
		int val1 = expression_to_int( exp1 );
		int val2 = expression_to_int( exp2 );
		if (val1 > val2) rexp = int_to_expression(val1);
		else rexp = int_to_expression( val2 );
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
	(void) asprintf(&num, "%d", *Gcount_nlc);

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
	entity  new_ent, mod_ent;
	char    *name, *num;
	entity  dynamic_area;
	ram	new_dynamic_ram;

	debug( 7, "make_nsp_entity", "doing\n");
	Gcount_nsp++;
	(void) asprintf(&num, "%d", Gcount_nsp);

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
	entity  new_ent, mod_ent;
	char    *name, *num;
	entity	dynamic_area;
	ram	new_dynamic_ram;


	debug( 7, "make_nub_entity", "doing\n");
	Gcount_nub++;
	(void) asprintf(&num, "%d", Gcount_nub);

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
    fprintf(fp,"%s", words_to_string(words_expression(exp, NIL)));
    if(CDR(aux_l) != NIL)
       fprintf(fp,",");
   }
}

/* bool undefined_statement_list_p( (list) l )			AL 04/93
 * Returns true if l is made of 2 undefined or continue statement.
 */
bool undefined_statement_list_p( l )
list l;
{
	bool 		local_bool;
	statement 	first, second;

	debug(7, "undefined_statement_list_p","doing\n");
	if ( (l == NIL) || (gen_length(l) != 2) )
		return( false );

	first = STATEMENT(CAR( l ));
	second = STATEMENT(CAR(CDR( l )));
	local_bool = ( first == statement_undefined )
		     && ( second == statement_undefined );

	/* Newgen does not support list of undefined objects */
	if(!local_bool) {
	  local_bool = continue_statement_p(first) && continue_statement_p(second);
	}

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

        lhs = binary_call_lhs(c);
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
