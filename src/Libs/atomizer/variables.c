/* -- variables.c
 *
 * package atomizer :  Alexis Platonoff, juillet 91
 * --
 *
 * This file contains functions that creates the new temporary entities and
 * the new auxiliary entities, and functions that compute the "basic" of an
 * expression.
 */

#include <stdio.h>
extern int fprintf();
#include <string.h>

#include "genC.h"
#include "ri.h"
#include "graph.h"
#include "dg.h"

#include "ri-util.h"
/* #include "constants.h" */
#include "misc.h"
#include "list.h"

#include "loop_normalize.h"
#include "atomizer.h"

/* These globals variables count the number of temporary and auxiliary
 * entities. Each time such a variable is created, the corresponding
 * counter is incremented.
 */
int count_tmp = 0;
int count_aux = 0;


/*============================================================================*/
/* entity make_new_entity(basic ba, int kind): Returns a new entity.
 * This entity is either a new temporary or a new auxiliary variable.
 * The parameter "kind" gives the kind of entity to produce.
 * "ba" gives the basic (ie the type) of the entity to create.
 *
 * The number of the temporaries is given by a global variable named
 * "count_tmp".
 * The number of the auxiliary variables is given by a global variable named
 * "count_aux".
 *
 * Called functions:
 *       _ current_module() : loop_normalize/utils.c
 *       _ FindOrCreateEntity() : syntax/declaration.c
 *       _ CurrentOffsetOfArea() : syntax/declaration.c
 */
entity make_new_entity(ba, kind)
basic ba;
int kind;
{
extern int count_tmp, count_aux;
extern list integer_entities, real_entities, logical_entities, complex_entities,
            double_entities, char_entities;

entity new_ent, mod_ent;
char prefix[4], *name, *num;
int number;

/* The first letter of the local name depends on the basic:
 *       int --> I
 *     real  --> F (float single precision)
 *    others --> O
 */
switch(basic_tag(ba))
  {
  case is_basic_int: { (void) sprintf(prefix, "I"); break;}
  case is_basic_float:
    {
    if(basic_float(ba) == DOUBLE_PRECISION_SIZE)
      (void) sprintf(prefix, "O");
    else
      (void) sprintf(prefix, "F");
    break;
    }
  default: (void) sprintf(prefix, "O");
  }

/* The three following letters are whether "TMP", for temporaries
 * or "AUX" for auxiliary variables.
 */
switch(kind)
  {
  case TMP_ENT:
    {
    number = (++count_tmp);
    (void) sprintf(prefix+1, "TMP");
    break;
    }
  case AUX_ENT:
    {
    number = (++count_aux);
    (void) sprintf(prefix+1, "AUX");
    break;
    }
  default: user_error("make_new_entity", "Bad kind of entity: %d", kind);
  }

mod_ent = current_module(entity_undefined);
num = malloc(32);
(void) sprintf(num, "%d", number);

/* The first part of the full name is the concatenation of the define
 * constant ATOMIZER_MODULE_NAME and the local name of the module
 * entity.
 */
name = strdup(concatenate(ATOMIZER_MODULE_NAME, entity_local_name(mod_ent),
                          MODULE_SEP_STRING, prefix, num, (char *) NULL));

new_ent = make_entity(name,
		      make_type(is_type_variable,
			        make_variable(ba,
					      NIL)),
                      make_storage(is_storage_rom, UU),
		      make_value(is_value_unknown, UU));
/*
new_ent = make_entity(name,
		      make_type(is_type_variable,
			        make_variable(ba,
					      NIL)),
                      make_storage(is_storage_ram, ram_undefined),
		      make_value(is_value_unknown, UU));
*/
/* The storage is made to DYNAMIC, ie the variable is local. */
/*
dynamic_area = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME,
                                  DYNAMIC_AREA_LOCAL_NAME);
new_dynamic_ram = make_ram(mod_ent, 
                           dynamic_area,
                           CurrentOffsetOfArea(dynamic_area, new_ent),
                           NIL);
storage_ram(entity_storage(new_ent)) = new_dynamic_ram;
*/

/* The new entity is stored in the list of entities of the same type. */
switch(basic_tag(ba))
  {
  case is_basic_int:
    {
    integer_entities = CONS(ENTITY, new_ent, integer_entities);
    break;
    }
  case is_basic_float:
    {
    if(basic_float(ba) == DOUBLE_PRECISION_SIZE)
      double_entities = CONS(ENTITY, new_ent, double_entities);
    else
      real_entities = CONS(ENTITY, new_ent, real_entities);
    break;
    }
  case is_basic_logical:
    {
    logical_entities = CONS(ENTITY, new_ent, logical_entities);
    break;
    }
  case is_basic_complex:
    {
    complex_entities = CONS(ENTITY, new_ent, complex_entities);
    break;
    }
  case is_basic_string:
    {
    char_entities = CONS(ENTITY, new_ent, char_entities);
    break;
    }
  default:break;
  }

return new_ent;
}



/*============================================================================*/
/* basic basic_of_expression(expression exp): Makes a basic of the same
 * basic as the expression "exp". Indeed, "exp" will be assigned to
 * a temporary variable, which will have the same declaration as "exp".
 */
basic basic_of_expression(exp)
expression exp;
{
syntax sy = expression_syntax(exp);

debug(6, "basic_of_expression", "\n");

if(nlc_linear_expression_p(exp))
  return(make_basic(is_basic_int, 4));

switch(syntax_tag(sy))
  {
  case is_syntax_reference :
    {
    type exp_type = entity_type(reference_variable(syntax_reference(sy)));

    if (type_tag(exp_type) != is_type_variable)
      pips_error("make_tmp_basic", "Bad reference type tag");
    return(variable_basic(type_variable(exp_type)));
    }
  case is_syntax_call : return(basic_of_call(syntax_call(sy)));
  case is_syntax_range : return(basic_of_expression(range_lower(syntax_range(sy))));
  default : pips_error("basic_of_expression", "Bad syntax tag");
    /* Never go there... */
    return make_basic(is_basic_overloaded, 4);
  }
}



/*============================================================================*/
/* basic basic_of_call(call c): returns the basic of the result given by the
 * call "c".
 */
basic basic_of_call(c)
call c;
{
  entity e = call_function(c);
  tag t = value_tag(entity_initial(e));

  switch (t)
    {
    case is_value_code: return(basic_of_external(c));
    case is_value_intrinsic: return(basic_of_intrinsic(c));
    case is_value_symbolic: break;
    case is_value_constant: return(basic_of_constant(c));
    case is_value_unknown: pips_error("basic_of_call", "unknown function %s\n",
				      entity_name(e));
    default: pips_error("basic_of_call", "unknown tag %d\n", t);
      /* Never go there... */
    }
  return make_basic(is_basic_overloaded,4 );
}



/*============================================================================*/
/* basic basic_of_external(call c): returns the basic of the result given by
 * the call to an external function.
 */
basic basic_of_external(c)
call c;
{
type call_type, return_type;

debug(7, "basic_of_call", "External call\n");

call_type = entity_type(call_function(c));
if (type_tag(call_type) != is_type_functional)
  pips_error("basic_of_external", "Bad call type tag");

return_type = functional_result(type_functional(call_type));
if (type_tag(return_type) != is_type_variable)
  pips_error("basic_of_external", "Bad return call type tag");

return(variable_basic(type_variable(return_type)));
}



/*============================================================================*/
/* basic basic_of_intrinsic(call c): returns the basic of the result given by
 * call to an intrinsic function. This basic must be computed with the
 * basic of the arguments of the intrinsic.
 */
basic basic_of_intrinsic(c)
call c;
{
  basic rb;
  entity call_func;

  debug(7, "basic_of_call", "Intrinsic call\n");

  call_func = call_function(c);

  if(ENTITY_LOGICAL_OPERATOR_P(call_func))
    rb = make_basic(is_basic_logical, 4);
  else
    {
      list call_args = call_arguments(c);
      if (call_args == NIL)
	/* I don't know the type since there is no arguments !
	   Bug encountered with a FMT=* in a PRINT.
	   RK, 21/02/1994 : */
	rb = make_basic(is_basic_overloaded, 1);
/*	rb = make_basic(is_basic_int, 4); */
      else {
      expression arg1, arg2;
	arg1 = EXPRESSION(CAR(call_args));
	if(CDR(call_args) == NIL)
	  rb = basic_of_expression(arg1);
	else
	  {
	    arg2 = EXPRESSION(CAR(CDR(call_args)));
	    rb = basic_union(arg1, arg2);
	  }

      }}
  return(rb);
}



/*============================================================================*/
/* basic basic_of_constant(call c): returns the basic of the call to a
 * constant.
 */
basic basic_of_constant(c)
call c;
{
type call_type, return_type;

debug(7, "basic_of_call", "Constant call\n");

call_type = entity_type(call_function(c));
if (type_tag(call_type) != is_type_functional)
  pips_error("basic_of_constant", "Bad call type tag");

return_type = functional_result(type_functional(call_type));
if (type_tag(return_type) != is_type_variable)
  pips_error("basic_of_constant", "Bad return call type tag");

return(variable_basic(type_variable(return_type)));
}



/*============================================================================*/
/* basic basic_union(expression exp1 exp2): returns the basic of the
 * expression which has the most global basic. Then, between "int" and
 * "float", the most global is "float".
 *
 * Note: there are two different "float" : DOUBLE PRECISION and REAL.
 */
basic basic_union(exp1, exp2)
expression exp1, exp2;
{
basic b1, b2;
tag t1, t2;

b1 = basic_of_expression(exp1);
b2 = basic_of_expression(exp2);
t1 = basic_tag(b1);
t2 = basic_tag(b2);

debug(7, "basic_union", "Tags: exp1 = %d, exp2 = %d\n", t1, t2);

if( (t1 != is_basic_complex) && (t1 != is_basic_float) &&
    (t1 != is_basic_int) && (t2 != is_basic_complex) &&
    (t2 != is_basic_float) && (t2 != is_basic_int) )
  pips_error("basic_union",
	     "Bad basic tag for expression in numerical function");

if(t1 == is_basic_complex)
  return(b1);
if(t2 == is_basic_complex)
  return(b2);
if(t1 == is_basic_float)
  {
  if( (t2 != is_basic_float) || (basic_float(b1) == DOUBLE_PRECISION_SIZE) )
    return(b1);
  return(b2);
  }
if(t2 == is_basic_float)
  return(b2);
return(b1);
}
