/*

  $Id$

  Copyright 1989-2010 MINES ParisTech
  Copyright 2009-2010 HPC Project

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
/* package abstract location. Amira Mensi 2010
 *
 * File: abstract_location.c
 *
 *
 * This file contains various useful functions to modelize a heap.
 *
 */
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "points_to_private.h"
#include "ri-util.h"
#include "effects-util.h"
#include "text.h"
#include "text-util.h"
#include "misc.h"
#include "properties.h"


/*
  Heap Modelization

  See pipsmake-rc.tex, propertiews ABSTRACT_HEAP_LOCATIONS and
  ALIASING_ACROSS_TYPES, for documentation.
*/

/* check if an entity b may represent a bucket or a set of buckets in
   the heap. */
bool entity_heap_location_p(entity b)
{
  bool bucket_p = entity_all_heap_locations_p(b) ||
    entity_all_module_heap_locations_p(b);

  if(!bucket_p) {
      const char* ln = entity_local_name(b);
      const char* found = strstr(ln, HEAP_AREA_LOCAL_NAME);
      bucket_p = found != NULL;
  }

  return bucket_p;
}

entity entity_flow_or_context_sentitive_heap_location(int stmt_number, type t)
{
  entity e;
  string s;
  asprintf(&s,HEAP_AREA_LOCAL_NAME "_l_%d",stmt_number);

  e = FindOrCreateEntity(get_current_module_name(),s);
  free(s);
  if(type_undefined_p(entity_type(e))) {
    entity f = get_current_module_entity();
    entity a = module_to_heap_area(f);
    ram r = make_ram(f, a, UNKNOWN_RAM_OFFSET, NIL);

    /* FI: Beware, the symbol table is updated but this is not
       reflected in pipsmake.rc */
    type ct = compute_basic_concrete_type(t);
    entity_type(e) = ct;
    entity_storage(e) = make_storage_ram(r);
    entity_initial(e) = make_value_unknown();
    entity_kind(e) = ABSTRACT_LOCATION;
    (void) add_C_variable_to_area(a, e);
  }
  else {
    /* We might be in trouble, unless a piece of code is
       reanalyzed. Let's assume the type is unchanged */
    type ct = compute_basic_concrete_type(t);
    type et = entity_type(e);
    // FI: too strong if 1-D arrays and pointers can be assimilated
    //pips_assert("The type is unchanged",
    // type_equal_p(ct, et));
    if(!type_equal_p(ct,et)) {
      if(pointer_type_p(ct) && array_type_p(et)) {
	type pt = type_to_pointed_type(ct);
	basic pb = variable_basic(type_variable(pt));
	variable etv = type_variable(et);
	basic eb = variable_basic(etv);
	int d = (int) gen_length(variable_dimensions(etv));
	if(d==1 && basic_equal_p(pb, eb)) 
	  ;
	else
	  pips_assert("The type is unchanged or compatible", false);
      }
      else if(pointer_type_p(et) && array_type_p(ct)) {
	type pt = type_to_pointed_type(et);
	basic pb = variable_basic(type_variable(pt));
	variable etv = type_variable(ct);
	basic eb = variable_basic(etv);
	int d = (int) gen_length(variable_dimensions(etv));
	if(d==1 && basic_equal_p(pb, eb)) 
	  ;
	else
	  pips_assert("The type is unchanged or compatible", false);
      }
      else
	pips_assert("The type is unchanged or compatible", false);
    }
    free_type(ct);
  }
  return e;

}

bool entity_flow_or_context_sentitive_heap_location_p(entity e)
{
  bool result = false;
  const char* ln = entity_local_name(e);
  string found = strstr(ln, ANYWHERE_LOCATION);

  pips_debug(9, "input entity: %s\n", ln);
  pips_debug(9, "found (1) = : %s\n", found);

  if (found == NULL)
    {
      found = strstr(ln, HEAP_AREA_LOCAL_NAME);
      pips_debug(9, "found (2) = : %s\n", found);
      if (found!=NULL)
	{
	  size_t found_n = strspn(found, HEAP_AREA_LOCAL_NAME);
	  ln = &found[found_n];
	  pips_debug(9, "ln : %s\n", ln);
	  found = strstr(ln, "_l_");
	  pips_debug(9, "found (3) = : %s\n", found);
	  result = (found != NULL);
	}
      else
	result = false;
    }
  else
    result = false;
  pips_debug(9, "result = %d\n", (int) result);
  return result;
}

/**
   @brief generate the type of the allocated area from the malloc argument
   @param e is the malloc argument expression
   @return a new type (no sharing)

   Store dependent types are not generated. It means that whenever the
   size of the allocated space is store dependent, the returned type
   is an array of unbounded_dimension.
 */
type malloc_arg_to_type(expression e)
{
  type t = type_undefined;
  syntax s = expression_syntax(e);
  expression n = expression_undefined;
  sizeofexpression sizeof_exp = sizeofexpression_undefined;
  bool scalar_p = false;;

  pips_debug(5, "begin for expression:%s\n",
	     words_to_string(words_expression(e, NIL)));

  if (syntax_sizeofexpression_p(s))
    {
      scalar_p = true;
      sizeof_exp = syntax_sizeofexpression(s);
    }
  else if (syntax_call_p(s))
    {
      call c = syntax_call(s);
      entity func = call_function(c);
      list func_args = call_arguments(c);

      if (same_string_p(entity_local_name(func), MULTIPLY_OPERATOR_NAME))
	{
	  pips_debug(5, "multiply operator found\n");
	  expression arg1 = EXPRESSION(CAR(func_args));
	  expression arg2 = EXPRESSION(CAR(CDR(func_args)));

	  /* which one is the sizeof operator ? try the second one first */
	  if (syntax_sizeofexpression_p(expression_syntax(arg2)))
	    {
	      pips_debug(5," second arg is a sizeof expression\n");
	      sizeof_exp = syntax_sizeofexpression(expression_syntax(arg2));
	      n = make_op_exp("-", copy_expression(arg1), int_to_expression(1));
	    }
	  else if (syntax_sizeofexpression_p(expression_syntax(arg1)))
	    {
	      pips_debug(5," first arg is a sizeof expression\n");
	      sizeof_exp = syntax_sizeofexpression(expression_syntax(arg1));
	      n = make_op_exp("-", copy_expression(arg2), int_to_expression(1));
	    }
	  else
	    {
	      n = make_unbounded_expression();
	    }
	}
      else
	{
	  n = make_op_exp("-", copy_expression(e), int_to_expression(1));
	}
    }
  else
    n = make_op_exp("-", copy_expression(e), int_to_expression(1));

  if (!expression_undefined_p(n) && !unbounded_expression_p(n)
      && !expression_constant_p(n))
    {
      pips_debug(5, "non constant number of elements "
		 "-> generating unbounded dimension\n");
      free_expression(n);
      n = make_unbounded_expression();
    }

  if (sizeofexpression_undefined_p(sizeof_exp))
    {
      t = make_type_variable
	(make_variable(make_basic_int(DEFAULT_CHARACTER_TYPE_SIZE),
		       CONS(DIMENSION,
			    make_dimension(int_to_expression(0), n),NIL),
		       NIL));
    }
  else if (sizeofexpression_type_p(sizeof_exp))
    {
      type t_sizeof_exp = sizeofexpression_type(sizeof_exp);
      variable t_var_sizeof_exp = type_variable_p(t_sizeof_exp) ?
	type_variable(t_sizeof_exp) : variable_undefined;
      pips_assert("type must be variable",
		  !variable_undefined_p(t_var_sizeof_exp));

      if (scalar_p)
	t = make_type_variable
	  (make_variable(copy_basic(variable_basic(t_var_sizeof_exp)),
			 gen_full_copy_list(variable_dimensions
					    (t_var_sizeof_exp)),
			 NIL));
      else
	t = make_type_variable
	  (make_variable(copy_basic(variable_basic(t_var_sizeof_exp)),
			 CONS(DIMENSION,
			      make_dimension(int_to_expression(0), n),
			      gen_full_copy_list(variable_dimensions
						 (t_var_sizeof_exp))),
			 NIL));
    }
  else /* sizeofexpression is an expression */
    {
      type t_sizeof_exp =
	expression_to_type(sizeofexpression_expression(sizeof_exp));
      variable t_var_sizeof_exp = type_variable_p(t_sizeof_exp) ?
	type_variable(t_sizeof_exp) : variable_undefined;

      if (scalar_p)
	t = make_type_variable
	  (make_variable(copy_basic(variable_basic( t_var_sizeof_exp)),
			 gen_full_copy_list(variable_dimensions
					    (t_var_sizeof_exp)),
			 NIL));
      else
	t = make_type_variable
	  (make_variable(copy_basic(variable_basic(t_var_sizeof_exp)),
			 CONS(DIMENSION,
			      make_dimension(int_to_expression(0), n),
			      gen_full_copy_list(variable_dimensions
						 (t_var_sizeof_exp))),
			 NIL));
      free_type(t_sizeof_exp);
    }
  pips_debug(5, "end with type %s\n",
	     words_to_string(words_type(t, NIL, false)));
  return t;
}

/**
   @brief generate an abstract heap location entity
   @param t is type of the allocated space
   @param psi is a pointer towards a structure which contains context
               and/or flow sensitivity information
   @return an abstract heap location entity
 */
entity malloc_type_to_abstract_location(type t, sensitivity_information *psi)
{
    entity e = entity_undefined;
  const char* opt = get_string_property("ABSTRACT_HEAP_LOCATIONS");
  bool type_sensitive_p = !get_bool_property("ALIASING_ACROSS_TYPES");

  pips_debug(8, "begin for type %s\n",
	     words_to_string(words_type(t, NIL, false)));
  /* in case we want an anywhere abstract heap location : the property
     ABSTRACT_HEAP_LOCATIONS is set to "unique" and a unique abstract
     location is used for all heap buckets. */
  if(strcmp(opt, "unique")==0){
    if(type_sensitive_p)
      e = entity_all_heap_locations_typed(t);
    else
      e = entity_all_heap_locations();
  }

  /* in case the property ABSTRACT_HEAP_LOCATIONS is set to
     "insensitive": an abstract location is used for each function. */
  else if(strcmp(opt, "insensitive")==0){
    if(type_sensitive_p)
      e = entity_all_module_heap_locations_typed(psi->current_module,
						 copy_type(t));
    else
      e = entity_all_module_heap_locations(psi->current_module);
  }

  /* in case the property ABSTRACT_HEAP_LOCATIONS is set to
     "flow-sensitive" or "context-sensitive".

     No difference here between the two values. The diffferent
     behavior will show when points-to and effects are translated at
     call sites

     At this level, we want to use the statement number or the
     statement ordering. The statement ordering is safer because two
     statements or more can be put on the same line. The statement
     number is more user-friendly.

     There is no need to distinguish between types since the statement
     ordering is at least as effective.
  */
  else if(strcmp(opt, "flow-sensitive")==0
	  || strcmp(opt, "context-sensitive")==0 )
    e = entity_flow_or_context_sentitive_heap_location
      (statement_number(psi->current_stmt),
       copy_type(t));
  else
    pips_user_error("Unrecognized value for property ABSTRACT_HEAP_LOCATION:"
		    " \"%s\"", opt);

  pips_debug(8, "returning entity %s of type %s\n", entity_name(e),
	     words_to_string(words_type(entity_type(e), NIL, false)));

  return e;
}

/**
   @brief generate an abstract heap location entity
   @param malloc_exp is the argument expression of the call to malloc
   @param psi is a pointer towards a structure which contains context
               and/or flow sensitivity information
   @return an abstract heap location entity
 */
entity malloc_to_abstract_location(expression malloc_exp,
				   sensitivity_information *psi)
{
  entity e = entity_undefined;

  pips_debug(8, "begin for expression %s\n",
	     words_to_string(words_expression(malloc_exp, NIL)));
  type t = malloc_arg_to_type(malloc_exp);
  e = malloc_type_to_abstract_location(t, psi);
  free_type(t);
  pips_debug(8, "returning entity %s of type %s\n", entity_name(e),
	     words_to_string(words_type(entity_type(e), NIL, false)));

  return e;
}

/**
   @brief generate an abstract heap location entity
   @param n is the first argument expression of the call to calloc
   @param size is the second argument expression of the call to calloc
   @param psi is a pointer towards a structure which contains context
               and/or flow sensitivity information
   @return an abstract heap location entity
 */
entity calloc_to_abstract_location(expression n, expression size,
				   sensitivity_information *psi)
{
  expression malloc_exp = make_op_exp("*",
				      copy_expression(n),
				      copy_expression(size));
  entity e = malloc_to_abstract_location(malloc_exp, psi);
  free_expression(malloc_exp);
  return(e);
}

/* to handle malloc instruction : type t = (cast)
 * malloc(sizeof(expression). This function return a reference
 * according to the value of the property ABSTRACT_HEAP_LOCATIONS */
/* the list of callers will be added to ensure the context sensitive
   property. We should keep in mind that context and sensitive
   properties are orthogonal and we should modify them in pipsmake.*/
reference original_malloc_to_abstract_location(expression lhs,
					       type  __attribute__ ((unused)) var_t,
					       type __attribute__ ((unused)) cast_t,
					       expression sizeof_exp,
					       entity f,
					       statement stmt)
{
  reference r = reference_undefined;
  entity e = entity_undefined;
  sensitivity_information si =
    make_sensitivity_information(stmt,
				 f,
				 NIL);
  //string opt = get_string_property("ABSTRACT_HEAP_LOCATIONS");
  //bool type_sensitive_p = !get_bool_property("ALIASING_ACROSS_TYPES");

  e = malloc_to_abstract_location(sizeof_exp, &si);
  if(!entity_array_p(e))
    r = make_reference(e , NIL);
  else
  r = make_reference(e , CONS(EXPRESSION, int_to_expression(0), NIL));
  /* /\* in case we want an anywhere abstract heap location : the property */
/*      ABSTRACT_HEAP_LOCATIONS is set to "unique" and a unique abstract */
/*      location is used for all heap buckets. *\/ */
/*   if(strcmp(opt, "unique")==0){ */
/*     if(type_sensitive_p) { */
/*       e = entity_all_heap_locations_typed(var_t); */
/*       r = make_reference(e , NIL); */
/*     } */
/*     else { */
/*       e = entity_all_heap_locations(); */
/*       r = make_reference(e , NIL); */
/*     } */
/*   } */

/*   /\* in case the property ABSTRACT_HEAP_LOCATIONS is set to */
/*      "insensitive": an abstract location is used for each function. *\/ */
/*   else if(strcmp(opt, "insensitive")==0){ */
/*     if(type_sensitive_p) { */
/*       e = entity_all_module_heap_locations_typed(get_current_module_entity(), */
/* 						 var_t); */
/*       r = make_reference(e , NIL); */
/*     } */
/*     else { */
/*       e = entity_all_module_heap_locations(f); */
/*       r = make_reference(e , NIL); */
/*     } */
/*   } */

/*   /\* in case the property ABSTRACT_HEAP_LOCATIONS is set to */
/*      "flow-sensitive" or "context-sensitive". */

/*      No difference here between the two values. The diffferent */
/*      behavior will show when points-to and effects are translated at */
/*      call sites */

/*      At this level, we want to use the statement number or the */
/*      statement ordering. The statement ordering is safer because two */
/*      statements or more can be put on the same line. The statement */
/*      number is more user-friendly. */

/*      There is no need to distinguish between types since the statement */
/*      ordering is at least as effective. */
/*   *\/ */
/*   else if(strcmp(opt, "flow-sensitive")==0 */
/* 	  || strcmp(opt, "context-sensitive")==0 ){ */
/*     e = entity_flow_or_context_sentitive_heap_location(stmt_number, var_t); */
/*     r = make_reference(e , NIL); */
/*   } */
/*   else { */
/*     pips_user_error("Unrecognized value for property ABSTRACT_HEAP_LOCATION:" */
/* 		    " \"%s\"", opt); */
/*   } */

/*   pips_debug(8, "Reference to "); */

  return r;
}

/*
 * FLOW AND CONTEXT SENSITIVITY INFORMATION HOOK
 *
 * FI->AM: How come the properties are not exploited here?
 *
 * FI->AM: do you understand that the data structure "si" is copied back?
 */
sensitivity_information make_sensitivity_information(statement current_stmt,
						     entity current_module,
						     list enclosing_flow)
{
  sensitivity_information si;
  si.current_stmt = current_stmt;
  si.current_module = current_module;
  si.enclosing_flow = enclosing_flow;
  return si;
}

