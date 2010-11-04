/*

  $Id: utils.c 17413 2010-06-24 06:59:11Z creusillet $

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
#include "ri-util.h"
#include "effects-util.h"
#include "misc.h"
#include "text-util.h"
 
/***************************************/

/**
   @param exp is an effect index expression which is either the rank or an entity corresponding to a struct, union or enum field
   @param l_fields is the list of fields of the corresponding struct, union or enum
   @return the entity corresponding to the field.
 */
entity effect_field_dimension_entity(expression exp, list l_fields)
{
  if(expression_constant_p(exp))
    {
      int rank = expression_to_int(exp);
      return ENTITY(gen_nth(rank-1, l_fields));
    }
  else
    {
      return expression_to_entity(exp);
    }
}

/**
   @brief recursively walks thru current_l_ind and current_type in parallel until a pointer dimension is found.
   @param current_l_ind is a list of effect reference indices.
   @param current_type is the corresponding type in the original entity type arborescence 
   @param exact_p is a pointer to a bool, which is set to true if the result is not an approximation.
   @return -1 if no index corresponds to a pointer dimension in current_l_ind, the rank of the least index that may correspond to
      a pointer dimension in current_l_ind otherwise. If this information is exact, *exact_p is set to true.
 */
static int effect_indices_first_pointer_dimension_rank(list current_l_ind, type current_type, bool *exact_p)
{
  int result = -1; /*assume there is no pointer */
  basic current_basic = variable_basic(type_variable(current_type));
  size_t current_nb_dim = gen_length(variable_dimensions(type_variable(current_type)));

  pips_debug(8, "input type : %s\n", type_to_string(current_type));
  pips_debug(8, "current_basic : %s, and number of dimensions %d\n", basic_to_string(current_basic), (int) current_nb_dim);

  pips_assert("there should be no effect on variable names\n", gen_length(current_l_ind) >= current_nb_dim);
  

  switch (basic_tag(current_basic)) 
    {
    case is_basic_pointer:
      {
	// no need to test if gen_length(current_l_ind) >= current_nb_dim because of previous assert
	result = (int) current_nb_dim;
	*exact_p = true;
	break;
      }
    case is_basic_derived:
      {
	int i;
	current_type = entity_type(basic_derived(current_basic));
	
	if (type_enum_p(current_type))
	  result = -1;
	else
	  {
	    
	    /*first skip array dimensions if any*/
	    for(i=0; i< (int) current_nb_dim; i++, POP(current_l_ind));
	    pips_assert("there must be at least one index left for the field\n", gen_length(current_l_ind) > 0);
	    
	    list l_fields = derived_type_fields(current_type);
	    
	    entity current_field_entity = effect_field_dimension_entity(EXPRESSION(CAR(current_l_ind)), l_fields);
	    
	    if (variable_phi_p(current_field_entity) || same_string_p(entity_local_name(current_field_entity), UNBOUNDED_DIMENSION_NAME))
	      {
		while (!ENDP(l_fields))
		  {
		    int tmp_result = -1;
		    entity current_field_entity = ENTITY(CAR(l_fields));
		    type current_type =  basic_concrete_type(entity_type(current_field_entity));
		    size_t current_nb_dim = gen_length(variable_dimensions(type_variable(current_type)));
		    
		    if (gen_length(CDR(current_l_ind)) >= current_nb_dim)
		      // consider this field only if it can be an effect on this field
		      tmp_result = effect_indices_first_pointer_dimension_rank(CDR(current_l_ind), current_type, exact_p);
		    
		    POP(l_fields);
		    if (tmp_result >= 0)
		      result = result < 0 ? tmp_result : (tmp_result <= result ? tmp_result : result);
		    free_type(current_type);
		  }
		
		*exact_p = (result < 0);
		if (result >= 0) result ++; // do not forget the field index !
	      }
	    else
	      {
		
		current_type = basic_concrete_type(entity_type(current_field_entity));
		result = effect_indices_first_pointer_dimension_rank(CDR(current_l_ind), current_type, exact_p);
		if (result >=0) result++; // do not forget the field index ! 
		free_type(current_type);
	      }
	  }
	break;
      }
    default:
      {
	result = -1;
	*exact_p = true;
	break;
      }
    }
  
  pips_debug(8, "returning %d\n", result);
  return result;
  
}


/**
   @brief walks thru ref indices and ref entity type arborescence in parallel until a pointer dimension is found.
   @param ref is an effect reference
   @param exact_p is a pointer to a bool, which is set to true if the result is not an approximation.
   @return -1 if no index corresponds to a pointer dimension, the rank of the least index that may correspond to
      a pointer dimension in current_l_ind otherwise. If this information is exact, *exact_p is set to true.
 */
int effect_reference_first_pointer_dimension_rank(reference ref, bool *exact_p)
{ 
  entity ent = reference_variable(ref);
  list current_l_ind = reference_indices(ref);
  type ent_type = entity_type(ent);
  int result;

  pips_debug(8, "input reference : %s\n", words_to_string(effect_words_reference(ref)));
  
  if (!type_variable_p(ent_type))
    {
      result = -1;
    }
  else
    {
      type current_type = basic_concrete_type(ent_type);
      result = effect_indices_first_pointer_dimension_rank(current_l_ind, current_type, exact_p);
      free_type(current_type);
    }

  return result;

}

/**
   @param ref is an effect reference
   @param exact_p is a pointer to a bool, which is set to true if the result is not an approximation.
   @return false if no index corresponds to a pointer dimension, false if any index may correspond to
      a pointer dimension. If this information is exact, *exact_p is set to true.
 */
bool effect_reference_contains_pointer_dimension_p(reference ref, bool *exact_p)
{
  int pointer_rank;
  pointer_rank = effect_reference_first_pointer_dimension_rank(ref, exact_p);
  return (pointer_rank >= 0);
}


/**
   @param ref is an effect reference
   @param exact_p is a pointer to a bool, which is set to true if the result is not an approximation.
   @return true if the effect reference may dereference a pointer, false otherwise.
 */
bool effect_reference_dereferencing_p(reference ref, bool * exact_p)
{
  list l_ind = reference_indices(ref);
  bool result;
  int p_rank;

  if (entity_all_locations_p(reference_variable(ref)))
    {
      result = true;
      *exact_p = false;
    }
  else
    {
      p_rank = effect_reference_first_pointer_dimension_rank(ref, exact_p);
  

      if (p_rank == -1)
	result = false;
      else
	result = p_rank < (int) gen_length(l_ind);
    }
  return result;
}

/*************************************************************************/
/* cell references */

static type r_variable_cell_reference_to_type(list ref_l_ind, type current_type)
{
  type t = type_undefined;  /* the return type */
  pips_assert("input type tag should be variable\n", type_variable_p(current_type));

  basic current_basic = variable_basic(type_variable(current_type)); /* current basic */
  list l_current_dim = variable_dimensions(type_variable(current_type)); /* current type array dimensions */
  size_t current_nb_dim = gen_length(l_current_dim);
     
  pips_debug(8, "input type : %s\n", type_to_string(current_type));
  pips_debug(8, "current_basic : %s, and number of dimensions %d\n", basic_to_string(current_basic), (int) current_nb_dim);

  /* the remainder of the function heavily relies on the following assumption */
  pips_assert("there should be no memory access paths to variable names\n", gen_length(ref_l_ind) >= current_nb_dim);

 /*first skip array dimensions if any*/
  for(int i=0; i< (int) current_nb_dim; i++, POP(ref_l_ind));
  
  if (ENDP(ref_l_ind)) /* We have reached the current basic */
    {
      /* Warning : qualifiers are set to NIL, because I do not see
	 the need for something else for the moment. BC.
      */
      t = make_type(is_type_variable, make_variable(copy_basic(current_basic), NIL, NIL));
    }
  else
    {
      /* The cell reference contains indices that go beyond the current type array dimensions.
         This can happen if and only if the current basic is a pointer or a derived 
         (typedef have been eliminated by the use of basic_concrete_type).
      */
      switch (basic_tag(current_basic))
	{
	case is_basic_pointer:
	  {
	    type new_current_type = basic_concrete_type(basic_pointer(current_basic));
	    POP(ref_l_ind); /* pop the pointer dimension */
	    t = r_variable_cell_reference_to_type(ref_l_ind, new_current_type);
	    free_type(new_current_type);
	    break;
	  }
	case is_basic_derived:
	  {
	    /* the next reference index should be a field entity */
	    expression next_index = EXPRESSION(CAR(ref_l_ind));
	    syntax s = expression_syntax(next_index);
	    if (syntax_reference_p(s))
	      {
		entity next_index_e = reference_variable(syntax_reference(s));
		if (entity_field_p(next_index_e))
		  {
		    type new_current_type = basic_concrete_type(entity_type(next_index_e));
		    POP(ref_l_ind); /* pop the field dimension */
		    t = r_variable_cell_reference_to_type(ref_l_ind, new_current_type);
		    free_type(new_current_type);
		  }
		else
		  pips_internal_error("the current basic tag is derived, but corresponding index is not a field entity\n");
	      }
	    else
	      pips_internal_error("the current basic tag is derived, but corresponding index is not a reference\n");
	    break;
	  }      
	default:
	  {
	    pips_internal_error("unexpected basic tag\n");
	  }
	}
    }
      
  ifdebug(8)
    {
      variable v = type_variable(t);
      pips_debug(8, "output type is: %s\n",  type_to_string(t));
      pips_debug(8, "with basic : %s, and number of dimensions %d\n", 
		 basic_to_string(variable_basic(v)), 
		 (int) gen_length(variable_dimensions(v)));
    }
  return t;
}

/**
 @brief computes the type of a cell reference representing a memory access path. 
        Cell references are not compatible with entity typing: spurious dimensions 
        are added to handle struct fields and the dereferencing operator.
	BEWARE : does not work if field entity indices have been converted to ranks.
 @param ref is a reference from a cell.
 @return a *newly allocated* type corresponding to the type of the memory cells targeted by the access path.
 */
type cell_reference_to_type(reference ref)
{
  type t = type_undefined;
  type ref_type = basic_concrete_type(entity_type(reference_variable(ref)));

  pips_debug(6, "input reference: %s \n",  words_to_string(words_reference(ref,NIL)));

  if(type_variable_p(ref_type))
    {
      t = r_variable_cell_reference_to_type(reference_indices(ref), ref_type);
    }
  else if(type_functional_p(ref_type))
    {
      /* A reference to a function returns a pointer to a function
	 of the very same time */
      t = make_type(is_type_variable,
		    make_variable
		    (make_basic(is_basic_pointer, copy_type(ref_type)),
		     NIL, NIL));
    }
  else
    {
      pips_internal_error("Bad reference type tag %d \"%s\" for reference %s\n",
			  type_tag(ref_type),
			  type_to_string(ref_type),
			  entity_name(reference_variable(ref)));
    }
  free_type(ref_type);

  return t;
}

type cell_to_type(cell c)
{
  pips_assert("a cell cannot be a gap yet\n", !cell_gap_p(c));
  reference ref = cell_reference_p(c)? cell_reference(c) : preference_reference(cell_preference(c));

  return cell_reference_to_type(ref);
}
