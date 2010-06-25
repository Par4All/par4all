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
#include "properties.h"
#include "preprocessor.h"
 
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

