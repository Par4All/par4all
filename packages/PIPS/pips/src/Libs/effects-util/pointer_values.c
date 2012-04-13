/*

  $Id$

  Copyright 1989-2010 MINES ParisTech
  Copyright 2010 HPC Project

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

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "effects.h"
#include "effects-util.h"
#include "misc.h"

/***************** ABSTRACT VALUES */

entity undefined_pointer_value_entity()
{
  entity u = entity_undefined;
  string u_name = strdup(concatenate(ANY_MODULE_NAME,
				     MODULE_SEP_STRING,
				     UNDEFINED_POINTER_VALUE_NAME,
				     NULL));
  u = gen_find_tabulated(u_name, entity_domain);
  if(entity_undefined_p(u)) {
    type tv = make_type_void(NIL);
    variable v = make_variable(make_basic_pointer(tv), NIL, NIL);
    type t = make_type_variable(v);
    u = make_entity(u_name,
		    t, make_storage_rom(), make_value_unknown());
    entity_kind(u)=ABSTRACT_LOCATION;
  }
  return u;
}


cell make_undefined_pointer_value_cell()
{
  entity u = undefined_pointer_value_entity();
  return make_cell_reference(make_reference(u, NIL));
}

bool undefined_pointer_value_entity_p(entity e)
{
  bool res;
  res = same_string_p(entity_local_name(e), UNDEFINED_POINTER_VALUE_NAME);
  res = res && same_string_p(entity_module_name(e), ANY_MODULE_NAME);
  return res;
}

bool undefined_pointer_value_cell_p(cell c)
{
  reference r;
  if (cell_gap_p(c)) return false;
  else if (cell_reference_p(c))
    r = cell_reference(c);
  else
    r = preference_reference(cell_preference(c));
  return(undefined_pointer_value_entity_p(reference_variable(r)));
}

entity null_pointer_value_entity()
{
  return entity_null_locations();
}


cell make_null_pointer_value_cell()
{
  entity u = null_pointer_value_entity();
  return make_cell_reference(make_reference(u, NIL));
}

bool null_pointer_value_entity_p(entity e)
{
  return entity_null_locations_p(e);
}

bool null_pointer_value_cell_p(cell c)
{
  reference r;
  if (cell_gap_p(c)) return false;
  else if (cell_reference_p(c))
    r = cell_reference(c);
  else
    r = preference_reference(cell_preference(c));
  return(null_pointer_value_entity_p(reference_variable(r)));
}

bool abstract_pointer_value_entity_p(entity e)
{
  return (undefined_pointer_value_entity_p(e)
	  || null_pointer_value_entity_p(e));
}

bool abstract_pointer_value_cell_p(cell c)
{
  return (abstract_pointer_value_entity_p(cell_entity(c)));
}

/***************** SHORTCUTS FOR MAKING POINTER_VALUES CELL_RELATIONS */


cell_relation make_value_of_pointer_value(cell c1, cell c2, tag app_tag, descriptor d)
{
  interpreted_cell ic1 = make_interpreted_cell(c1, make_cell_interpretation_value_of());
  interpreted_cell ic2 = make_interpreted_cell(c2, make_cell_interpretation_value_of());
  cell_relation pv = cell_relation_undefined;

  if (null_pointer_value_cell_p(c1) || undefined_pointer_value_cell_p(c1))
      pv = make_cell_relation(ic2, ic1, make_approximation(app_tag, UU), d);
  else
    pv = make_cell_relation(ic1, ic2, make_approximation(app_tag, UU), d);
  return(pv);
}

cell_relation make_address_of_pointer_value(cell c1, cell c2, tag app_tag, descriptor d)
{
  interpreted_cell ic1 = make_interpreted_cell(c1, make_cell_interpretation_value_of());
  interpreted_cell ic2 = make_interpreted_cell(c2, make_cell_interpretation_address_of());
  cell_relation pv = make_cell_relation(ic1, ic2, make_approximation(app_tag, UU), d);
  return(pv);
}

/***************** UTILS */

/*
  @brief tests the syntactic equality of the corresponding cells of two pointer_value relations
  @param pv1 is a pointer_value
  @param pv2 is another pointer value
  @return true if the input pointer values are syntactically equal.

  if both pvs are value_of pvs, they are considered equal if their first cells are equal
  and second cells are equal but also if the first cell of the first pv is equal to
  the second cell of the second pv and conversely.
 */
bool pv_cells_syntactically_equal_p(cell_relation pv1, cell_relation pv2)
{


  bool value_of_1_p = cell_relation_second_value_of_p(pv1);
  bool value_of_2_p = cell_relation_second_value_of_p(pv1);

  if ( (value_of_1_p && !value_of_2_p) || (value_of_2_p && !value_of_1_p))
    return false;

  cell c_first_1 = cell_relation_first_cell(pv1);
  cell c_second_1 = cell_relation_second_cell(pv1);

  cell c_first_2 = cell_relation_first_cell(pv2);
  cell c_second_2 = cell_relation_second_cell(pv2);

  int n_first_first = cell_compare(&c_first_1, &c_first_2);

  if (n_first_first == 0)
    {
      int n_second_second = cell_compare(&c_second_1, &c_second_2);
      
      if (n_second_second != 0)
	return false;
    }
  else
    {
      if (!value_of_1_p)
	return false;
      else /* value_of pvs, try to see if their cells are inverted */
	{
	  int n_first_second = cell_compare(&c_first_1, &c_second_2);
	  if (n_first_second == 0)
	    {
	      int n_second_first = cell_compare(&c_second_1, &c_first_2);
	      
	      if (n_second_first != 0)
		return false;
	    }	
	  else
	    return false;
	  
	}
    }
  
  descriptor d1 = cell_relation_descriptor(pv1);
  descriptor d2 = cell_relation_descriptor(pv1);

  if (descriptor_none_p(d1) && descriptor_none_p(d2))
    {
      return true;
    }
  else
    pips_internal_error("Convex pointer_values not implemented yet");
  
  return false;
}

/*
  @brief tests the syntactic equality of the corresponding cells of two pointer_value relations
  @param pv1 is a pointer_value
  @param pv2 is another pointer value
  @return true if the input pointer values are syntactically equal.

  if both pvs are value_of pvs, they are considered equal if their first cells are equal
  and second cells are equal but also if the first cell of the first pv is equal to
  the second cell of the second pv and conversely.
 */
bool pv_cells_mergeable_p(cell_relation pv1, cell_relation pv2)
{


  bool value_of_1_p = cell_relation_second_value_of_p(pv1);
  bool value_of_2_p = cell_relation_second_value_of_p(pv1);

  if ( (value_of_1_p && !value_of_2_p) || (value_of_2_p && !value_of_1_p))
    return false;

  cell c_first_1 = cell_relation_first_cell(pv1);
  cell c_second_1 = cell_relation_second_cell(pv1);

  cell c_first_2 = cell_relation_first_cell(pv2);
  cell c_second_2 = cell_relation_second_cell(pv2);

  int n_first_first = cell_compare(&c_first_1, &c_first_2);

  if (n_first_first == 0)
    {
      int n_second_second = cell_compare(&c_second_1, &c_second_2);
      
      if (n_second_second != 0)
	{
	  if (cell_entity(c_second_1) != cell_entity(c_second_2)
	      || (gen_length(reference_indices(cell_any_reference(c_second_1)))
		  != gen_length(reference_indices(cell_any_reference(c_second_2)))))
	    return false;
	}
    }
  else
    {
      if (!value_of_1_p)
	return false;
      else /* value_of pvs, try to see if their cells are inverted */
	{
	  int n_first_second = cell_compare(&c_first_1, &c_second_2);
	  if (n_first_second == 0)
	    {
	      int n_second_first = cell_compare(&c_second_1, &c_first_2);
	      
	      if (n_second_first != 0)
		return false;
	    }	
	  else
	    return false;
	  
	}
    }
  
  descriptor d1 = cell_relation_descriptor(pv1);
  descriptor d2 = cell_relation_descriptor(pv1);

  if (descriptor_none_p(d1) && descriptor_none_p(d2))
    {
      return true;
    }
  else
    pips_internal_error("Convex pointer_values not implemented yet");
  
  return false;
}

