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
    area a = make_area(0,NIL); /* Size and layout are unknown */
    type t = make_type_area(a);
    u = make_entity(u_name,
		    t, make_storage_rom(), make_value_unknown());
  }
  return u;
}

entity null_pointer_value_entity()
{
  entity u = entity_undefined;
  string u_name = strdup(concatenate(ANY_MODULE_NAME,
				     MODULE_SEP_STRING,
				     NULL_POINTER_VALUE_NAME,
				     NULL));
  u = gen_find_tabulated(u_name, entity_domain);
  if(entity_undefined_p(u)) {
    area a = make_area(0,NIL); /* Size and layout are unknown */
    type t = make_type_area(a);
    u = make_entity(u_name,
		    t, make_storage_rom(), make_value_unknown());
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

cell make_null_pointer_value_cell()
{
  entity u = null_pointer_value_entity();
  return make_cell_reference(make_reference(u, NIL));
}

bool null_pointer_value_entity_p(entity e)
{
  bool res;
  res = same_string_p(entity_local_name(e), NULL_POINTER_VALUE_NAME);
  res = res && same_string_p(entity_module_name(e), ANY_MODULE_NAME);
  return res;  
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



/***************** SHORTCUTS FOR MAKING POINTER_VALUES CELL_RELATIONS */


cell_relation make_value_of_pointer_value(cell c1, cell c2, tag app_tag, descriptor d)
{
  interpreted_cell ic1 = make_interpreted_cell(c1, make_cell_interpretation_value_of());
  interpreted_cell ic2 = make_interpreted_cell(c2, make_cell_interpretation_value_of());
	      
  cell_relation pv = make_cell_relation(ic1, ic2, make_approximation(app_tag, UU), d);
  return(pv);
}

cell_relation make_address_of_pointer_value(cell c1, cell c2, tag app_tag, descriptor d)
{
  interpreted_cell ic1 = make_interpreted_cell(c1, make_cell_interpretation_value_of());
  interpreted_cell ic2 = make_interpreted_cell(c2, make_cell_interpretation_address_of());
	      
  cell_relation pv = make_cell_relation(ic1, ic2, make_approximation(app_tag, UU), d);
  return(pv);
}
