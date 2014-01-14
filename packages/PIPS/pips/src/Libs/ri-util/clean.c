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
#include <stdio.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"

#include "misc.h"
#include "ri-util.h"



static entity current_function = entity_undefined;
static const char *current_function_local_name = NULL;


static void set_current_function(function)
entity function;
{
    pips_assert("set_current_function", top_level_entity_p(function)
		|| compilation_unit_p(entity_module_name(function)));
    current_function = function;
    current_function_local_name = module_local_name(function);
}



static bool local_entity_of_current_function_p(e)
entity e;
{
    return(same_string_p(entity_module_name(e), current_function_local_name));
}


/* Useful when the user edit a source file and parse it again or when
   a program transformation is performed by prettyprinting and
   reparsing. */
void GenericCleanEntities(list el, entity function, bool fortran_p)
{
  /* FI: A memory leak is better than a dangling pointer? */
  /* My explanation: CleanLocalEntities is called by MakeCurrentFunction when
   * the begin_inst rule is reduced; by that time, the module entity and its
   * formal parameters have already been parsed and allocated... but they
   * are not yet fully initialized; thus the gen_full_free is a killer while
   * the undefinitions are ok.
   */
  /*gen_full_free_list(function_local_entities);*/

  list pe = list_undefined;

  for(pe=el; !ENDP(pe); POP(pe)) {
    entity e = ENTITY(CAR(pe));
    storage s = entity_storage(e);

    pips_debug(8, "Clean up %s?\n", entity_local_name(e));

    if(!storage_undefined_p(s) && storage_ram_p(s)) {
      entity sec = ram_section(storage_ram(s));
      type t = entity_type(sec);
      if(!type_undefined_p(t)) {
          pips_assert("t is an area", type_area_p(t));
          gen_remove(&(area_layout(type_area(t))),e);
      }
    }

    /* In C, parameter typing may have already occured and the return
       entity may be already defined. They must be preserved. */
    if(fortran_p
       || (!storage_undefined_p(entity_storage(e))
	   && !storage_formal_p(entity_storage(e))
	   && !storage_return_p(entity_storage(e)))) {
      pips_debug(8, "Clean up %s? YES\n", entity_local_name(e));
      if(!type_undefined_p(entity_type(e))) {
	/* free_type(entity_type(e)); */
	entity_type(e) = type_undefined;
      }
      if(!storage_undefined_p(entity_storage(e))) {
	/* free_storage(entity_storage(e)); */
	entity_storage(e) = storage_undefined;
      }
      if(!value_undefined_p(entity_initial(e))) {
	/* free_value(entity_initial(e)); */
	entity_initial(e) = value_undefined;
      }
      if(!fortran_p) {
	/* In C, the entity must be removed from the declarations and
	   from the symbol table */
	gen_remove(&(code_declarations(value_code(entity_initial(function)))),e);
	free_entity(e);
      }
    }
    else {
      pips_debug(8, "Clean up %s? NO\n", entity_local_name(e));
    }
  }
}

void GenericCleanLocalEntities(entity function, bool fortran_p)
{
  list function_local_entities = NIL;
  //list pe = NIL;

  set_current_function(function);

  function_local_entities =
    gen_filter_tabulated(local_entity_of_current_function_p, 
			 entity_domain);

  GenericCleanEntities(function_local_entities, function, fortran_p);

  gen_free_list(function_local_entities);
}

/* Fortran version */
void CleanLocalEntities(entity function)
{
  GenericCleanLocalEntities(function, true);
}

/* C language version */
void CCleanLocalEntities(entity function)
{
  GenericCleanLocalEntities(function, false);
}

/* Useful for ParserError()? */
void RemoveLocalEntities(function)
entity function;
{

    set_current_function(function);

#if 0
    list function_local_entities =
	gen_filter_tabulated(local_entity_of_current_function_p, 
			     entity_domain);
#endif

    /* FI: dangling pointers? Some variables may be referenced in area_layouts of
       global common entities! */
    /* gen_full_free_list(function_local_entities); */
    /* A gen_multi_recurse would be required but it's hard to be at the
       list level to remove the elements?!? */
    pips_assert("implemented", false);
}
