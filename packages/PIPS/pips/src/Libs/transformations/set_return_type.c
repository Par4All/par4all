/*

  $Id $

  Copyright 1989-2014 MINES ParisTech
  Copyright 2010 HPC-Project

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
#include "misc.h"
#include "properties.h"
#include "preprocessor.h"
#include "pipsdbm.h"
#include "resources.h"
#include "control.h"


bool set_return_type_as_typedef( const char *mod_name ) {
  // Use this module name and this environment variable to set
  statement module_statement = PIPS_PHASE_PRELUDE(mod_name,
              "SET_RETURN_TYPE_DEBUG_LEVEL");


  // User give the new type via a string property
  const char* s_new_type = get_string_property("SET_RETURN_TYPE_AS_TYPEDEF_NEW_TYPE");

  // We'll now try to recover a valid typedef from this string
  entity e_new_type = entity_undefined;

  list entities = gen_filter_tabulated(gen_true, entity_domain);
  FOREACH(entity, e, entities ) {
    /**
     *  Lookup over whole symbol table looking for a corresponding typedef
     *  This is unsecure since there can be typedef in different compilation
     *  unit sharing the same local name. The prefix in pips is random to
     *  distinguish between them
     */
    if(typedef_entity_p(e) && strcmp(entity_user_name(e),s_new_type) == 0) {
      e_new_type = e;
    }
  }

  pips_assert("Requested typedef must be defined", !entity_undefined_p(e_new_type));

  // Create the new return type
  type new_return_type = MakeTypeVariable(make_basic_typedef(e_new_type),NULL);

  entity func = module_name_to_entity(mod_name);
  pips_assert("func should be functionnal typed\n",type_functional_p(entity_type(func)));

  functional f = type_functional(entity_type(func));
  functional_result(f) = new_return_type;


  entity cu = module_entity_to_compilation_unit_entity(func);
  RemoveEntityFromCompilationUnit(func,cu);
  AddEntityToCompilationUnit(func,cu);


  // Put back the new statement module
  PIPS_PHASE_POSTLUDE(module_statement);
}
