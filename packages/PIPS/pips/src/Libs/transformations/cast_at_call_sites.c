/*

 $Id$

 Copyright 1989-2010 MINES ParisTech
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
#include "effects.h"
#include "accel-util.h"
#include "callgraph.h"



/**
 * This pass force call site params to be casted
 */
bool cast_at_call_sites(const char *mod_name) {
  // Use this module name and this environment variable to set
  statement module_statement = PIPS_PHASE_PRELUDE(mod_name,
      "CAST_AT_CALL_SITES_DEBUG_LEVEL");

  entity m = get_current_module_entity();

  list
      callers =
          callees_callees((callees)db_get_memory_resource(DBR_CALLERS,mod_name, true));
  list callers_statement = callers_to_statements(callers);
  list call_sites = callers_to_call_sites(callers_statement,m);
  /* we may have to change the call sites, prepare iterators over call sites arguments here */
  FOREACH(CALL,c,call_sites) {
    list args = call_arguments(c);
    FOREACH(PARAMETER,p,functional_parameters(type_functional(entity_type(m)))) {
      expression * arg = (expression*)REFCAR(args);
      type type_in_func_prototype = parameter_type(p);

      if(!pointer_type_p(type_in_func_prototype) && array_type_p(type_in_func_prototype)) {
        type t = copy_type(type_in_func_prototype);
        variable v = type_variable(t);
        variable_dimensions(v) = CDR(variable_dimensions(v));
        type_in_func_prototype = make_type_variable(make_variable(make_basic_pointer(t),NIL,NIL));
      }


      *arg = make_expression(make_syntax_cast(make_cast(copy_type(type_in_func_prototype),
                                                        *arg)),
                             normalized_undefined);
      POP(args);
    }
  }
  for (list citer = callers, siter = callers_statement; !ENDP(citer); POP(citer), POP(siter))
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, STRING(CAR(citer)),STATEMENT(CAR(siter)));

  // Put back the new statement module
  PIPS_PHASE_POSTLUDE(module_statement);
}
