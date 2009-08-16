/*

  $Id$

  Copyright 1989-2009 MINES ParisTech

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

#include "genC.h"
#include "linear.h"

// newgen
#include "ri.h"
// needs some arg_label & vertex_label types
// #include "graph.h"

#include "properties.h"
#include "misc.h"
#include "ri-util.h"

#include "resources.h"
#include "pipsdbm.h"

#include "control.h" // for clean_up_sequences
// #include "hwac.h"
extern void freia_spoc_compile(string, statement);

#define HWAC_INPUT "HWAC_INPUT"
#define HWAC_TARGET "HWAC_TARGET"

#define FREIA_API "freia"
#define SPOC_HW "spoc"

int hardware_accelerator(string module)
{
  string input = get_string_property(HWAC_INPUT);
  string target = get_string_property(HWAC_TARGET);

  debug_on("PIPS_HWAC_DEBUG_LEVEL");

  // this will be usefull
  statement mod_stat =
    (statement) db_get_memory_resource(DBR_CODE, module, true);
  set_current_module_statement(mod_stat);
  set_current_module_entity(module_name_to_entity(module));

  // proper effects, chains ? summary effects ??
  /*
  set_proper_rw_effects((statement_effects)
	db_get_memory_resource(DBR_PROPER_EFFECTS, module_name,	true));

  graph chains = (graph) db_get_memory_resource(DBR_CHAINS, module_name, true);
  */

  pips_debug(1, "considering module %s\n", module);

  if (!same_string_p(input, FREIA_API))
    pips_internal_error("expecting '%s' input, got '%s'",
			FREIA_API, input);

  if (!same_string_p(target, SPOC_HW))
    pips_internal_error("expecting '%s' target hardware, got '%s'",
			SPOC_HW, target);

  // just call a stupid algorithm for testing purposes...
  // if (same_string_p(input, FREIA_API) && same_string_p(target, SPOC_HW))
  freia_spoc_compile(module, mod_stat);

  // some cleanup
  clean_up_sequences(mod_stat);

  // put new code
  DB_PUT_MEMORY_RESOURCE(DBR_CODE, module, mod_stat);

  // update/release resources
  reset_current_module_statement();
  reset_current_module_entity();
  /*
  reset_proper_rw_effects();
  reset_ordering_to_statement();
  */

  debug_off();
  return true;
}
