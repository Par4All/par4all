/*

  $Id$

  Copyright 1989-2011 MINES ParisTech

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

#include "genC.h"
#include "linear.h"

// newgen
#include "ri.h"
#include "effects.h"

#include "misc.h"
#include "ri-util.h"
#include "effects-util.h"
#include "callgraph.h"
#include "properties.h"

#include "resources.h"
#include "pipsdbm.h"

#include "control.h" // for clean_up_sequences
#include "effects-generic.h" // {set,reset}_proper_rw_effects

#include "freia.h"
#include "hwac.h"

static void freia_unroll_while_for_spoc(statement s)
{
  pips_internal_error("not implemented yet!");
}

int freia_unroll_while(string module)
{
  debug_on("PIPS_HWAC_DEBUG_LEVEL");

  if (!get_bool_property("HWAC_FREIA_SPOC_UNROLL_WHILE"))
    // do nothing... should be prevented from pipsmake?
    return true;

  pips_debug(1, "considering module %s\n", module);

  // else do the stuff
  statement mod_stat =
    (statement) db_get_memory_resource(DBR_CODE, module, true);
  set_current_module_statement(mod_stat);
  set_current_module_entity(module_name_to_entity(module));

  // do the job
  freia_unroll_while_for_spoc(mod_stat);

  // put updated code and accelerated helpers
  DB_PUT_MEMORY_RESOURCE(DBR_CODE, module, mod_stat);

  reset_current_module_statement();
  reset_current_module_entity();
  debug_off();
  return true;
}
