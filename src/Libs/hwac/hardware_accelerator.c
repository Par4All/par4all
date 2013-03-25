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
// needs some arg_label & vertex_label types
// #include "graph.h"

#include "misc.h"
#include "ri-util.h"
#include "effects-util.h"
#include "callgraph.h"

#include "resources.h"
#include "pipsdbm.h"

#include "control.h" // for clean_up_sequences
#include "effects-generic.h" // {set,reset}_proper_rw_effects

#include "freia.h"
#include "hwac.h"

static string commenter(__attribute__((unused)) entity e)
{
  return strdup(" hardware accelerator generated variable");
  //return string_undefined; // no comment?
}

/************************************************************ FREIA COMPILERS */

/* compile FREIA calls for some target.
 */
static bool freia_compiler(const string module, const string hardware)
{
  debug_on("PIPS_HWAC_DEBUG_LEVEL");
  pips_debug(1, "considering module %s for hardware %s\n", module, hardware);

  // this will be usefull
  statement mod_stat =
    (statement) db_get_memory_resource(DBR_CODE, module, true);
  set_current_module_statement(mod_stat);
  set_current_module_entity(module_name_to_entity(module));
  // should be pure?
  set_cumulated_rw_effects((statement_effects)
      db_get_memory_resource(DBR_CUMULATED_EFFECTS, module, false));
  // useless... but...
  set_proper_rw_effects((statement_effects)
      db_get_memory_resource(DBR_PROPER_EFFECTS, module, false));
  push_generated_variable_commenter(commenter);

  pips_debug(1, "considering module %s\n", module);

  // accelerated code generation
  string freia_file = freia_compile(module, mod_stat, hardware);

  // some code cleanup
  clean_up_sequences(mod_stat);

  // put updated code and accelerated helpers
  DB_PUT_MEMORY_RESOURCE(DBR_CODE, module, mod_stat);

  // update callees
  DB_PUT_MEMORY_RESOURCE(DBR_CALLEES, module,
                         (void*) compute_callees(mod_stat));

  // other possibly generated resources
  // Argh, as soon as it is put, it is obsolete. Indeed, the code is updated,
  // and the file depends on the code and resources derived from the code.
  if (freia_spoc_p(hardware))
    DB_PUT_NEW_FILE_RESOURCE(DBR_SPOC_FILE, module, freia_file);
  else if (freia_terapix_p(hardware))
    DB_PUT_NEW_FILE_RESOURCE(DBR_TERAPIX_FILE, module, freia_file);
  else if (freia_opencl_p(hardware))
    DB_PUT_NEW_FILE_RESOURCE(DBR_OPENCL_FILE, module, freia_file);
  else if (freia_sigmac_p(hardware))
    DB_PUT_NEW_FILE_RESOURCE(DBR_SIGMAC_FILE, module, freia_file);
  // else no helper file for AIPO target, which is just a code transformation

  // release resources
  // ??? free statement_effects? MEMORY LEAK...
  // but some statements contents where freed
  // there may be some sharing between * effects & statements?
  pop_generated_variable_commenter();
  reset_cumulated_rw_effects();
  reset_proper_rw_effects();
  reset_current_module_statement();
  reset_current_module_entity();

  debug_off();
  return true;
}

bool freia_spoc_compiler(const string module)
{
  return freia_compiler(module, "spoc");
}

bool freia_terapix_compiler(const string module)
{
  return freia_compiler(module, "terapix");
}

bool freia_aipo_compiler(const string module)
{
  return freia_compiler(module, "aipo");
}

bool freia_opencl_compiler(const string module)
{
  return freia_compiler(module, "opencl");
}

bool freia_sigmac_compiler(const string module)
{
  return freia_compiler(module, "sigmac");
}

/* int freia_vhdl_compiler(string module) :-)
 */
