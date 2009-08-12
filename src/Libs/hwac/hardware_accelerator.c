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

#include "ri.h"

#include "properties.h"
#include "misc.h"
#include "ri-util.h"

#include "resources.h"
#include "pipsdbm.h"

#include "hwac.h"

#define HWAC_INPUT "HWAC_INPUT"
#define HWAC_TARGET "HWAC_TARGET"

#define FREIA_API "freia"
#define SPOC_HW "spoc"

static void remove_continues(sequence seq)
{
  list ls = NIL;
  string_buffer comments = string_buffer_make(true);
  FOREACH(STATEMENT, s, sequence_statements(seq))
  {
    if (!continue_statement_p(s))
    {
      if (!string_buffer_empty_p(comments))
      {
	if (!statement_with_empty_comment_p(s))
	{
	  string_buffer_append(comments, statement_comments(s));
	  free(statement_comments(s));
	  statement_comments(s) = NULL;
	}
	statement_comments(s) = string_buffer_to_string(comments);
	string_buffer_reset(comments);
      }
      ls = CONS(STATEMENT, s, ls);
    }
    else // continue statement, removed
    {
      // ??? what about labels?
      // keep comments! attach them to the next available statement in
      // the sequence, otherwise goodbye!
      if (!statement_with_empty_comment_p(s))
	string_buffer_append(comments, statement_comments(s));
      free_statement(s);
    }
  }
  ls = gen_nreverse(ls);
  gen_free_list(sequence_statements(seq));
  string_buffer_free(&comments);
  sequence_statements(seq) = ls;
}

static void cleanup_continues(statement stat)
{
  gen_recurse(stat, sequence_domain, gen_true, remove_continues);
}

int hardware_accelerator(string module)
{
  string input = get_string_property(HWAC_INPUT);
  string target = get_string_property(HWAC_TARGET);

  debug_on("PIPS_HWAC_DEBUG_LEVEL");

  // this will be usefull
  statement mod_stat =
    (statement) db_get_memory_resource(DBR_CODE, module, TRUE);
  set_current_module_statement(mod_stat);
  set_current_module_entity(module_name_to_entity(module));

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

  // mimimal cleanup
  cleanup_continues(mod_stat);

  // put new code
  DB_PUT_MEMORY_RESOURCE(DBR_CODE, module, mod_stat);

  // update/release resources
  reset_current_module_statement();
  reset_current_module_entity();

  debug_off();
  return true;
}
