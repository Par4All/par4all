/*

  $Id$

  Copyright 1989-2014 MINES ParisTech
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
#include "pipsdbm.h"
#include "text.h"
#include "text-util.h"
#include "prettyprint.h"

#include "pointer_values.h"

text text_pv(entity __attribute__ ((unused)) module, int __attribute__ ((unused)) margin, statement s)
{
  list lpv = cell_relations_list(load_pv(s));

  return(text_pointer_values(lpv, "Pointer values:"));
}

bool generic_print_code_pv(char * module_name, pv_context * ctxt)
{
  bool success;

  set_current_module_statement( (statement)
				db_get_memory_resource(DBR_CODE, module_name, true));
  set_current_module_entity(module_name_to_entity(module_name));

  set_pv((*ctxt->db_get_pv_func)(module_name));

  list l_in = (*ctxt->db_get_in_pv_func)(module_name);
  list l_out = (*ctxt->db_get_in_pv_func)(module_name);

  init_prettyprint(text_pv);
  text t = make_text(NIL);
  MERGE_TEXTS(t, text_pointer_values(l_in, "IN Pointer values:"));
  MERGE_TEXTS(t, text_pointer_values(l_out, "OUT Pointer values:"));
  MERGE_TEXTS(t, text_module(get_current_module_entity(),
			     get_current_module_statement()));
  success = make_text_resource_and_free(module_name, DBR_PRINTED_FILE, ".pv", t);
  close_prettyprint();

  reset_current_module_entity();
  reset_current_module_statement();
  reset_pv();
  return success;
}


bool print_code_simple_pointer_values(char * module_name)
{
  pv_context ctxt = make_simple_pv_context();
  bool success = generic_print_code_pv(module_name, &ctxt);
  reset_pv_context(&ctxt);
  return success;
}

void generic_print_code_gen_kill_pv(char * module_name)
{
}

bool print_code_simple_gen_kill_pointer_values(char * module_name)
{
  pv_context ctxt = make_simple_pv_context();
  generic_print_code_gen_kill_pv(module_name);
  reset_pv_context(&ctxt);
  return(true);
}
