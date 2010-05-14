/*

  $Id$

  Copyright 1989-2010 MINES ParisTech

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

#include "gfc2pips-private.h"

#include "c_parser_private.h"
#include "misc.h"
#include "text-util.h"
#include <stdio.h>


list gfc_module_callees = NULL;

/**
 * @brief Add an entity to the list of callees
 *
 */
void gfc2pips_add_to_callees(entity e) {

  if(!intrinsic_entity_p(e) && strcmp_(entity_local_name(e), CurrentPackage)
      != 0) {
    gfc2pips_debug(5, "Add callee : %s\n", entity_local_name( e ) );
    gfc_module_callees = CONS(string,entity_local_name(e),gfc_module_callees);
  }
}

