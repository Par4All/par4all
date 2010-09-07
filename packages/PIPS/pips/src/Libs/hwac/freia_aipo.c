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

#include <stdint.h>
#include <stdlib.h>

#include "genC.h"
#include "misc.h"

#include "linear.h"

#include "ri.h"

#include "freia.h"
#include "freia_spoc_private.h"
#include "hwac.h"

void freia_aipo_compile_calls
(string module,
 list /* of statements */ ls,
 hash_table occs,
 int number)
{
  // build DAG for ls
  pips_debug(3, "considering %d statements\n", (int) gen_length(ls));
  pips_assert("some statements", ls);

  list added_stats = NIL;
  dag fulld = build_freia_dag(module, ls, number, occs, &added_stats);

  // ??? append possibly extracted copies?
  // should it be NIL because it is not useful in AIPO->AIPO?
  freia_insert_added_stats(ls, added_stats);

  // cleanup
  free_dag(fulld);
}
