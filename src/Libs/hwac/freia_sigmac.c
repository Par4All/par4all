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

#include <stdint.h>
#include <stdlib.h>

#include "genC.h"
#include "misc.h"
#include "linear.h"

#include "ri.h"

#include "freia.h"
#include "hwac.h"

#include "freia_sigmac.h"


static int dagvtx_sigmac_priority(const dagvtx * pv1, const dagvtx * pv2)
{
  const dagvtx v1 = *pv1, v2 = *pv2;
  return dagvtx_number(v1)-dagvtx_number(v2);
}

static void sigmac_compile(
  string module,
  dag d,
  string fname,
  int n_split,
  FILE * helper)
{
  fprintf(helper,
          "// code module=%s fname=%s split=%d\n",
          module, fname, n_split);
}

/*
  @brief compile one dag with AIPO optimizations
  @param ls statements underlying the full dag
  @param occs image occurences
  @param exchanges statements to exchange because of dependences
  @return the list of allocated intermediate images
*/
list freia_sigmac_compile_calls
(string module,
 dag fulld,
 sequence sq,
 list ls,
 const hash_table occs,
 hash_table exchanges,
 const set output_images,
 FILE * helper_file,
 set helpers,
 int number)
{
  pips_debug(3, "considering %d statements\n", (int) gen_length(ls));
  pips_assert("some statements", ls);

  // about aipo statistics: no helper file to put them...

  list added_before = NIL, added_after = NIL;
  freia_dag_optimize(fulld, exchanges, &added_before, &added_after);

  // intermediate images
  hash_table init = hash_table_make(hash_pointer, 0);
  list new_images = dag_fix_image_reuse(fulld, init, occs);

  // dump final optimised dag
  dag_dot_dump_prefix(module, "dag_cleaned_", number, fulld,
                      added_before, added_after);

  string fname_fulldag = strdup(cat(module, "_sigmac", HELPER, itoa(number)));

  list ld =
    dag_split_on_scalars(fulld, dagvtx_other_stuff_p, NULL,
                         (gen_cmp_func_t) dagvtx_sigmac_priority,
                         NULL, output_images);

  pips_debug(3, "dag initial split in %d dags\n", (int) gen_length(ld));

  int n_split = 0;

  set stats = set_make(set_pointer), dones = set_make(set_pointer);

  FOREACH(dag, d, ld)
  {
    if (dag_no_image_operation(d))
      continue;

    // fix statements connexity
    dag_statements(stats, d);
    freia_migrate_statements(sq, stats, dones);
    set_union(dones, dones, stats);

    sigmac_compile(module, d, fname_fulldag, n_split, helper_file);

    n_split++;
  }

  set_free(stats);
  set_free(dones);

  // now may put actual allocations, which messes up statement numbers
  list reals =
    freia_allocate_new_images_if_needed(ls, new_images, occs, init, NULL);

  // ??? should it be NIL because it is not useful in AIPO->AIPO?
  freia_insert_added_stats(ls, added_before, true);
  added_before = NIL;
  freia_insert_added_stats(ls, added_after, false);
  added_after = NIL;

  // cleanup
  gen_free_list(new_images);
  hash_table_free(init);

  return reals;
}
