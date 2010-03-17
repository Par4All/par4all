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

#include <stdint.h>
#include <stdlib.h>

#include "genC.h"
#include "misc.h"
#include "freia_spoc.h"

#include "linear.h"
#include "pipsdbm.h"

#include "ri.h"
#include "ri-util.h"
#include "properties.h"

#include "freia_spoc_private.h"
#include "hwac.h"

#define cat concatenate

#define FUNC_C "functions.c"

/* return malloc'ed "foo.database/Src/%{module}_helper_functions.c"
 */
static string helper_file_name(string func_name)
{
  string src_dir = db_get_directory_name_for_module(func_name);
  string fn = strdup(cat(src_dir, "/", func_name, HELPER FUNC_C, NULL));
  free(src_dir);
  return fn;
}

static bool freia_skip_op_p(const statement s)
{
  call c = freia_statement_to_call(s);
  string called = c? entity_user_name(call_function(c)): "";
  // ??? what about freia_common_check* ?
  return same_string_p(called, "freia_common_create_data")
    ||   same_string_p(called, "freia_common_destruct_data");
}

static bool is_alloc(const statement s)
{
  call c = freia_statement_to_call(s);
  string called = c? entity_user_name(call_function(c)): "";
  return same_string_p(called, "freia_common_create_data");
}

static bool is_dealloc(const statement s)
{
  call c = freia_statement_to_call(s);
  string called = c? entity_user_name(call_function(c)): "";
  return same_string_p(called, "freia_common_destruct_data");
}

/* I reorder a little bit statements, so that allocs & deallocs are up
 * front or in the back.
 */
static set cmp_subset = NULL;
/* order two statements for qsort.
 * s1 before s2 => -1
 */
static int freia_cmp_statement(const statement * s1, const statement * s2)
{
  pips_assert("some subset of statements to reorder...", cmp_subset);
  if (!set_belong_p(cmp_subset, *s1) || !set_belong_p(cmp_subset, *s2))
    return statement_number(*s1) - statement_number(*s2);

  // else we have to do something
  const call
    c1 = statement_call_p(*s1)?
      instruction_call(statement_instruction((statement) *s1)): NULL,
    c2 = statement_call_p(*s2)?
      instruction_call(statement_instruction((statement) *s2)): NULL;
  bool
    s1r = c1? ENTITY_C_RETURN_P(call_function(c1)): false,
    s2r = c2? ENTITY_C_RETURN_P(call_function(c2)): false,
    s1a = is_alloc(*s1), s2a = is_alloc(*s2),
    s1d = is_dealloc(*s1), s2d = is_dealloc(*s2);
  if (s1r || s2r) pips_assert("one return in sequence", s1r ^ s2r);

  pips_debug(9, "%"_intFMT" %s is %d %d %d\n", statement_number(*s1),
	     c1? entity_name(call_function(c1)): "", s1r, s1a, s1d);
  pips_debug(9, "%"_intFMT" %s is %d %d %d\n", statement_number(*s2),
	     c2? entity_name(call_function(c2)): "", s2r, s2a, s2d);

  int order = 0;
  string why = "";

  // return at the back, obviously...
  if (s1r) order = 1, why = "return1";
  else if (s2r) order = -1, why = "return2";
  // allocs at the front (there may be in initialisations, which up front)
  else if (s1a && !s2a) order = -1, why = "alloc1";
  else if (s2a && !s1a) order = 1, why = "alloc2";
  // deallocs at the back
  else if (s1d && !s2d) order = 1, why = "free1";
  else if (s2d && !s1d) order = -1, why = "free2";
  // else keep statement initial order
  else order = statement_number(*s1)-statement_number(*s2), why = "stat";

  pips_assert("total order", order!=0);

  pips_debug(7, "%"_intFMT" %s %"_intFMT" (%s)\n", statement_number(*s1),
	     order==-1? "<": ">", statement_number(*s2), why);

  return order;
}

#include "effects-generic.h"

/* tell whether a statement has no effects on images.
 * if so, it may be included somehow in the pipeline
 * and just skipped, provided stat scalar dependencies
 * are taken care of.
 */
static bool some_effects_on_images(statement s)
{
  int img_effect = false;
  list cumu = effects_effects(load_cumulated_rw_effects(s));
  FOREACH(effect, e, cumu) {
    if (freia_image_variable_p(effect_variable(e))) {
      img_effect = true;
      break;
    }
  }
  return img_effect;
}

typedef struct {
  list /* of list of statements */ seqs;
} freia_info;

/** consider a sequence */
static bool sequence_flt(sequence sq, freia_info * fsip)
{
  pips_debug(9, "considering sequence...\n");
  cmp_subset = set_make(set_pointer);

  list /* of statements */ ls = NIL, ltail = NIL;
  FOREACH(statement, s, sequence_statements(sq))
  {
    // the statement is kept if it is an AIPO call
    bool freia_api = freia_statement_aipo_call_p(s);
    bool keep_stat = freia_api ||
      // ??? it is an image allocation in the middle of the code...
      // or it has no image effects
      (ls && (freia_skip_op_p(s) || !some_effects_on_images(s)));

    pips_debug(7, "statement %"_intFMT": %skeeped\n",
	       statement_number(s), keep_stat? "": "not ");

    if (keep_stat)
    {
      if (freia_api)
      {
	ls = gen_nconc(ltail, ls), ltail = NIL;
	ls = CONS(statement, s, ls);
      }
      else // else accumulate in the "other list" waiting for something...
      {
	ltail = CONS(statement, s, ltail);
      }
    }
    else
      if (ls!=NIL) {
	set_assign_list(cmp_subset, ls);
	gen_sort_list(sequence_statements(sq),
		      (gen_cmp_func_t) freia_cmp_statement);
	ls = gen_nreverse(ls);
	fsip->seqs = CONS(list, ls, fsip->seqs);
	ls = NIL;
      }
  }

  // end of sequence reached
  if (ls!=NIL) {
    set_assign_list(cmp_subset, ls);
    gen_sort_list(sequence_statements(sq),
		  (gen_cmp_func_t) freia_cmp_statement);
    ls = gen_nreverse(ls);
    fsip->seqs = CONS(list, ls, fsip->seqs);
    ls = NIL;
  }

  if (ltail) gen_free_list(ltail);
  set_free(cmp_subset), cmp_subset = NULL;
  return true;
}

string freia_compile(string module, statement mod_stat)
{
  freia_info fsi;
  fsi.seqs = NIL;

  // collect freia api functions...
  if (statement_call_p(mod_stat))
  {
    // argh, there is only one statemnt in the code, and no sequence
    if (freia_statement_aipo_call_p(mod_stat))
      fsi.seqs = CONS(list, CONS(statement, mod_stat, NIL), NIL);
  }
  else // look for sequences
    gen_context_recurse(mod_stat, &fsi,
			sequence_domain, sequence_flt, gen_null);

  // output file
  string file = helper_file_name(module);
  if (file_readable_p(file))
    pips_user_error("file '%s' already here, cannot reapply transformation\n");

  pips_debug(1, "generating file '%s'\n", file);
  FILE * helper = safe_fopen(file, "w");
  fprintf(helper, FREIA_SPOC_INCLUDES);

  // generate stuff
  int n_dags = 0;
  FOREACH(list, ls, fsi.seqs)
  {
    freia_spoc_compile_calls(module, ls, helper, n_dags++);
    gen_free_list(ls);
  }

  // cleanup
  gen_free_list(fsi.seqs);
  safe_fclose(helper, file);

  return file;
}
