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
#include "freia.h"
#include "freia_spoc.h"
#include "freia_terapix.h"

#include "linear.h"
#include "pipsdbm.h"

#include "ri.h"
#include "effects.h"
#include "ri-util.h"
#include "effects-util.h"

#include "freia_spoc_private.h"
#include "hwac.h"

#define FUNC_C "functions.c"

/*************************************************************** BASIC UTILS */

/* return malloc'ed "foo.database/Src/%{module}_helper_functions.c"
 */
static string helper_file_name(string func_name)
{
  string src_dir = db_get_directory_name_for_module(func_name);
  string fn = strdup(cat(src_dir, "/", func_name, HELPER FUNC_C));
  free(src_dir);
  return fn;
}

static bool freia_skip_op_p(const statement s)
{
  call c = freia_statement_to_call(s);
  const char* called = c? entity_user_name(call_function(c)): "";
  // ??? what about freia_common_check* ?
  return same_string_p(called, FREIA_ALLOC)
    ||   same_string_p(called, FREIA_FREE);
}

/******************************************************** REORDER STATEMENTS */

/* I reorder a little bit statements, so that allocs & deallocs are up
 * front or in the back.
 */

/* order two statements for qsort.
 * s1 before s2 => -1
 */
static int freia_cmp_statement(const statement * s1, const statement * s2)
{
  const call
    c1 = statement_call_p(*s1)?
      instruction_call(statement_instruction((statement) *s1)): NULL,
    c2 = statement_call_p(*s2)?
      instruction_call(statement_instruction((statement) *s2)): NULL;
  bool
    s1r = c1? ENTITY_C_RETURN_P(call_function(c1)): false,
    s2r = c2? ENTITY_C_RETURN_P(call_function(c2)): false,
    s1a = is_freia_alloc(*s1), s2a = is_freia_alloc(*s2),
    s1d = is_freia_dealloc(*s1), s2d = is_freia_dealloc(*s2);
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


// tmp debug
/*
static void stmt_list_nb(string msg, list l)
{
  fprintf(stderr, "! %s: ", msg);
  FOREACH(statement, s, l)
    fprintf(stderr, " %"_intFMT, statement_number(s));
  fprintf(stderr, "\n");
}

static void stmt_seq_nb_flt(sequence sq, string msg)
{
  stmt_list_nb(msg, sequence_statements(sq));
}

static void stmt_seq_nb(string msg)
{
  gen_context_recurse(get_current_module_statement(), msg,
                      sequence_domain, stmt_seq_nb_flt, gen_null);
}
*/

static void sort_subsequence(list ls, sequence sq)
{
  set cmp = set_make(set_pointer);
  set_assign_list(cmp, ls);
  // stmt_list_nb("ls", ls);
  // stmt_seq_nb("before");

  // sort ls
  gen_sort_list(ls, (gen_cmp_func_t) freia_cmp_statement);

  // extract sub sequence
  list head = NIL, lcmp = NIL, tail = NIL;
  FOREACH(statement, s, sequence_statements(sq))
  {
    if (set_belong_p(cmp, s))
      lcmp = CONS(statement, s, lcmp);
    else
      if (!lcmp)
        head = CONS(statement, s, head);
      else
        tail = CONS(statement, s, tail);
  }

  // sort subsequence
  gen_sort_list(lcmp, (gen_cmp_func_t) freia_cmp_statement);

  // rebuild sequence
  gen_free_list(sequence_statements(sq));
  sequence_statements(sq) =
    gen_nconc(gen_nreverse(head), gen_nconc(lcmp, gen_nreverse(tail)));

  // stmt_seq_nb("after");
  set_free(cmp);
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

/********************************************************* RECURSION HELPERS */

typedef struct {
  list /* of list of statements */ seqs;
} freia_info;

/** consider a sequence */
static bool sequence_flt(sequence sq, freia_info * fsip)
{
  pips_debug(9, "considering sequence...\n");

  list /* of statements */ ls = NIL, ltail = NIL;

  // use a copy, because the sequence is rewritten inside the loop...
  // see sort_subsequence. I guess should rather delay it...
  list lseq = gen_copy_seq(sequence_statements(sq));

  FOREACH(statement, s, lseq)
  {
    // the statement is kept if it is an AIPO call
    bool freia_api = freia_statement_aipo_call_p(s);
    bool keep_stat = freia_api ||
      // ??? it is an image allocation in the middle of the code...
      // or it has no image effects
      (ls && (freia_skip_op_p(s) || !some_effects_on_images(s)));

    pips_debug(7, "statement %"_intFMT": %skept\n",
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
    else // the sequence is cut on this statement
    {
      if (ls!=NIL)
      {
        sort_subsequence(ls, sq);
        fsip->seqs = CONS(list, ls, fsip->seqs);
        ls = NIL;
      }
      if (ltail) gen_free_list(ltail), ltail = NIL;
    }
  }

  // end of sequence reached
  if (ls!=NIL)
  {
    sort_subsequence(ls, sq);
    fsip->seqs = CONS(list, ls, fsip->seqs);
    ls = NIL;
  }

  if (ltail) gen_free_list(ltail), ltail = NIL;
  gen_free_list(lseq);
  return true;
}

/**************************************************************** DO THE JOB */

/* freia_compile:
 * compile freia module & statement for target
 * - collect sequences of freia operations
 * - create file for generated functions
 * - call hardware-specific code generation for each sequence
 */
string freia_compile(string module, statement mod_stat, string target)
{
  if (!freia_valid_target_p(target))
    pips_internal_error("unexpected target %s", target);

  freia_info fsi;
  fsi.seqs = NIL;
  freia_init_dep_cache();

  // collect freia api functions...
  if (statement_call_p(mod_stat))
  {
    // argh, there is only one statement in the code, and no sequence
    if (freia_statement_aipo_call_p(mod_stat))
      fsi.seqs = CONS(list, CONS(statement, mod_stat, NIL), NIL);
  }
  else // look for sequences
  {
    gen_context_recurse(mod_stat, &fsi,
                        sequence_domain, sequence_flt, gen_null);
  }

  // output file if any
  string file = NULL;
  FILE * helper = NULL;
  if (freia_spoc_p(target) || freia_terapix_p(target))
  {
    file = helper_file_name(module);
    if (file_readable_p(file))
      pips_user_error("file '%s' already exists: "
                      "cannot reapply transformation\n", file);
    helper = safe_fopen(file, "w");
    pips_debug(1, "generating file '%s'\n", file);
  }

  // headers
  if (freia_spoc_p(target))
    fprintf(helper, FREIA_SPOC_INCLUDES);
  else if (freia_terapix_p(target))
    fprintf(helper, FREIA_TRPX_INCLUDES);

  // hmmm...
  hash_table occs = freia_build_image_occurrences(mod_stat);

  int n_dags = 0;
  FOREACH(list, ls, fsi.seqs)
  {
    list allocated = NIL;

    if (freia_spoc_p(target))
      allocated = freia_spoc_compile_calls(module, ls, occs, helper, n_dags);
    else if (freia_terapix_p(target))
      allocated = freia_trpx_compile_calls(module, ls, occs, helper, n_dags);
    else if (freia_aipo_p(target))
      freia_aipo_compile_calls(module, ls, occs, n_dags);
    gen_free_list(ls);

    if (allocated)
    {
      FOREACH(entity, img, allocated)
        add_declaration_statement(mod_stat, img);
      gen_free_list(allocated);
    }

    n_dags++;
  }

  freia_clean_image_occurrences(occs);
  freia_close_dep_cache();

  // cleanup
  gen_free_list(fsi.seqs);
  if (helper) safe_fclose(helper, file);

  return file;
}
