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
#include "properties.h"

#include "freia_spoc_private.h"
#include "hwac.h"

#define FUNC_C "functions.c"

/**************************************************** LOOK FOR IMAGE SHUFFLE */

static bool fis_call_flt(call c, bool * shuffle)
{
  list args = call_arguments(c);
  if (ENTITY_ASSIGN_P(call_function(c)))
  {
    pips_assert("assign takes two args", gen_length(args)==2);
    entity a1 = expression_to_entity(EXPRESSION(CAR(args)));
    entity a2 = expression_to_entity(EXPRESSION(CAR(CDR(args))));
    if (a2!=entity_undefined &&
        freia_image_variable_p(a1) && freia_image_variable_p(a2))
    {
      *shuffle = true;
      gen_recurse_stop(NULL);
      return false;
    }
  }
  return true;
}

static bool freia_image_shuffle(statement s)
{
  bool shuffle = false;
  gen_context_recurse(s, &shuffle, call_domain, fis_call_flt, gen_null);
  return shuffle;
}

/*************************************************************** BASIC UTILS */

/* return malloc'ed "foo.database/Src/%{module}_helper_functions.c"
 * should depend on target? could mix targets?
 */
static string helper_file_name(string func_name)
{
  string src_dir = db_get_directory_name_for_module(func_name);
  string fn = strdup(cat(src_dir, "/", func_name, HELPER FUNC_C));
  free(src_dir);
  return fn;
}

bool freia_skip_op_p(const statement s)
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

static void sort_subsequence(list ls, sequence sq)
{
  set cmp = set_make(set_pointer);
  set_assign_list(cmp, ls);

  // sort ls
  gen_sort_list(ls, (gen_cmp_func_t) freia_cmp_statement);

  // extract sub sequence in a separate list
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

  set_free(cmp);
}

/*********************************************************** LOOK AT EFFECTS */

#include "effects-generic.h"

/* tell whether a statement has no effects on images.
 * if so, it may be included somehow in the pipeline
 * and just skipped, provided stat scalar dependencies
 * are taken care of.
 */
bool freia_some_effects_on_images(statement s)
{
  int img_effect = false;
  list cumu = effects_effects(load_cumulated_rw_effects(s));
  FOREACH(effect, e, cumu)
  {
    if (freia_image_variable_p(effect_variable(e)))
    {
      img_effect = true;
      break;
    }
  }
  return img_effect;
}

/* update the set of written variables, as seen from effects, with a statement
 * this is really to rebuild a poor's man dependency graph in order to move
 * some statements around...
 */
static void update_written_variables(set written, statement s)
{
  list cumu = effects_effects(load_cumulated_rw_effects(s));
  FOREACH(effect, e, cumu)
    if (effect_write_p(e))
      set_add_element(written, written, effect_variable(e));
}

/* @return whether statement depends on some written variables
 */
static bool statement_depends_p(set written, statement s)
{
  bool depends = false;
  list cumu = effects_effects(load_cumulated_rw_effects(s));
  FOREACH(effect, e, cumu)
  {
    if (effect_read_p(e) && set_belong_p(written, effect_variable(e)))
    {
      depends = true;
      break;
    }
  }
  return depends;
}

/* move statements in l ahead of s in sq
 * note that ls is in reverse order...
 */
static void move_ahead(list ls, statement target, sequence sq)
{
  // for faster list inclusion test
  set tomove = set_make(hash_pointer);
  set_append_list(tomove, ls);

  // build new sequence list in reverse order
  list nsq = NIL;
  FOREACH(statement, s, sequence_statements(sq))
  {
    if (set_belong_p(tomove, s))
      continue; // skip!
    if (s==target)
      // insert list now!
      nsq = gen_nconc(gen_copy_seq(ls), nsq);
    // and keep current statement
    nsq = CONS(statement, s, nsq);
  }

  // update sequence with new list
  gen_free_list(sequence_statements(sq));
  sequence_statements(sq) = gen_nreverse(nsq);

  // clean up
  set_free(tomove);
}

/********************************************************* RECURSION HELPERS */

typedef struct {
  list /* of list of statements */ seqs;
} freia_info;

/* detect one lone aipo statement out of a sequence...
 */
static bool fsi_stmt_flt(statement s, freia_info * fsip)
{
  if (freia_statement_aipo_call_p(s))
  {
    instruction i = (instruction) gen_get_ancestor(instruction_domain, s);
    if (i && instruction_sequence_p(i)) return false;
    // not a sequence handled by fsi_seq_flt...
    fsip->seqs = CONS(list, CONS(statement, s, NIL), fsip->seqs);
  }
  return true;
}

/** consider a sequence */
static bool fsi_seq_flt(sequence sq, freia_info * fsip)
{
  pips_debug(9, "considering sequence...\n");

  list /* of statements */ ls = NIL, ltail = NIL, lup = NIL;

  // use a copy, because the sequence is rewritten inside the loop...
  // see sort_subsequence. I guess should rather delay it...
  list lseq = gen_copy_seq(sequence_statements(sq));

  set written = set_make(hash_pointer);

  FOREACH(statement, s, lseq)
  {
    // the statement is kept if it is an AIPO call
    bool freia_api = freia_statement_aipo_call_p(s);
    bool keep_stat = freia_api ||
      // ??? it is an image allocation in the middle of the code...
      // or it has no image effects
      (ls && (freia_skip_op_p(s) || !freia_some_effects_on_images(s)));

    pips_debug(7, "statement %"_intFMT": %skept\n",
               statement_number(s), keep_stat? "": "not ");

    if (keep_stat)
    {
      if (freia_api)
      {
        ls = gen_nconc(ltail, ls), ltail = NIL;
        ls = CONS(statement, s, ls);
        update_written_variables(written, s);
      }
      else // else accumulate in the "other list" waiting for something...
      {
        if (statement_depends_p(written, s) ||
            (statement_call_p(s) &&
             ENTITY_C_RETURN_P(call_function(statement_call(s)))))
        {
          ltail = CONS(statement, s, ltail);
          update_written_variables(written, s);
        }
        else
          // we can move it up...
          lup = CONS(statement, s, lup);
      }
    }
    else // the sequence is cut on this statement
    {
      if (lup && ls)
        move_ahead(lup, STATEMENT(CAR(gen_last(ls))), sq);
      if (lup) gen_free_list(lup), lup = NIL;
      if (ls!=NIL)
      {
        sort_subsequence(ls, sq);
        fsip->seqs = CONS(list, ls, fsip->seqs);
        ls = NIL;
      }
      if (ltail) gen_free_list(ltail), ltail = NIL;
      set_clear(written);
    }
  }

  // end of sequence reached
  if (ls!=NIL)
  {
    if (lup && ls)
      move_ahead(lup, STATEMENT(CAR(gen_last(ls))), sq);
    sort_subsequence(ls, sq);
    fsip->seqs = CONS(list, ls, fsip->seqs);
    ls = NIL;
  }

  // cleanup
  if (ltail) gen_free_list(ltail), ltail = NIL;
  if (lup) gen_free_list(lup), lup = NIL;
  gen_free_list(lseq);
  set_free(written);
  return true;
}

/****************************************** SORT LIST OF EXTRACTED SEQUENCES */

static _int min_statement(list ls)
{
  _int min = -1;
  FOREACH(statement, s, ls)
    min = (min==-1)? statement_number(s):
    (min<statement_number(s)? min: statement_number(s));
  return min;
}

static hash_table fsi_number = NULL;

static int fsi_cmp(const list * l1, const list * l2)
{
  return (int) ((_int) hash_get(fsi_number, *l1) -
                (_int) hash_get(fsi_number, *l2));
}

static void fsi_sort(list lls)
{
  pips_assert("static fsi_number is clean", !fsi_number);
  fsi_number = hash_table_make(hash_pointer, 0);
  FOREACH(list, ls, lls)
    hash_put(fsi_number, ls, (void*) min_statement(ls));
  gen_sort_list(lls, (gen_cmp_func_t) fsi_cmp);
  hash_table_free(fsi_number), fsi_number = NULL;
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

  // check for image shuffles...
  if (freia_image_shuffle(mod_stat))
  {
    if (get_bool_property("FREIA_ALLOW_IMAGE_SHUFFLE"))
      pips_user_warning("image shuffle found in %s, "
                        "freia compilation may result in wrong code!\n",
                        module);
    else
      pips_user_error("image shuffle found in %s, "
                      "see FREIA_ALLOW_IMAGE_SHUFFLE property to continue.\n",
                      module);
  }

  freia_info fsi = { NIL };
  freia_init_dep_cache();

  // collect freia api functions...
  gen_context_multi_recurse(mod_stat, &fsi,
                            sequence_domain, fsi_seq_flt, gen_null,
                            statement_domain, fsi_stmt_flt, gen_null,
                            NULL);
  // sort sequences by increasing statement numbers
  fsi_sort(fsi.seqs);

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
  hash_table occs = freia_build_image_occurrences(mod_stat, NULL);
  set output_images = freia_compute_current_output_images();

  // first explicitely build and fix the list of dags,
  // with some reverse order hocus-pocus for outputs computations...
  list lsi = gen_nreverse(gen_copy_seq(fsi.seqs));
  list ldags = NIL;
  int n_dags = gen_length(fsi.seqs);
  FOREACH(list, ls, lsi)
  {
    dag d = freia_build_dag(module, ls, --n_dags, occs, output_images, ldags);
    ldags = CONS(dag, d, ldags);
  }
  gen_free_list(lsi), lsi = NIL;

  list lcurrent = ldags;
  n_dags = 0;
  bool compile_lone = get_bool_property("FREIA_COMPILE_LONE_OPERATIONS");
  FOREACH(list, ls, fsi.seqs)
  {
    // get corresponding dag, which should be destroyed by the called compiler
    dag d = DAG(CAR(lcurrent));
    lcurrent = CDR(lcurrent);

    // maybe skip lone operations
    if (!compile_lone && gen_length(ls)==1)
    {
      n_dags++;
      gen_free_list(ls);
      continue;
    }

    list allocated = NIL;

    if (freia_spoc_p(target))
      allocated = freia_spoc_compile_calls(module, d, ls, occs, output_images,
                                           helper, n_dags);
    else if (freia_terapix_p(target))
      allocated = freia_trpx_compile_calls(module, d, ls, occs, output_images,
                                           helper, n_dags);
    else if (freia_aipo_p(target))
      allocated = freia_aipo_compile_calls(module, d, ls, occs, n_dags);

    if (allocated)
    {
      FOREACH(entity, img, allocated)
        add_declaration_statement(mod_stat, img);
      gen_free_list(allocated);
    }

    n_dags++;

    // cleanup list contents on the fly
    gen_free_list(ls);
    free_dag(d);
  }

  // cleanup
  freia_clean_image_occurrences(occs);
  freia_close_dep_cache();
  set_free(output_images), output_images = NULL;
  gen_free_list(fsi.seqs);
  gen_free_list(ldags);
  if (helper) safe_fclose(helper, file);

  return file;
}
