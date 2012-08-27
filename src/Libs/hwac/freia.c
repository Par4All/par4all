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

/************************************************************ CLEANUP STATUS */

typedef struct {
  // entities: variables used to hold freia status returns
  set used_for_status;
  // entities: set of generated helper functions, may be NULL
  set helper_functions;
} fcs_ctx;

static bool fcs_call_flt(call c, fcs_ctx * ctx)
{
  entity called = call_function(c);
  if (freia_assignment_p(called))
  {
    list largs = call_arguments(c);
    pips_assert("two arguments to = or |=", gen_length(largs)==2);
    expression erhs = EXPRESSION(CAR(CDR(largs)));

    // catch and eat-up "foo |= 0;"
    _int val;
    if (ENTITY_BITWISE_OR_UPDATE_P(called) &&
        expression_integer_value(erhs, &val) && val==0)
    {
      call_function(c) = entity_intrinsic(CONTINUE_FUNCTION_NAME);
      gen_free_list(call_arguments(c)), call_arguments(c) = NIL;
      return false;
    }

    // else, is it a call to some aipo or helper function?
    syntax op = expression_syntax(erhs);
    if (!syntax_call_p(op))
      return false;

    // is it an aipo or helper call?
    call rhs = syntax_call(op);
    if (entity_freia_api_p(call_function(rhs)) ||
        (ctx->helper_functions &&
         set_belong_p(ctx->helper_functions, call_function(rhs))))
    {
      // record scrapped status variable
      syntax slhs = expression_syntax(EXPRESSION(CAR(largs)));
      pips_assert("left hand side is a scalar reference",
                  syntax_reference_p(slhs) &&
                  !reference_indices(syntax_reference(slhs)));
      set_add_element(ctx->used_for_status, ctx->used_for_status,
                      reference_variable(syntax_reference(slhs)));

      // and scrap its assignement!
      call_function(c) = call_function(rhs);
      call_arguments(c) = call_arguments(rhs);

      // hmmm... small memory leak in assigned lhs,
      // but I'm not sure of effect references...
      call_arguments(rhs) = NIL;
      free_call(rhs);
      gen_free_list(largs);
    }
  }
  // no second level anyway? what about ","?
  return false;
}

/*
  foo |= freia_aipo_* or helpers  -> freia_aipo_* or helpers
  foo = freia_aipo_*  or helpers  -> freia_aipo_* or helpers
  foo |= 0                        -> <nope>
  declaration: foo = 0 (FREIA_OK), if it is not initialized.
  ??? not managed: freia_aipo_ calls in declarations, return, ","?
*/
static void freia_cleanup_status(statement s, set helpers)
{
  fcs_ctx ctx = { set_make(set_pointer), helpers };

  // cleanup statements
  gen_context_recurse(s, &ctx, call_domain, fcs_call_flt, gen_null);

  // fix status variable initializations
  SET_FOREACH(entity, var, ctx.used_for_status)
  {
    if (value_unknown_p(entity_initial(var)))
    {
      free_value(entity_initial(var));
      entity_initial(var) =
        make_value_expression(call_to_expression(freia_ok()));
    }
  }

  // cleanup
  set_free(ctx.used_for_status);
}

/**************************************************** LOOK FOR IMAGE SHUFFLE */

typedef struct {
  set shuffled;
} fis_ctx;

static bool fis_call_flt(call c, fis_ctx * ctx)
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
      set_add_element(ctx->shuffled, ctx->shuffled, a1);
      set_add_element(ctx->shuffled, ctx->shuffled, a2);
      return false;
    }
  }
  return true;
}

/* @return whether there is an image shuffle, i.e. image pointer assignments
 */
static bool freia_image_shuffle(statement s, set shuffled)
{
  fis_ctx ctx = { shuffled };
  gen_context_recurse(s, &ctx, call_domain, fis_call_flt, gen_null);
  return set_size(shuffled)!=0;
}

/******************************************************* SWITCH CAST TO COPY */

static void sctc_call_rwt(call c, int * count)
{
  entity func = call_function(c);
  if (same_string_p(entity_local_name(func), AIPO "cast"))
  {
    call_function(c) = local_name_to_top_level_entity(AIPO "copy");
    (*count)++;
  }
}

/* @brief switch all image casts to image copies in "s"
 */
static void switch_cast_to_copy(statement s)
{
  int count = 0;
  gen_context_recurse(s, &count, call_domain, gen_true, sctc_call_rwt);
  // ??? I may have look into declarations? freia calls in declarations
  // are not really implemented yet.
  pips_user_warning("freia_cast switched to freia_copy: %d\n", count);
}


/*************************************************************** BASIC UTILS */

/* return malloc'ed "foo.database/Src/%{module}_helper_functions.c"
 * should it depend on the target? no, because we could mix targets?
 */
static string helper_file_name(string func_name)
{
  string src_dir = db_get_directory_name_for_module(func_name);
  string fn = strdup(cat(src_dir, "/", func_name, HELPER FUNC_C));
  free(src_dir);
  return fn;
}

static bool freia_stmt_free_p(const statement s)
{
  call c = freia_statement_to_call(s);
  const char* called = c? entity_user_name(call_function(c)): "";
  return same_string_p(called, FREIA_FREE);
}

bool freia_skip_op_p(const statement s)
{
  call c = freia_statement_to_call(s);
  const char* called = c? entity_user_name(call_function(c)): "";
  // ??? what about freia_common_check* ?
  return same_string_p(called, FREIA_ALLOC)
    ||   same_string_p(called, FREIA_FREE);
}

/* move statements in l ahead of s in sq
 * note that ls is in reverse order...
 */
static void move_ahead(list ls, statement target, sequence sq)
{
  // for faster list inclusion test
  set tomove = set_make(set_pointer);
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

/******************************************************** REORDER STATEMENTS */

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

/* debug helper */
static string stmt_nb(void * s)
{
  return itoa((int) statement_number((statement) s));
}

/* remove non aipo statements at the head of ls
   @return updated list
*/
static list clean_list_head(list ls)
{
  list head = ls, prev = NIL;
  while (ls && !freia_statement_aipo_call_p(STATEMENT(CAR(ls))))
    prev = ls, ls = CDR(ls);
  if (prev) CDR(prev) = NIL, gen_free_list(head);
  return ls;
}

/* remove unrelated stuff at head & tail, so that it does not come in later on
   and possibly trouble the compilation.
   @return updated list
*/
static list clean_statement_list_for_aipo(list ls)
{
  return gen_nreverse(clean_list_head(gen_nreverse(clean_list_head(ls))));
}

/* reorder a little bit statements, so that allocs & deallocs are up
 * front or in the back, and may be removed by the previous function.
 */
static void sort_subsequence(list ls, sequence sq)
{
  ifdebug(8) {
    pips_debug(8, "input: ");
    gen_fprint(stderr, "ls", ls, (gen_string_func_t) stmt_nb);
  }

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

  ifdebug(8) {
    pips_debug(8, "output: ");
    gen_fprint(stderr, "ls", ls, (gen_string_func_t) stmt_nb);
  }
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
  ifdebug(9)
    set_fprint(stderr, "written", written,
               (gen_string_func_t) entity_local_name);
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

/************************************ RECURSION HELPERS TO EXTRACT SEQUENCES */

typedef struct {
  // list of list of statements extracted from sequences
  list seqs;
   // list of statement (from seqs) -> its owner sequence if any
  hash_table sequence;
  // count kept while recursing
  int enclosing_loops;
  // elements in the list encountered in potential inclosing loops,
  // which impacts recomputed use-def dependencies that can be loop-carried.
  set in_loops;
} freia_info;

static bool fsi_loop_flt( __attribute__((unused)) gen_chunk * o,
                          freia_info * fsip)
{
  fsip->enclosing_loops++;
  return true;
}

static void fsi_loop_rwt( __attribute__((unused)) gen_chunk * o,
                          freia_info * fsip)
{
  fsip->enclosing_loops--;
}

/* detect one lone aipo statement out of a sequence...
 */
static bool fsi_stmt_flt(statement s, freia_info * fsip)
{
  if (freia_statement_aipo_call_p(s))
  {
    instruction i = (instruction) gen_get_ancestor(instruction_domain, s);
    if (i && instruction_sequence_p(i)) return false;
    // not a sequence handled by fsi_seq_flt...
    list ls = CONS(statement, s, NIL);
    fsip->seqs = CONS(list, ls, fsip->seqs);
    if (fsip->enclosing_loops)
      set_add_element(fsip->in_loops, fsip->in_loops, ls);
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

  set written = set_make(set_pointer);

  FOREACH(statement, s, lseq)
  {
    // the statement is kept if it is an AIPO call
    bool freia_api = freia_statement_aipo_call_p(s);
    bool keep_stat = freia_api ||
      // ??? it is an image allocation in the middle of the code...
      // or it has no image effects
      (ls && (freia_skip_op_p(s) || !freia_some_effects_on_images(s)));

    // quick fix for performance: stop on timer calls
    if (keep_stat && statement_call_p(s))
    {
      entity called = call_function(statement_call(s));
      if (same_string_p(entity_local_name(called), "gettimeofday") ||
          same_string_p(entity_local_name(called), "freia_common_wait"))
        keep_stat = false;
    }

    pips_debug(7, "statement %"_intFMT": %skept\n",
               statement_number(s), keep_stat? "": "not ");

    // FREIA API & "intermediate" statements are kept
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
             ENTITY_C_RETURN_P(call_function(statement_call(s)))) ||
            // free are kept in the tail... not caught as a W effect?
            freia_stmt_free_p(s))
        {
          pips_debug(8, "statement %d in ltail\n", (int) statement_number(s));
          ltail = CONS(statement, s, ltail);
          update_written_variables(written, s);
        }
        else
        {
          pips_debug(8, "statement %d in lup\n", (int) statement_number(s));
          // we can move it up...
          lup = CONS(statement, s, lup);
        }
      }
    }
    else // the sequence must be cut on this statement
    {
      if (lup && ls)
        move_ahead(lup, STATEMENT(CAR(gen_last(ls))), sq);
      if (lup) gen_free_list(lup), lup = NIL;
      if (ls!=NIL)
      {
        sort_subsequence(ls, sq);
        ls = clean_statement_list_for_aipo(ls);
        fsip->seqs = CONS(list, ls, fsip->seqs);
        hash_put(fsip->sequence, ls, sq);
        if (fsip->enclosing_loops)
          set_add_element(fsip->in_loops, fsip->in_loops, ls);
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
    ls = clean_statement_list_for_aipo(ls);
    fsip->seqs = CONS(list, ls, fsip->seqs);
    hash_put(fsip->sequence, ls, sq);
    if (fsip->enclosing_loops)
      set_add_element(fsip->in_loops, fsip->in_loops, ls);
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

/********************************************************** SEQUENCE CLEANUP */
/*
  some more cleanup added to the main sequence.
  some existing passes could do a better job, but they would require
  working helper effects and managing pointers. Moreover, I know that I can
  ignore deallocations.
*/

static void collect_images(reference r, set referenced)
{
  entity var = reference_variable(r);
  if (freia_image_variable_p(var) && !set_belong_p(referenced, var))
    set_add_element(referenced, referenced, var);
}

static bool freia_cleanup_sequence_rec(statement modstat, set referenced)
{
  //fprintf(stderr, "[freia_cleanup_main_sequence] %p\n", modstat);
  bool changed = false;
  instruction i = statement_instruction(modstat);

  if (instruction_sequence_p(i))
  {
    sequence seq = instruction_sequence(i);
    list rstats = gen_nreverse(gen_copy_seq(sequence_statements(seq)));
    // set of referenced images

    FOREACH(statement, s, rstats)
    {
      // handle seqs in seqs with an explicit recursion
      if (statement_sequence_p(s))
        changed = freia_cleanup_sequence_rec(s, referenced);
      else if (freia_statement_aipo_call_p(s))
      {
        call c = freia_statement_to_call(s);
        const freia_api_t * api =
          hwac_freia_api(entity_local_name(call_function(c)));
        if (api->arg_img_out==1)
        {
          entity img = expression_to_entity(EXPRESSION(CAR(call_arguments(c))));
          //fprintf(stderr, "considering %s on %s\n",
          //        api->function_name, entity_local_name(img));
          if (!set_belong_p(referenced, img) &&
              // hey, we keep written parameters!
              // ??? what about global images, if any?
              !formal_parameter_p(img))
          {
            changed = true;
            free_instruction(statement_instruction(s));
            statement_instruction(s) = make_continue_instruction();
          }
        }
      }
      // update seen variables, but must skip deallocations!
      // ??? what about skipping alloc as well?
      if (!is_freia_dealloc(s))
        gen_context_recurse(s, referenced,
                            // could also do loop indexes, but no images
                            reference_domain, gen_true, collect_images);
      // also look in initializations
      FOREACH(entity, var, statement_declarations(s))
        gen_context_recurse(entity_initial(var), referenced,
                            reference_domain, gen_true, collect_images);

    }

    gen_free_list(rstats);
  }
  return changed;
}

static bool freia_cleanup_main_sequence(statement modstat)
{
  set referenced = set_make(hash_pointer);
  bool changed = freia_cleanup_sequence_rec(modstat, referenced);
  set_free(referenced);
  return changed;
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
  pips_assert("we need some dependable effects for our purpose",
              !get_bool_property("CONSTANT_PATH_EFFECTS"));

  if (!freia_valid_target_p(target))
    pips_internal_error("unexpected target %s", target);

  // check for image shuffles...
  set shuffled = set_make(set_pointer);
  if (freia_image_shuffle(mod_stat, shuffled))
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

  // freia_aipo_cast -> freia_aipo_copy before further compilation
  if (get_bool_property("FREIA_CAST_IS_COPY"))
    switch_cast_to_copy(mod_stat);

  freia_info fsi =
    { NIL, hash_table_make(hash_pointer, 0), 0, set_make(set_pointer) };
  freia_init_dep_cache();

  // collect freia api functions...
  gen_context_multi_recurse(mod_stat, &fsi,
                            // collect sequences
                            sequence_domain, fsi_seq_flt, gen_null,
                            statement_domain, fsi_stmt_flt, gen_null,
                            // just count potential enclosing loops...
                            loop_domain, fsi_loop_flt, fsi_loop_rwt,
                            whileloop_domain, fsi_loop_flt, fsi_loop_rwt,
                            forloop_domain, fsi_loop_flt, fsi_loop_rwt,
                            unstructured_domain, fsi_loop_flt, fsi_loop_rwt,
                            NULL);

  // check safe return
  pips_assert("loop count back to zero", fsi.enclosing_loops==0);

  // sort sequences by increasing statement numbers
  fsi_sort(fsi.seqs);

  // output file if any
  string file = NULL;
  set helpers = NULL;
  FILE * helper = NULL;
  if (freia_spoc_p(target) || freia_terapix_p(target) || freia_opencl_p(target))
  {
    file = helper_file_name(module);
    if (file_readable_p(file))
      pips_user_error("file '%s' already exists: "
                      "cannot reapply transformation\n", file);
    helper = safe_fopen(file, "w");
    pips_debug(1, "generating file '%s'\n", file);

    // we will also record created helpers
    helpers = set_make(set_pointer);
  }

  // headers
  if (freia_spoc_p(target))
    fprintf(helper, "%s", FREIA_SPOC_INCLUDES);
  else if (freia_terapix_p(target))
    fprintf(helper, "%s", FREIA_TRPX_INCLUDES);
  else if (freia_opencl_p(target))
    fprintf(helper, "%s", FREIA_OPENCL_INCLUDES);

  // hmmm... should rely on use-defs
  hash_table occs = freia_build_image_occurrences(mod_stat, NULL, NULL, NULL);
  set output_images = freia_compute_current_output_images();
  // hmmm... panic mode
  set_union(output_images, output_images, shuffled);

  // first explicitely build and fix the list of dags,
  // with some reverse order hocus-pocus for outputs computations...
  list lsi = gen_nreverse(gen_copy_seq(fsi.seqs));
  list ldags = NIL;
  int n_dags = gen_length(fsi.seqs);
  FOREACH(list, ls, lsi)
  {
    dag d = freia_build_dag(module, ls, --n_dags, occs, output_images, ldags,
                            set_belong_p(fsi.in_loops, ls));
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
    hash_table exchanges = NULL;
    sequence sq = NULL;
    if (hash_defined_p(fsi.sequence, ls))
    {
      exchanges = hash_table_make(hash_pointer, 0);
      sq = hash_get(fsi.sequence, ls);
    }

    if (freia_spoc_p(target))
      allocated = freia_spoc_compile_calls(module, d, sq, ls, occs, exchanges,
                                       output_images, helper, helpers, n_dags);
    else if (freia_terapix_p(target))
      allocated = freia_trpx_compile_calls(module, d, sq, ls, occs, exchanges,
                                       output_images, helper, helpers, n_dags);
    else if (freia_opencl_p(target))
      allocated = freia_opencl_compile_calls(module, d, sq, ls, occs, exchanges,
                                       output_images, helper, helpers, n_dags);
    else if (freia_aipo_p(target))
      allocated = freia_aipo_compile_calls(module, d, ls, occs, exchanges,
                                           n_dags);

    if (exchanges)
    {
      list lssq = sequence_statements((sequence) hash_get(fsi.sequence, ls));
      HASH_FOREACH(dagvtx, v1, dagvtx, v2, exchanges)
        gen_exchange_in_list(lssq, dagvtx_statement(v1), dagvtx_statement(v2));
    }

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
    if (exchanges) hash_table_free(exchanges);
  }

  // some code cleanup
  freia_cleanup_main_sequence(mod_stat);

  // some more code cleanup
  if (get_bool_property("FREIA_CLEANUP_STATUS"))
    // remove err = or err |= freia & helper functions
    freia_cleanup_status(mod_stat, helpers);

  // cleanup
  freia_clean_image_occurrences(occs);
  freia_close_dep_cache();
  set_free(output_images), output_images = NULL;
  set_free(shuffled), shuffled = NULL;
  if (helpers) set_free(helpers), helpers = NULL;
  gen_free_list(fsi.seqs);
  set_free(fsi.in_loops);
  hash_table_free(fsi.sequence);
  gen_free_list(ldags);
  if (helper) safe_fclose(helper, file);

  return file;
}
