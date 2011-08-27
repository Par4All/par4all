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
#include "ri-util.h"

#include "pipsdbm.h"
#include "properties.h"

#include "freia.h"
#include "hwac.h"

/* (re)open opencl file
 */
static string get_opencl_file_name(string func_name)
{
  string src_dir = db_get_directory_name_for_module(func_name);
  string file = strdup(cat(src_dir, "/", func_name, "_helper_functions.cl"));
  free(src_dir);
  return file;
}

/* @return whether this vertex is mergeable for OpenCL
 */
static bool opencl_mergeable_p(const dagvtx v)
{
  const freia_api_t * api = dagvtx_freia_api(v);
  bool mergeable = api? api->opencl.mergeable: false;
  pips_debug(7, "%"_intFMT" vertex is %smergeable\n",
             dagvtx_number(v), mergeable? "": "not ");
  return mergeable;
}

/* qsort helper: return -1 for v1 before v2
 */
static int dagvtx_opencl_priority(const dagvtx * pv1, const dagvtx * pv2)
{
  const dagvtx v1 = *pv1, v2 = *pv2;
  bool m1 = opencl_mergeable_p(v1), m2 = opencl_mergeable_p(v2);
  if (m1 && !m2)
    return -1;
  else if (!m1 && m2)
    return 1;
  else
    return dagvtx_number(v1)-dagvtx_number(v2);
}

/* @brief perform opencl compilation on mergeable dag
 */
static int opencl_compile_mergeable_dag(
  string module,
  dag d, list ls,
  string split_name, int n_cut,
  set global_remainings, hash_table signatures,
  FILE * helper, FILE * opencl, set helpers, int stnb)
{
  string cut_name = strdup(cat(split_name, "_", itoa(n_cut)));
  pips_debug(3, "compiling %s cut %d, %d stats\n",
             split_name, n_cut, (int) gen_length(dag_vertices(d)));
  ifdebug(4) dag_dump(stderr, cut_name, d);

  dag_dot_dump(module, cut_name, d, NIL);

  pips_assert("some input or output images", dag_inputs(d) || dag_outputs(d));

  set remainings = set_make(set_pointer);
  set_append_vertex_statements(remainings, dag_vertices(d));

  list lparams = NIL;

  string_buffer
    help = string_buffer_make(true),
    help_body = string_buffer_make(true),
    opcl = string_buffer_make(true),
    opcl_body = string_buffer_make(true),
    opcl_tail = string_buffer_make(true);

  sb_cat(help,
         "\n"
         "// helper function ", cut_name, "\n"
         "freia_status ", cut_name, "(");
  sb_cat(opcl,
         "\n"
         "// opencl function ", cut_name, "\n"
         "kernel ", cut_name, "(");

  int n_outs = gen_length(dag_outputs(d));
  int n_ins = gen_length(dag_inputs(d));
  int nargs = 0;
  int n_params = 0;

  if (n_outs)
    sb_cat(opcl_tail, "    // set output pixels\n");
  int i = 0;
  FOREACH(dagvtx, v, dag_outputs(d))
  {
    sb_cat(help, nargs? ",": "", "\n  " FREIA_IMAGE "o", itoa(i));
    sb_cat(opcl, nargs? ",": "", "\n  " OPENCL_IMAGE "o", itoa(i));
    sb_cat(help_body, nargs? ", ": "", "o", itoa(i));
    // o<out>[i] = t<n>;
    sb_cat(opcl_tail, "    o", itoa(i), "[i] = t");
    sb_cat(opcl_tail, itoa((int) dagvtx_number(v)), ";\n");
    nargs++;
    i++;
  }

  if (n_ins)
    sb_cat(opcl_body, "    // get input pixels\n");
  for (i = 0; i<n_ins; i++)
  {
    sb_cat(help, nargs? ",": "", "\n  const " FREIA_IMAGE "i", itoa(i));
    sb_cat(opcl, nargs? ",": "", "\n  const " OPENCL_IMAGE "i", itoa(i));
    sb_cat(help_body, nargs? ", ": "", "i", itoa(i));
    // pixel in<in> = i<in>[i];
    sb_cat(opcl_body, "    " OPENCL_PIXEL "in", itoa(i), " = i");
    sb_cat(opcl_body, itoa(i), "[i];\n");
    nargs++;
  }
  // other arguments to come...

  // helper call image arguments
  list limg = NIL;
  FOREACH(dagvtx, voa, dag_outputs(d))
    limg = CONS(entity, vtxcontent_out(dagvtx_content(voa)), limg);
  FOREACH(dagvtx, via, dag_inputs(d))
    limg = CONS(entity, vtxcontent_out(dagvtx_content(via)), limg);
  limg = gen_nreverse(limg);

  sb_cat(opcl_body, "    // pixel computations\n");

  // actual computations...
  list vertices = gen_nreverse(gen_copy_seq(dag_vertices(d)));
  FOREACH(dagvtx, v, vertices)
  {
    // skip input nodes
    if (dagvtx_number(v)==0) continue;

    // get details...
    vtxcontent vc = dagvtx_content(v);
    pips_assert("there is a statement",
                pstatement_statement_p(vtxcontent_source(vc)));
    statement s = pstatement_statement(vtxcontent_source(vc));
    call c = freia_statement_to_call(s);
    int opid = dagvtx_opid(v);
    const freia_api_t * api = get_freia_api(opid);
    pips_assert("freia api found", api!=NULL);

    // update for helper call arguments...
    lparams = gen_nconc(lparams,
                        freia_extract_params(opid, call_arguments(c),
                                             help, opcl, NULL, &nargs));

    // pixel t<vertex> = <op>(args...);
    sb_cat(opcl_body,
           "    " OPENCL_PIXEL "t", itoa((int) dagvtx_number(v)),
           " = ", api->opencl.macro, "(");

    // input image arguments
    list preds = dag_vertex_preds(d, v);
    bool nao = 0;
    FOREACH(dagvtx, p, preds)
    {
      if (dagvtx_number(p)==0)
        sb_cat(opcl_body, nao++? ", ": "",
               "in", itoa(gen_position(p, dag_inputs(d))-1));
      else
        sb_cat(opcl_body, nao++? ", ": "", "t", itoa(dagvtx_number(p)));
    }
    gen_free_list(preds), preds = NIL;

    // other (scalar) input arguments
    pips_assert("no output parameters", !api->arg_misc_out);
    for (int i=0; i<(int) api->arg_misc_in; i++)
    {
      sb_cat(help, ",\n  ", api->arg_in_types[i], " c", itoa(n_params));
      sb_cat(opcl, ",\n  ", api->arg_in_types[i], " c", itoa(n_params));
      sb_cat(help_body, ", c", itoa(n_params));
      sb_cat(opcl_body, nao++? ", ": "", "c", itoa(n_params));
      n_params++;
    }

    sb_cat(opcl_body, ");\n");
  }
  gen_free_list(vertices), vertices = NIL;

  // tail
  sb_cat(help, ")\n{\n  call opencl kernel ", cut_name, "(");
  string_buffer_append_sb(help, help_body);
  sb_cat(help, ");\n  return FREIA_OK;\n}\n");

  sb_cat(opcl, ")\n{\n  for (i)\n  {\n");
  string_buffer_append_sb(opcl, opcl_body);
  string_buffer_append_sb(opcl, opcl_tail);
  sb_cat(opcl, "  }\n}\n");

  // cleanup compiled statements
  FOREACH(dagvtx, v, dag_vertices(d))
  {
    pstatement ps = vtxcontent_source(dagvtx_content(v));
    if (pstatement_statement_p(ps))
      hwac_kill_statement(pstatement_statement(ps));
  }

  // handle function image arguments
  freia_add_image_arguments(limg, &lparams);

  // - and substitute its call...
  stnb = freia_substitute_by_helper_call(NULL, global_remainings, remainings,
                                         ls, cut_name, lparams, helpers, stnb);

  hash_put(signatures, local_name_to_top_level_entity(cut_name),
           (void*) (_int) n_outs);

  // actual print
  string_buffer_to_file(help, helper);
  string_buffer_to_file(opcl, opencl);

  // cleanup
  string_buffer_free(&help);
  string_buffer_free(&help_body);
  string_buffer_free(&opcl);
  string_buffer_free(&opcl_body);
  string_buffer_free(&opcl_tail);

  free(cut_name);

  return stnb;
}

/* extract subdags of merged operations and compile them
 * @param d dag to compile, which is destroyed in the process...
 */
static void opencl_merge_and_compile(
  string module,
  list ls,
  dag d,
  string fname_fulldag,
  int n_split,
  const dag fulld,
  const set output_images,
  FILE * helper,
  FILE * opencl,
  set helpers,
  hash_table signatures)
{
  string split_name = strdup(cat(fname_fulldag, "_", itoa(n_split)));
  pips_debug(3, "compiling for %s\n", split_name);

  dag_dot_dump(module, split_name, d, NIL);

  int n_cut = 0;

  set done = set_make(set_pointer);
  set_assign_list(done, dag_inputs(d));
  set current = set_make(set_pointer);
  list lcurrent = NIL;

  set global_remainings = set_make(set_pointer);
  set_assign_list(global_remainings, ls);

  int stnb = -1;

  while (true)
  {
    pips_debug(3, "%s cut %d\n", split_name, n_cut);

    // build homogeneous sub dag
    list computables;
    bool again = true, mergeable = false;
    while (again &&
           (computables = dag_computable_vertices(d, done, done, current)))
    {
      gen_sort_list(computables, (gen_cmp_func_t) dagvtx_opencl_priority);
      again = false;
      FOREACH(dagvtx, v, computables)
      {
        // append only if consistent, necessarily true for the first
        if (dagvtx_number(v)==0) // just eat them
        {
          set_add_element(done, done, v);
          again = true;
        }
        // extract all homogeneous vertices consistent with the current
        else
        {
          if (!lcurrent)
            // first one set the pitch
            mergeable = opencl_mergeable_p(v);
          if (opencl_mergeable_p(v)==mergeable)
          {
            lcurrent = CONS(dagvtx, v, lcurrent);
            set_add_element(done, done, v);
            set_add_element(current, current, v);
            again = true;
          }
        }
      }
      // cleanup
      gen_free_list(computables), computables = NIL;
    }

    // nothing, this is the end
    if (!lcurrent) break;

    // restore vertices order
    lcurrent = gen_nreverse(lcurrent);

    pips_debug(4, "got %d %smergeable vertices\n",
               (int) gen_length(lcurrent), mergeable? "": "non ");

    if (mergeable && gen_length(lcurrent)>1)
    {
      // actually build subdag if something to merge
      dag nd = make_dag(NIL, NIL, NIL);
      FOREACH(dagvtx, v, lcurrent)
        dag_append_vertex(nd, copy_dagvtx_norec(v));
      dag_compute_outputs(nd, NULL, output_images, NIL, false);
      dag_cleanup_other_statements(nd);

      // ??? should not be needed?
      freia_hack_fix_global_ins_outs(fulld, nd);

      // and compile!
      stnb = opencl_compile_mergeable_dag
        (module, nd, ls, split_name, n_cut, global_remainings,
         signatures, helper, opencl, helpers, stnb);
    }

    // cleanup initial dag??
    FOREACH(dagvtx, v, lcurrent)
      dag_remove_vertex(d, v);
    freia_hack_fix_global_ins_outs(fulld, d);

    // cleanup loop state
    gen_free_list(lcurrent), lcurrent = NIL;
    set_clear(current);

    // next!
    n_cut++;
  }

  // cleanup
  set_free(global_remainings);
  set_free(current);
  set_free(done);
  free(split_name);
}

/*
  @brief compile one dag for OPENCL
  @param ls statements underlying the full dag
  @param occs image occurences
  @param exchanges statements to exchange because of dependences
  @param output_images as a surrogate to use-def chains
  @param helper_file output C file for generated code
  @return the list of allocated intermediate images
*/
list freia_opencl_compile_calls
(string module,
 dag fulld,
 list /* of statements */ ls,
 const hash_table occs,
 hash_table exchanges,
 const set output_images,
 FILE * helper_file,
 set helpers,
 int number)
{
  pips_debug(3, "considering %d statements\n", (int) gen_length(ls));
  pips_assert("some statements", ls);

  int n_op_init, n_op_init_copies;
  freia_aipo_count(fulld, &n_op_init, &n_op_init_copies);

  list added_stats = freia_dag_optimize(fulld, exchanges);

  int n_op_opt, n_op_opt_copies;
  freia_aipo_count(fulld, &n_op_opt, &n_op_opt_copies);

  fprintf(helper_file,
          "\n"
          "// dag %d: %d ops and %d copies, "
          "optimized to %d ops and %d+%d copies\n",
          number, n_op_init, n_op_init_copies,
          n_op_opt, n_op_opt_copies, (int) gen_length(added_stats));

  // opencl file
  string opencl_file = get_opencl_file_name(module);
  FILE * opencl;
  if (file_readable_p(opencl_file))
    opencl = safe_fopen(opencl_file, "a");
  else
  {
    opencl = safe_fopen(opencl_file, "w");
    fprintf(opencl, FREIA_OPENCL_INCLUDES
            "// generated opencl for function %s\n", module);
  }
  fprintf(opencl, "\n// opencl for dag %d\n", number);

  // intermediate images
  hash_table init = hash_table_make(hash_pointer, 0);
  list new_images = dag_fix_image_reuse(fulld, init, occs);

  // dump final optimised dag
  dag_dot_dump_prefix(module, "dag_cleaned_", number, fulld, added_stats);

  string fname_fulldag = strdup(cat(module, HELPER, itoa(number)));

  list ld =
    dag_split_on_scalars(fulld, dagvtx_other_stuff_p,
                         (gen_cmp_func_t) dagvtx_opencl_priority,
                         NULL, output_images);

  pips_debug(3, "dag initial split in %d dags\n", (int) gen_length(ld));

  int n_split = 0;

  if (get_bool_property(opencl_merge_prop))
  {
    FOREACH(dag, d, ld)
    {
      if (dag_no_image_operation(d))
        continue;

      opencl_merge_and_compile(module, ls, d, fname_fulldag, n_split,
                               fulld, output_images, helper_file, opencl,
                               helpers, init);

      n_split++;
    }
  }
  // else, do nothing, this is basically like AIPO

  // now may put actual allocations, which messes up statement numbers
  list reals =
    freia_allocate_new_images_if_needed(ls, new_images, occs, init, init);

  freia_insert_added_stats(ls, added_stats);
  added_stats = NIL;

  // cleanup
  gen_free_list(new_images);
  hash_table_free(init);
  safe_fclose(opencl, opencl_file);
  free(opencl_file);

  return reals;
}
