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

/* @return OpenCL helper file name for function
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

/* @return opencl type for freia type
 */
static string opencl_type(string t)
{
  if (same_string_p(t, "int32_t")) return "int";
  pips_internal_error("unexpected type: %s\n", t);
  return NULL;
}

/* @brief perform OpenCL compilation on mergeable dag
   the generated code relies on some freia-provided runtime
*/
static int opencl_compile_mergeable_dag(
  string module,
  dag d, list ls,
  string split_name, int n_cut,
  set global_remainings, hash_table signatures,
  FILE * helper_file, FILE * opencl_file, set helpers, int stnb)
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
    helper = string_buffer_make(true),
    helper_decls = string_buffer_make(true),
    helper_body = string_buffer_make(true),
    opencl = string_buffer_make(true),
    opencl_init = string_buffer_make(true),
    opencl_body = string_buffer_make(true),
    opencl_tail = string_buffer_make(true),
    compile = string_buffer_make(true);

  sb_cat(helper,
         "\n"
         "// helper function ", cut_name, "\n"
         "freia_status ", cut_name, "(");

  sb_cat(helper_decls,
         "  freia_status err = FREIA_OK;\n"
         "  freia_data2d_opencl *pool= frclTarget.pool;\n");

  int n_outs = gen_length(dag_outputs(d)), n_ins = gen_length(dag_inputs(d));

  // ??? hey, what about vol(cst(3))?
  pips_assert("some images to process", n_ins+n_outs);

  // set pitch and other stuff... we need an image!
  string ref = "i0";
  if (n_outs) ref = "o0";
  sb_cat(helper_decls,
         "  int pitch = ", ref, "->row[1] - ", ref, "->row[0];\n"
         "  size_t workSize[2];\n" // why 2?
         "  workSize[0] = ", ref, "->heightWa;\n"
         "  uint32_t bpp = ", ref, "->bpp>>4;\n"
         "  cl_kernel kernel,\n"
         "\n"
         "  // handle on the fly compilation...\n"
         "  static int to_compile = 1;\n"
         "\n"
         "  if (to_compile) {\n"
         "    freia_status cerr = ", cut_name, "_compile();\n"
         "    // compilation may have failed\n"
         "    if (cerr) return cerr;\n"
         "    to_compile = 0;\n"
         "  }\n"
         "\n"
         "  // now get kernel, which must be there...\n"
         "  kernel = ", cut_name, "_kernel[bpp];\n");

  sb_cat(opencl,
         "\n"
         "// opencl function ", cut_name, "\n"
         "KERNEL void ", cut_name, "(");

  // count stuff in the generated code
  int nargs = 0, n_params = 0, cl_args = 0;

  if (n_outs)
    sb_cat(opencl_tail, "    // set output pixels\n");
  int i = 0;
  FOREACH(dagvtx, v, dag_outputs(d))
  {
    string si = strdup(itoa(i));
    sb_cat(helper, nargs? ",": "", "\n  " FREIA_IMAGE "o", si);
    sb_cat(opencl, nargs? ",": "",
           "\n  " OPENCL_IMAGE "o", si,
           ",\n  int ofs_o", si);
    // image p<out> = o<out> + ofs_o<out>;
    sb_cat(opencl_init,
           "  " OPENCL_IMAGE "p", si, " = ", "o", si, " + ofs_o", si, ";\n");
    // sb_cat(helper_body, nargs? ", ": "", "o", itoa(i));
    sb_cat(helper_body,
           "  err |= clSetKernelArg(kernel, ", itoa(cl_args),
           ", sizeof(cl_mem), &pool[o", si, "->clId]);\n");
    cl_args++;
    sb_cat(helper_body,
           "  err |= clSetKernelArg(kernel, ", itoa(cl_args),
           ", sizeof(cl_int), &ofs_o", si, ");\n");
    cl_args++;
    // p<out>[i] = t<n>;
    sb_cat(opencl_tail,
           "    p", si, "[i] = t", itoa((int) dagvtx_number(v)), ";\n");
    nargs++;
    free(si);
    i++;
  }

  if (n_ins)
    sb_cat(opencl_body, "    // get input pixels\n");
  for (i = 0; i<n_ins; i++)
  {
    string si = strdup(itoa(i));
    sb_cat(helper, nargs? ",": "", "\n  const " FREIA_IMAGE "i", si);
    sb_cat(opencl, nargs? ",": "",
           "\n  " OPENCL_IMAGE "i", si, ", // const?\n  int ofs_i", si);
    // image j<in> = i<in> + ofs_i<out>;
    sb_cat(opencl_init,
           "  " OPENCL_IMAGE "j", si, " = ", "i", si, " + ofs_i", si, ";\n");
    // sb_cat(helper_body, nargs? ", ": "", "i", si);
    sb_cat(helper_body,
           "  err |= clSetKernelArg(kernel, ", itoa(cl_args),
           ", sizeof(cl_mem), &pool[i", si, "->clId]);\n");
    cl_args++;
    sb_cat(helper_body,
           "  err |= clSetKernelArg(kernel, ", itoa(cl_args),
           ", sizeof(cl_int), &ofs_i", si, ");\n");
    cl_args++;
    // pixel in<in> = j<in>[i];
    sb_cat(opencl_body,
           "    " OPENCL_PIXEL "in", si, " = j", si, "[i];\n");
    nargs++;
    free(si);
  }

  // size parameters to handle an image row
  sb_cat(opencl, cl_args? ",": "", "\n  int width,\n  int pitch");
  sb_cat(helper_body,
         "  err |= clSetKernelArg(kernel, ", itoa(cl_args),
         ", sizeof(cl_int), &width);\n");
  cl_args++;
  sb_cat(helper_body,
         "  err |= clSetKernelArg(kernel, ", itoa(cl_args),
         ", sizeof(cl_int), &pitch);\n");
  cl_args++;

  // there are possibly other kernel arguments yet to come...

  // helper call image arguments
  list limg = NIL;
  FOREACH(dagvtx, voa, dag_outputs(d))
    limg = CONS(entity, vtxcontent_out(dagvtx_content(voa)), limg);
  FOREACH(dagvtx, via, dag_inputs(d))
    limg = CONS(entity, vtxcontent_out(dagvtx_content(via)), limg);
  limg = gen_nreverse(limg);

  sb_cat(opencl_body, "    // pixel computations\n");

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
                                             helper, opencl, NULL, &nargs));

    // pixel t<vertex> = <op>(args...);
    sb_cat(opencl_body,
           "    " OPENCL_PIXEL "t", itoa((int) dagvtx_number(v)),
           " = ", api->opencl.macro, "(");

    // input image arguments
    list preds = dag_vertex_preds(d, v);
    bool nao = 0;
    FOREACH(dagvtx, p, preds)
    {
      if (dagvtx_number(p)==0)
        sb_cat(opencl_body, nao++? ", ": "",
               "in", itoa(gen_position(p, dag_inputs(d))-1));
      else
        sb_cat(opencl_body, nao++? ", ": "", "t", itoa(dagvtx_number(p)));
    }
    gen_free_list(preds), preds = NIL;

    // other (scalar) input arguments
    pips_assert("no output parameters", !api->arg_misc_out);
    for (int i=0; i<(int) api->arg_misc_in; i++)
    {
      string sn = strdup(itoa(n_params));
      sb_cat(helper, ",\n  ", api->arg_in_types[i], " c", sn);
      sb_cat(opencl, ",\n  ", opencl_type(api->arg_in_types[i]), " c", sn);
      //sb_cat(helper_body, ", c", sn);
      sb_cat(helper_body,
           "  err |= clSetKernelArg(kernel, ", itoa(cl_args),
           ", sizeof(cl_int), &c", sn, ");\n");
      cl_args++;
      sb_cat(opencl_body, nao++? ", ": "", "c", sn);
      free(sn);
      n_params++;
    }

    sb_cat(opencl_body, ");\n");
  }
  gen_free_list(vertices), vertices = NIL;

  // tail
  sb_cat(helper, ")\n{\n");
  string_buffer_append_sb(helper, helper_decls);
  sb_cat(helper,
         "\n"
         "  // set kernel parameters\n");
  string_buffer_append_sb(helper, helper_body);
  sb_cat(helper,
         "\n"
         "  // call kernel ", cut_name, "\n",
         "  err |= clEnqueueNDRangeKernel(\n"
         "            frclTarget.queue,\n"
         "            kernel,\n"
         "            1, // number of dimensions\n"
         "            NULL, // undefined for OpenCL 1.0\n"
         "            workSize, // x and y dimensions\n"
         "            NULL, // local size\n"
         "            0, // don't wait events\n"
         "            NULL,\n"
         "            NULL); // do not produce an event\n"
         "\n"
         "  return err;\n}\n");

  sb_cat(opencl, ")\n{\n");
  string_buffer_append_sb(opencl, opencl_init);
  sb_cat(opencl,
         "  int gid = get_global_id(1)*pitch + get_global_id(0);\n"
         "  int i;\n"
         "  for (i=gid; i < (gid+width); i++)\n  {\n");
  string_buffer_append_sb(opencl, opencl_body);
  string_buffer_append_sb(opencl, opencl_tail);
  sb_cat(opencl, "  }\n}\n");

  // OpenCL compilation
  sb_cat(compile,
         "\n"
         "// hold kernels for", cut_name, "\n"
         "static cl_kernel ", cut_name, "_kernel[2];\n"
         "\n"
         "// compile kernels for ", cut_name, "\n"
         "static freia_status ", cut_name, "_compile(void)\n"
         "{\n"
         "  // OpenCL source for ", cut_name, "\n"
         "  const char * ", cut_name, "_source = \"\n",
         FREIA_OPENCL_CL_INCLUDES);
  string_buffer_append_sb(compile, opencl);
  sb_cat(compile,
         "\";\n"
         "  freia_status err = FREIA_OK;\n"
         "  err |= get_compiled_opencl(", cut_name, "_source, \"",
                              cut_name, "\", \"-DPIXEL8\", &",
                              cut_name, "_kernel[0]);\n"
         "  err |= get_compiled_opencl(", cut_name, "_source, \"",
                              cut_name, "\", \"-DPIXEL16\", &",
                              cut_name, "_kernel[1]);\n"
         "  return err;\n"
         "}\n");

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

  // actual printing...
  string_buffer_to_file(compile, helper_file);
  string_buffer_to_file(helper, helper_file);
  string_buffer_to_file(opencl, opencl_file);

  // cleanup
  string_buffer_free(&helper);
  string_buffer_free(&helper_decls);
  string_buffer_free(&helper_body);
  string_buffer_free(&compile);
  string_buffer_free(&opencl);
  string_buffer_free(&opencl_init);
  string_buffer_free(&opencl_body);
  string_buffer_free(&opencl_tail);

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
  FILE * helper_file,
  FILE * opencl,
  set helpers,
  hash_table signatures)
{
  string split_name = strdup(cat(fname_fulldag, "_", itoa(n_split)));
  pips_debug(3, "compiling for %s\n", split_name);

  dag_dot_dump(module, split_name, d, NIL);

  int n_cut = 0;

  set done = set_make(set_pointer);
  set current = set_make(set_pointer);
  list lcurrent = NIL;

  set global_remainings = set_make(set_pointer);
  set_assign_list(global_remainings, ls);

  int stnb = -1;

  while (true)
  {
    pips_debug(3, "%s cut %d\n", split_name, n_cut);

    set_clear(done);
    set_assign_list(done, dag_inputs(d));

    // build an homogeneous sub dag
    list computables;
    bool again = true, mergeable = true;
    while (again &&
           (computables = dag_computable_vertices(d, done, done, current)))
    {
      gen_sort_list(computables, (gen_cmp_func_t) dagvtx_opencl_priority);
      again = false;
      // first, try to extract non mergeable nodes ahead
      if (!lcurrent)
      {
        FOREACH(dagvtx, v, computables)
        {
          if (!opencl_mergeable_p(v))
          {
            lcurrent = CONS(dagvtx, v, lcurrent);
            set_add_element(done, done, v);
            set_add_element(current, current, v);
            again = true;
            mergeable = false;
          }
        }
      }
      // then accumulate current status
      if (!again)
      {
        FOREACH(dagvtx, v, computables)
        {
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
         signatures, helper_file, opencl, helpers, stnb);
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
    fprintf(opencl,
            FREIA_OPENCL_CL_INCLUDES
            "// generated OpenCL kernels for function %s\n", module);
  }
  fprintf(opencl, "\n" "// opencl for dag %d\n", number);

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
