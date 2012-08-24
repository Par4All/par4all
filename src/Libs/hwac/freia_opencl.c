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

/* @brief choose a vertex, avoiding other stuff if the list is started
 */
static dagvtx choose_opencl_vertex(const list lv, bool started)
{
  pips_assert("list contains vertices", lv);
  if (started)
  {
    FOREACH(dagvtx, v, lv)
      if (!dagvtx_other_stuff_p(v))
        return v;
  }
  // just return the first vertex
  return DAGVTX(CAR(lv));
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

  dag_dot_dump(module, cut_name, d, NIL, NIL);

  pips_assert("some input or output images", dag_inputs(d) || dag_outputs(d));

  set remainings = set_make(set_pointer);
  set_append_vertex_statements(remainings, dag_vertices(d));

  list lparams = NIL;

  string_buffer
    // helper function
    helper = string_buffer_make(true),
    helper_decls = string_buffer_make(true),
    helper_body = string_buffer_make(true),
    helper_body_2 = string_buffer_make(true),
    helper_tail = string_buffer_make(true),
    // opencl function
    opencl = string_buffer_make(true),
    opencl_2 = string_buffer_make(true),
    opencl_head = string_buffer_make(true),
    opencl_body = string_buffer_make(true),
    opencl_tail = string_buffer_make(true),
    opencl_end = string_buffer_make(true),
    // compilation function
    compile = string_buffer_make(true);

  // runtime temporary limitation: one image is reduced
  entity reduced = NULL;

  // whether there is a kernel computation in dag
  bool has_kernel = false;

  sb_cat(helper,
         "\n"
         "// helper function ", cut_name, "\n"
         "freia_status ", cut_name, "(");

  sb_cat(helper_decls,
         "  freia_status err = FREIA_OK;\n");

  int n_outs = gen_length(dag_outputs(d)), n_ins = gen_length(dag_inputs(d));

  // ??? what about vol(cst(3))? we could use default bpp/width/height?
  pips_assert("some images to process", n_ins+n_outs);

  sb_cat(helper_decls,
         "\n"
         // hmmm... should be really done at init time. how to do that?
         "  // handle on the fly compilation...\n"
         "  static int to_compile = 1;\n"
         "  if (to_compile) {\n"
         "    err |= ", cut_name, "_compile();\n"
         "    // compilation may have failed\n"
         "    if (err) return err;\n"
         "    to_compile = 0;\n"
         "  }\n"
         "\n"
         "  // now get kernel, which must be have be compiled\n"
         "  uint32_t bpp = ", n_ins? "i0": "o0", "->bpp>>4;\n"
         "  cl_kernel kernel = ", cut_name, "_kernel[bpp];\n");

  sb_cat(opencl,
         "\n"
         "// opencl function ", cut_name, "\n"
         "KERNEL void ", cut_name, "(");

  // count stuff in the generated code
  int nargs = 0, n_params = 0, n_misc = 0, cl_args = 1;

  // output images
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
    sb_cat(opencl_head,
           "  " OPENCL_IMAGE "p", si, " = ", "o", si, " + ofs_o", si, ";\n");
    // , o<out>
    sb_cat(helper_body, ", o", si);
    // p<out>[i] = t<n>;
    sb_cat(opencl_tail,
           "    p", si, "[i] = t", itoa((int) dagvtx_number(v)), ";\n");
    cl_args+=2;
    nargs++;
    free(si);
    i++;
  }

  // input images
  if (n_ins)
    sb_cat(opencl_body, "    // get input pixels\n");
  for (i = 0; i<n_ins; i++)
  {
    string si = strdup(itoa(i));
    sb_cat(helper, nargs? ",": "", "\n  const " FREIA_IMAGE "i", si);
    sb_cat(opencl, nargs? ",": "",
           "\n  " OPENCL_IMAGE "i", si, ", // const?\n  int ofs_i", si);
    // image j<in> = i<in> + ofs_i<out>;
    sb_cat(opencl_head,
           "  " OPENCL_IMAGE "j", si, " = ", "i", si, " + ofs_i", si, ";\n");
    // , i<in>
    sb_cat(helper_body, ", i", si);
    // pixel in<in> = j<in>[i];
    sb_cat(opencl_body,
           "    " OPENCL_PIXEL "in", si, " = j", si, "[i];\n");
    cl_args+=2;
    nargs++;
    free(si);
  }

  // size parameters to handle an image row
  sb_cat(opencl, ",\n  int width,\n  int pitch");
  cl_args+=2;

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

    // vertex v number as a string
    string svn = strdup(itoa((int) dagvtx_number(v)));

    // get details...
    vtxcontent vc = dagvtx_content(v);
    pips_assert("there is a statement",
                pstatement_statement_p(vtxcontent_source(vc)));
    statement s = pstatement_statement(vtxcontent_source(vc));
    call c = freia_statement_to_call(s);
    int opid = dagvtx_opid(v);
    const freia_api_t * api = get_freia_api(opid);
    pips_assert("freia api found", api!=NULL);

    bool is_a_reduction = api->arg_misc_out;
    bool is_a_kernel = api->opencl.mergeable_kernel;

    // update for helper call arguments...
    // kernel operations are specialized, so there is no need to pass it.
    if (!is_a_kernel)
    {
      lparams = gen_nconc(lparams,
        freia_extract_params(opid, call_arguments(c), helper,
                             is_a_reduction? NULL: opencl, NULL, &nargs));
    }
    // input image arguments
    list preds = dag_vertex_preds(d, v);
    int nao = 0;

    // scalar output arguments: we are dealing with a reduction!
    if (is_a_reduction)
    {
      vtxcontent c = dagvtx_content(v);
      list li = vtxcontent_inputs(c);
      pips_assert("one input image", gen_length(li)==1);
      pips_assert("no scalar inputs", !api->arg_misc_in);

      // get the reduced image
      entity img = ENTITY(CAR(li));
      // we deal with only one image at the time...
      // it is a runtime limitation
      pips_assert("same image if any", !reduced ^ (img==reduced));
      if (!reduced)
      {
        reduced = img;
        sb_cat(helper_decls,
               "\n"
               "  // currently only one reduction structure...\n"
               "  freia_opencl_measure_status redres;\n");
        // must be the last argument!
        sb_cat(helper_body_2, ", &redres");
        sb_cat(helper_tail, "\n  // return reduction results\n");
        sb_cat(opencl_2, ",\n  GLOBAL TMeasure * redX");
        // declare them all
        sb_cat(opencl_head,
               "\n"
               "  // reduction stuff is currently hardcoded...\n"
               "  int vol = 0;\n"
               "  int2 mmin = { PIXEL_MAX, 0 };\n"
               "  int2 mmax = { PIXEL_MIN, 0 };\n"
               "  int idy = get_global_id(0);\n");
        sb_cat(opencl_end, "\n  // reduction copy out\n");
        n_misc = 1;
      }
      // inner loop reduction code
      sb_cat(opencl_body, "    ", api->opencl.macro, "(red", svn, ", ");
      dagvtx pred = DAGVTX(CAR(preds));
      int npred = (int) dagvtx_number(pred);
      if (npred==0)
        sb_cat(opencl_body, "in", itoa(gen_position(pred, dag_inputs(d))-1));
      else
        sb_cat(opencl_body, "t", itoa(npred));
      sb_cat(opencl_body, ");\n");

#define RED " = redres." // for a small code compaction

      // tail code to copy back stuff in OpenCL. ??? HARDCODED for now...
      if (same_string_p(api->compact_name, "max"))
      {
        sb_cat(opencl_end,  "  redX[idy].max = mmax.x;\n");
        sb_cat(helper_tail, "  *po", itoa(nargs-1), RED "maximum;\n");
      }
      else if (same_string_p(api->compact_name, "min"))
      {
        sb_cat(opencl_end, "  redX[idy].min = mmin.x;\n");
        sb_cat(helper_tail, "  *po", itoa(nargs-1), RED "minimum;\n");
      }
      else if (same_string_p(api->compact_name, "vol"))
      {
        sb_cat(opencl_end, "  redX[idy].vol = vol;\n");
        sb_cat(helper_tail, "  *po", itoa(nargs-1), RED "volume;\n");
      }
      else if (same_string_p(api->compact_name, "max!"))
      {
        sb_cat(opencl_end,
               "  redX[idy].max = mmax.x;\n"
               "  redX[idy].max_x = (uint) mmax.y;\n"
               "  redX[idy].max_y = idy;\n");
        sb_cat(helper_tail, "  *po", itoa(nargs-3), RED "maximum;\n");
        sb_cat(helper_tail, "  *po", itoa(nargs-2), RED "max_coord_x;\n");
        sb_cat(helper_tail, "  *po", itoa(nargs-1), RED "max_coord_y;\n");
      }
      else if (same_string_p(api->compact_name, "min!"))
      {
        sb_cat(opencl_end,
               "  redX[idy].min = mmin.x;\n"
               "  redX[idy].min_x = (uint) mmin.y;\n"
               "  redX[idy].min_y = idy;\n");
        sb_cat(helper_tail, "  *po", itoa(nargs-3), RED "minimum;\n");
        sb_cat(helper_tail, "  *po", itoa(nargs-2), RED "min_coord_x;\n");
        sb_cat(helper_tail, "  *po", itoa(nargs-1), RED "min_coord_y;\n");
      }
    }
    else if (is_a_kernel)
    {
      // record for adding specific stuff later...
      has_kernel = true;
      pips_assert("one input", gen_length(preds)==1);

      string sin =
        strdup(itoa(gen_position(DAGVTX(CAR(preds)),dag_inputs(d))-1));
      intptr_t k00, k01, k02, k10, k11, k12, k20, k21, k22;
      bool extracted = freia_extract_kernel_vtx(v, true,
                         &k00, &k01, &k02, &k10, &k11, &k12, &k20, &k21, &k22);

      // simplistic hypothesis...
      pips_assert("got kernel", extracted);
      pips_assert("trivial kernel",
           (k00==0 || k00==1) && (k01==0 || k01==1) && (k02==0 || k02==1) &&
           (k10==0 || k10==1) && (k11==0 || k11==1) && (k12==0 || k12==1) &&
           (k20==0 || k20==1) && (k21==0 || k21==1) && (k22==0 || k22==1));

      sb_cat(opencl_body, "    // TODO: handle borders!\n");
      // pixel t<vertex> = <init>;
      sb_cat(opencl_body,
             "    " OPENCL_PIXEL "t", svn, " = ", api->opencl.init, ";\n");
      // t<vertex> = <op>(t<vertex>, j[i+<shift....>]);
      if (k00) sb_cat(opencl_body, "    t", svn, " = ", api->opencl.macro,
                      "(t", svn, ", j", sin, "[i-1-width]);\n");
      if (k01) sb_cat(opencl_body, "    t", svn, " = ", api->opencl.macro,
                      "(t", svn, ", j", sin, "[i-width]);\n");
      if (k02) sb_cat(opencl_body, "    t", svn, " = ", api->opencl.macro,
                      "(t", svn, ", j", sin, "[i+1-width]);\n");
      if (k10) sb_cat(opencl_body, "    t", svn, " = ", api->opencl.macro,
                      "(t", svn, ", j", sin, "[i-1]);\n");
      if (k11) sb_cat(opencl_body, "    t", svn, " = ", api->opencl.macro,
                      "(t", svn, ", in", sin, ";\n");
      if (k12) sb_cat(opencl_body, "    t", svn, " = ", api->opencl.macro,
                      "(t", svn, ", j", sin, "[i+1]);\n");
      if (k20) sb_cat(opencl_body, "    t", svn, " = ", api->opencl.macro,
                      "(t", svn, ", j", sin, "[i-1+width]);\n");
      if (k21) sb_cat(opencl_body, "    t", svn, " = ", api->opencl.macro,
                      "(t", svn, ", j", sin, "[i+width]);\n");
      if (k22) sb_cat(opencl_body, "    t", svn, " = ", api->opencl.macro,
                      "(t", svn, ", j", sin, "[i+1+width]);\n");

      free(sin), sin = NULL;
    }
    else //  we are compiling some pixel operation
    {
      // pixel t<vertex> = <op>(args...);
      sb_cat(opencl_body,
             "    " OPENCL_PIXEL "t", svn, " = ", api->opencl.macro, "(");

      // macro arguments
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
      for (int i=0; i<(int) api->arg_misc_in; i++)
      {
        string sn = strdup(itoa(n_params));
        sb_cat(helper, ",\n  ", api->arg_in_types[i], " c", sn);
        sb_cat(opencl, ",\n  ", opencl_type(api->arg_in_types[i]), " c", sn);
        sb_cat(helper_body, ", c", sn);
        cl_args++;
        sb_cat(opencl_body, nao++? ", ": "", "c", sn);
        free(sn);
        n_params++;
      }

      // end of macro call
      sb_cat(opencl_body, ");\n");
    }

    // cleanup
    free(svn), svn = NULL;
  } // end of FOREACH on operations
  gen_free_list(vertices), vertices = NIL;

  // tail
  sb_cat(helper, ")\n{\n");
  string_buffer_append_sb(helper, helper_decls);
  sb_cat(helper,
         "\n"
         "  // call kernel ", cut_name, "\n",
         "  err |= freia_op_call_kernel(kernel");
  // tell about number of coming kernel parameters
  sb_cat(helper, ", ", itoa(n_outs));   // output images
  sb_cat(helper, ", ", itoa(n_ins));    // input images
  sb_cat(helper, ", ", itoa(n_params)); // input integer parameters
  sb_cat(helper, ", ", itoa(n_misc));   // output integer pointers
  string_buffer_append_sb(helper, helper_body);   // image & param args
  string_buffer_append_sb(helper, helper_body_2); // reduction args
  sb_cat(helper, ");\n");
  string_buffer_append_sb(helper, helper_tail);
  sb_cat(helper, "\n  return err;\n}\n");

  // OPENCL CODE
  string_buffer_append_sb(opencl, opencl_2);
  sb_cat(opencl, ")\n{\n");
  string_buffer_append_sb(opencl, opencl_head);
  sb_cat(opencl,
         // depends on worksize #dims: get_global_id(1)*pitch + g..g..id(0)
         "\n"
         "  // thread's pixel loop\n"
         "  int gid = pitch*get_global_id(0);\n"
         "  int i;\n"
         "  for (i=gid; i < (gid+width); i++)\n"
         "  {\n");
  if (has_kernel) sb_cat(opencl, "    // TODO: border?\n");
  string_buffer_append_sb(opencl, opencl_body);
  string_buffer_append_sb(opencl, opencl_tail);
  sb_cat(opencl, "  }\n");
  string_buffer_append_sb(opencl, opencl_end);
  sb_cat(opencl, "}\n");

  // OpenCL compilation
  sb_cat(compile,
         "\n"
         "// hold kernels for ", cut_name, "\n"
         "static cl_kernel ", cut_name, "_kernel[2];\n"
         "\n"
         "// compile kernels for ", cut_name, "\n"
         "static freia_status ", cut_name, "_compile(void)\n"
         "{\n"
         "  // OpenCL source for ", cut_name, "\n"
         "  const char * ", cut_name, "_source =\n");
  sb_cat(compile, "    \"" FREIA_OPENCL_CL_INCLUDES "\\n\"\n    ");
  string_buffer_append_c_string_buffer(compile, opencl, 4);
  sb_cat(compile,
         ";\n"
         "  freia_status err = FREIA_OK;\n"
         "  err |= freia_op_compile_kernel(", cut_name, "_source, "
                                        "\"", cut_name, "\", \"-DPIXEL8\","
                                        " &", cut_name, "_kernel[0]);\n"
         "  err |= freia_op_compile_kernel(", cut_name, "_source, "
                                        "\"", cut_name, "\", \"-DPIXEL16\","
                                        " &", cut_name, "_kernel[1]);\n"
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
  string_buffer_free(&helper_body_2);
  string_buffer_free(&helper_body);
  string_buffer_free(&helper_tail);
  string_buffer_free(&compile);
  string_buffer_free(&opencl);
  string_buffer_free(&opencl_2);
  string_buffer_free(&opencl_head);
  string_buffer_free(&opencl_body);
  string_buffer_free(&opencl_tail);
  string_buffer_free(&opencl_end);

  free(cut_name);

  return stnb;
}

/* extract subdags of merged operations and compile them
 * @param d dag to compile, which is destroyed in the process...
 */
static void opencl_merge_and_compile(
  string module,
  sequence sq,
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

  dag_dot_dump(module, split_name, d, NIL, NIL);

  int n_cut = 0;

  list // of vertices
    lmergeable = NIL,
    lnonmergeable = NIL;

  set // of vertices
    done = set_make(set_pointer),
    mergeable = set_make(set_pointer),       // consistent with lmergeable
    nonmergeable = set_make(set_pointer); // consistent with lnonmergeable

  set // of statements
    stats = set_make(set_pointer),
    dones = set_make(set_pointer);

  // overall remaining statements to compile
  set global_remainings = set_make(set_pointer);
  set_assign_list(global_remainings, ls);

  // statement number of merged stuff
  int stnb = -1;
  // statement number of last non mergeable vertex
  int max_stnb = -1;

  while (true)
  {
    pips_debug(3, "%s cut %d\n", split_name, n_cut);

    set_clear(done);
    set_assign_list(done, dag_inputs(d));

    // build an homogeneous sub dag
    list computables, initials;
    bool
      merge_reductions = get_bool_property("HWAC_OPENCL_MERGE_REDUCTIONS"),
      merge_kernels = get_bool_property("HWAC_OPENCL_MERGE_KERNEL_OPERATIONS"),
      compile_one_op = get_bool_property("HWAC_OPENCL_COMPILE_ONE_OPERATION");
    // hand manage reductions on one image only
    entity reduced = NULL;

    // we eat up all computable vertices, following dependences
    // we compute first a maximum set of non mergeable vertices
    bool again = true;
    while (again &&
           (computables = dag_computable_vertices(d, done, done, nonmergeable)))
    {
      ifdebug(5) {
        pips_debug(5, "%d computables\n", (int) gen_length(computables));
        gen_fprint(stderr, "computables", computables,
                   (gen_string_func_t) dagvtx_number_str);
      }

      again = false;

      FOREACH(dagvtx, v, computables)
      {
        if (!opencl_mergeable_p(v) ||
            (dagvtx_is_measurement_p(v) && !merge_reductions))
        {
          lnonmergeable = CONS(dagvtx, v, lnonmergeable);
          set_add_element(done, done, v);
          set_add_element(nonmergeable, nonmergeable, v);
          again = true;
        }
      }

      if (again)
        gen_free_list(computables), computables = NIL;
      // else loop exit
    }

    // save up for next phase with mergeables
    initials = computables;

    // then we keep on with extracting mergeable vertices
    again = true;
    while (again &&
           (computables = dag_computable_vertices(d, done, done, mergeable)))
    {
      ifdebug(5) {
        pips_debug(5, "%d computables\n", (int) gen_length(computables));
        gen_fprint(stderr, "computables", computables,
                   (gen_string_func_t) dagvtx_number_str);
      }

      gen_sort_list(computables, (gen_cmp_func_t) dagvtx_opencl_priority);
      again = false;

      FOREACH(dagvtx, v, computables)
      {
        // look for reductions
        // ??? current runtime limitation, there is only ONE image
        // with associated reductions, so the other one are kept out
        if (dagvtx_is_measurement_p(v))
        {
          // a measure is mergeable
          if (merge_reductions)
          {
            list li = vtxcontent_inputs(dagvtx_content(v));
            pips_assert("one input image to reduction", gen_length(li)==1);
            entity image = ENTITY(CAR(li));
            if (reduced && reduced!=image)
            {
              // try to put it with the preceeding non mergeable...
              // if it was computable with them in the previous round
              if (gen_in_list_p(v, initials)) {
                lnonmergeable = CONS(dagvtx, v, lnonmergeable);
                set_add_element(done, done, v);
                set_add_element(nonmergeable, nonmergeable, v);
                again = true;
              }
              // else it will be dealt with later on...
            }
            else  // this is THE reduced image for this  subdag
            {
              reduced = image;
              lmergeable = CONS(dagvtx, v, lmergeable);
              set_add_element(done, done, v);
              set_add_element(mergeable, mergeable, v);
              again = true;
            }
          }
          else // reduction are NOT merged
          {
            // try to put it with the preceeding non mergeable...
            if (gen_in_list_p(v, initials)) {
              lnonmergeable = CONS(dagvtx, v, lnonmergeable);
              set_add_element(done, done, v);
              set_add_element(nonmergeable, nonmergeable, v);
              again = true;
            }
          }
        }
        else if (opencl_mergeable_p(v))
        {
          // not a reduction, mergeable
          lmergeable = CONS(dagvtx, v, lmergeable);
          set_add_element(done, done, v);
          set_add_element(mergeable, mergeable, v);
          again = true;
        }
      }
      gen_free_list(computables), computables = NIL;
    }

    // cleanup list of remaining computables for nonmergeable
    if (initials) gen_free_list(initials), initials = NIL;

    // nothing in both lists, this is the end...
    if (!lmergeable && !lnonmergeable) break;

    // restore vertices order
    lmergeable = gen_nreverse(lmergeable);

    // we try to aggregate some kernel ops to this task...
    if (merge_kernels && lmergeable)
    {
      pips_debug(3, "looking for kernel ops in predecessors...\n");
      list added = NIL;
      FOREACH(dagvtx, v, lmergeable)
      {
        list preds = dag_vertex_preds(d, v);
        pips_debug(4, "%d predecessors to vertex %d\n",
                   (int) gen_length(preds), (int) dagvtx_number(v));
        FOREACH(dagvtx, p, preds)
        {
          if (set_belong_p(nonmergeable, p))
          {
            // it belongs to the previous set
            // we may consider merging it...
            const freia_api_t * api = get_freia_api_vtx(p);
            pips_debug(5, "predecessor is vertex %d (%s)\n",
                       (int) dagvtx_number(p), api->compact_name);
            intptr_t val;
            if (// this is a mergeable kernel
                api->opencl.mergeable_kernel &&
                // all it successors are arithmetic merged
                list_in_set_p(dagvtx_succs(p), mergeable) &&
                // and the kernel must be fully known
                freia_extract_kernel_vtx(p, true, &val, &val, &val, &val,
                                         &val, &val, &val, &val, &val))
            {
              // change status of p
              set_del_element(nonmergeable, nonmergeable, p);
              gen_remove(&lnonmergeable, p);
              added = CONS(dagvtx, p, added);
            }
          }
        }
        // FIX mergeable DELAYED...
        if (added)
        {
          FOREACH(dagvtx, v, added)
            set_add_element(mergeable, mergeable, v);
          // the added are put ahead in the initial order
          lmergeable = gen_nconc(gen_nreverse(added), lmergeable), added = NIL;
        }
        gen_free_list(preds);
      }
    }

    pips_debug(4, "got %d non-mergeables and %d mergeable vertices\n",
               (int) gen_length(lnonmergeable), (int) gen_length(lmergeable));

    if (lnonmergeable)
    {
      // fix statement connexity...
      set_clear(stats);
      FOREACH(dagvtx, v, lnonmergeable)
      {
        // keep track of previous which may have dependencies... hmmm...
        int n = (int) dagvtx_number(v);
        if (n>max_stnb) max_stnb = n;
        // statement to clean
        statement s = dagvtx_statement(v);
        if (s) set_add_element(stats, stats, s);
      }
      freia_migrate_statements(sq, stats, dones);
      set_union(dones, dones, stats);

      // cleanup initial dag??
      FOREACH(dagvtx, v, lnonmergeable)
        dag_remove_vertex(d, v);
      freia_hack_fix_global_ins_outs(fulld, d);

      gen_free_list(lnonmergeable), lnonmergeable = NIL;
      set_clear(nonmergeable);

      n_cut++; // this was a cut, next cut...
    }

    // then mergeables
    if (lmergeable)
    {
      // fix statement connexity
      set_clear(stats);
      FOREACH(dagvtx, v, lmergeable) {
        statement s = dagvtx_statement(v);
        if (s) set_add_element(stats, stats, s);
      }
      freia_migrate_statements(sq, stats, dones);
      set_union(dones, dones, stats);

      if (gen_length(lmergeable)>1 || compile_one_op)
      {
        // actually build subdag if something to merge
        dag nd = make_dag(NIL, NIL, NIL);
        FOREACH(dagvtx, v, lmergeable)
          dag_append_vertex(nd, copy_dagvtx_norec(v));
        dag_compute_outputs(nd, NULL, output_images, NIL, false);
        dag_cleanup_other_statements(nd);

        // ??? should not be needed?
        freia_hack_fix_global_ins_outs(fulld, nd);

        // ??? hack to ensure dependencies...
        if (max_stnb>stnb) stnb = max_stnb;

        // and compile!
        stnb = opencl_compile_mergeable_dag
          (module, nd, ls, split_name, n_cut, global_remainings,
           signatures, helper_file, opencl, helpers, stnb);
      }

      // cleanup initial dag??
      FOREACH(dagvtx, v, lmergeable)
        dag_remove_vertex(d, v);
      freia_hack_fix_global_ins_outs(fulld, d);

      // cleanup loop state
      gen_free_list(lmergeable), lmergeable = NIL;
      set_clear(mergeable);

      n_cut++; // next cut
    }
  }

  // cleanup
  set_free(global_remainings);
  set_free(mergeable);
  set_free(nonmergeable);
  set_free(done);
  set_free(stats);
  set_free(dones);
  free(split_name);
}

/*
  @brief compile one dag for OPENCL
  @param sq containing sequence
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

  int n_op_init, n_op_init_copies;
  freia_aipo_count(fulld, &n_op_init, &n_op_init_copies);

  list added_before = NIL, added_after = NIL;
  freia_dag_optimize(fulld, exchanges, &added_before, &added_after);

  int n_op_opt, n_op_opt_copies;
  freia_aipo_count(fulld, &n_op_opt, &n_op_opt_copies);

  fprintf(helper_file,
          "\n"
          "// dag %d: %d ops and %d copies, "
          "optimized to %d ops and %d+%d+%d copies\n",
          number, n_op_init, n_op_init_copies,
          n_op_opt, n_op_opt_copies,
          (int) gen_length(added_before), (int) gen_length(added_after));

  // opencl file
  string opencl_file = get_opencl_file_name(module);
  FILE * opencl;
  if (file_readable_p(opencl_file))
    opencl = safe_fopen(opencl_file, "a");
  else
  {
    opencl = safe_fopen(opencl_file, "w");
    fprintf(opencl,
            FREIA_OPENCL_CL_INCLUDES "\n"
            "// generated OpenCL kernels for function %s\n", module);
  }
  fprintf(opencl, "\n" "// opencl for dag %d\n", number);

  // intermediate images
  hash_table init = hash_table_make(hash_pointer, 0);
  list new_images = dag_fix_image_reuse(fulld, init, occs);

  // dump final optimised dag
  dag_dot_dump_prefix(module, "dag_cleaned_", number, fulld,
                      added_before, added_after);

  string fname_fulldag = strdup(cat(module, "_opencl", HELPER, itoa(number)));

  list ld =
    dag_split_on_scalars(fulld, dagvtx_other_stuff_p, choose_opencl_vertex,
                         (gen_cmp_func_t) dagvtx_opencl_priority,
                         NULL, output_images);

  pips_debug(3, "dag initial split in %d dags\n", (int) gen_length(ld));

  int n_split = 0;

  if (get_bool_property(opencl_merge_prop))
  {
    set stats = set_make(set_pointer), dones = set_make(set_pointer);
    FOREACH(dag, d, ld)
    {
      if (dag_no_image_operation(d))
        continue;

      // fix statements connexity
      dag_statements(stats, d);
      freia_migrate_statements(sq, stats, dones);
      set_union(dones, dones, stats);

      opencl_merge_and_compile(module, sq, ls, d, fname_fulldag, n_split,
                               fulld, output_images, helper_file, opencl,
                               helpers, init);

      n_split++;
    }
    set_free(stats);
    set_free(dones);
  }
  // else, do nothing, this is basically like AIPO

  // now may put actual allocations, which messes up statement numbers
  list reals =
    freia_allocate_new_images_if_needed(ls, new_images, occs, init, init);

  // hmmm... is this too late?
  freia_insert_added_stats(ls, added_before, true);
  added_before = NIL;
  freia_insert_added_stats(ls, added_after, false);
  added_after = NIL;

  // cleanup
  gen_free_list(new_images);
  hash_table_free(init);
  safe_fclose(opencl, opencl_file);
  free(opencl_file);

  return reals;
}
