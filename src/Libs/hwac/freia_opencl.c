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
   may be called on a one vertex dag for kernel operations
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

  // I could handle a closed dag, such as volume(cst(12))...
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
  bool has_kernel = false,
    need_N = false, need_S = false, need_W = false, need_E = false;

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

  // get worksize identifiers...
  sb_cat(opencl_head,
         "  // get id & compute global image shift\n"
         // here we assume a 1D worksize
         "  int threadid = get_global_id(0);\n"
         // depends on worksize #dims: get_global_id(1)*pitch + g..g..id(0)
         "  int shift = pitch*threadid;\n"
         "\n"
         "  // get input & output image pointers\n");

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
    // image p<out> = o<out> + ofs_o<out> + shift;
    sb_cat(opencl_head,
      "  " OPENCL_IMAGE "p", si, " = ", "o", si, " + ofs_o", si, " + shift;\n");
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
    // image j<in> = i<in> + ofs_i<out> + shift;
    sb_cat(opencl_head,
      "  " OPENCL_IMAGE "j", si, " = ", "i", si, " + ofs_i", si, " + shift;\n");
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
  sb_cat(opencl, ",\n"
         "  int width, // of the working area, vs image below\n"
         "  int pitch");
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
               "  int2 mmax = { PIXEL_MIN, 0 };\n");
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
        sb_cat(opencl_end,  "  redX[threadid].max = mmax.x;\n");
        sb_cat(helper_tail, "  *po", itoa(nargs-1), RED "maximum;\n");
      }
      else if (same_string_p(api->compact_name, "min"))
      {
        sb_cat(opencl_end, "  redX[threadid].min = mmin.x;\n");
        sb_cat(helper_tail, "  *po", itoa(nargs-1), RED "minimum;\n");
      }
      else if (same_string_p(api->compact_name, "vol"))
      {
        sb_cat(opencl_end, "  redX[threadid].vol = vol;\n");
        sb_cat(helper_tail, "  *po", itoa(nargs-1), RED "volume;\n");
      }
      else if (same_string_p(api->compact_name, "max!"))
      {
        sb_cat(opencl_end,
               "  redX[threadid].max = mmax.x;\n"
               "  redX[threadid].max_x = (uint) mmax.y;\n"
               "  redX[threadid].max_y = threadid;\n");
        sb_cat(helper_tail, "  *po", itoa(nargs-3), RED "maximum;\n");
        sb_cat(helper_tail, "  *po", itoa(nargs-2), RED "max_coord_x;\n");
        sb_cat(helper_tail, "  *po", itoa(nargs-1), RED "max_coord_y;\n");
      }
      else if (same_string_p(api->compact_name, "min!"))
      {
        sb_cat(opencl_end,
               "  redX[threadid].min = mmin.x;\n"
               "  redX[threadid].min_x = (uint) mmin.y;\n"
               "  redX[threadid].min_y = threadid;\n");
        sb_cat(helper_tail, "  *po", itoa(nargs-3), RED "minimum;\n");
        sb_cat(helper_tail, "  *po", itoa(nargs-2), RED "min_coord_x;\n");
        sb_cat(helper_tail, "  *po", itoa(nargs-1), RED "min_coord_y;\n");
      }
    }
    else if (is_a_kernel)
    {
      // NOTE about ILP: many intra sequence dependencies are generated...
      // - for 2 full morpho ops on the same input with few threads,
      //   5% perf won by loading all values, then intermixing operations.
      // - moving gards out does not change anything much wrt "?:" ops
      //   e.g. "if (is_X|is_Y) { op1 op2 }

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

      need_N = need_N || k00 || k01 || k02;
      need_W = need_W || k00 || k10 || k20;
      need_E = need_E || k02 || k12 || k22;
      need_S = need_S || k20 || k21 || k22;

      // pixel t<vertex> = <init>;
      sb_cat(opencl_body,
             "    " OPENCL_PIXEL "t", svn, " = ", api->opencl.init, ";\n");
      // t<vertex> = <op>(t<vertex>, boundary?init:j[i+<shift....>]);
      if (k00) sb_cat(opencl_body, "    t", svn, " = ", api->opencl.macro,
                      "(t", svn, ", (is_N|is_W)? ", api->opencl.init,
                      ": j", sin, "[i-pitch-1]);\n");
      if (k01) sb_cat(opencl_body, "    t", svn, " = ", api->opencl.macro,
                      "(t", svn, ", (is_N)? ", api->opencl.init,
                      ": j", sin, "[i-pitch]);\n");
      if (k02) sb_cat(opencl_body, "    t", svn, " = ", api->opencl.macro,
                      "(t", svn, ", (is_N|is_E)? ", api->opencl.init,
                      ": j", sin, "[i-pitch+1]);\n");
      if (k10) sb_cat(opencl_body, "    t", svn, " = ", api->opencl.macro,
                      "(t", svn, ", (is_W)? ", api->opencl.init,
                      ": j", sin, "[i-1]);\n");
      // most likely to be non null
      if (k11) sb_cat(opencl_body, "    t", svn, " = ", api->opencl.macro,
                      "(t", svn, ", in", sin, ");\n");
      if (k12) sb_cat(opencl_body, "    t", svn, " = ", api->opencl.macro,
                      "(t", svn, ", (is_E)? ", api->opencl.init,
                      ": j", sin, "[i+1]);\n");
      if (k20) sb_cat(opencl_body, "    t", svn, " = ", api->opencl.macro,
                      "(t", svn, ", (is_W|is_S)? ", api->opencl.init,
                      ": j", sin, "[i+pitch-1]);\n");
      if (k21) sb_cat(opencl_body, "    t", svn, " = ", api->opencl.macro,
                      "(t", svn, ", (is_S)? ", api->opencl.init,
                      ": j", sin, "[i+pitch]);\n");
      if (k22) sb_cat(opencl_body, "    t", svn, " = ", api->opencl.macro,
                      "(t", svn, ", (is_E|is_S)? ", api->opencl.init,
                      ": j", sin, "[i+pitch+1]);\n");

      free(sin), sin = NULL;
    }
    else //  we are compiling an arithmetic pixel operation
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
  if (get_bool_property("HWAC_OPENCL_SYNCHRONIZE_KERNELS"))
    sb_cat(helper, "\n"
           "  // synchronize...\n"
           "  freia_common_wait();\n");
  sb_cat(helper, "\n"
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
  if (has_kernel) {
    sb_cat(opencl,
           "\n"
           "  // detect N & S boundaries (assuming one thread per row)\n");
    if (need_N)
      sb_cat(opencl, "  int is_N = (threadid==0);\n");
    else
      sb_cat(opencl, "  // N not needed\n");
    if (need_S)
      sb_cat(opencl, "  int is_S = (threadid==(get_global_size(0)-1));\n");
    else sb_cat(opencl, "  // S not needed\n");
  }
  sb_cat(opencl,
         "\n"
         "  // thread's pixel loop\n"
         "  int i;\n"
         "  for (i=0; i<width; i++)\n"
         "  {\n");
  if (has_kernel) {
    sb_cat(opencl,
           "    // detect W & E boundaries (assuming one row per thread)\n");
    if (need_W)
      sb_cat(opencl, "    int is_W = (i==0);\n");
    else
      sb_cat(opencl, "    // W not needed\n");
    if (need_E)
      sb_cat(opencl, "    int is_E = (i==(width-1));\n");
    else
      sb_cat(opencl, "    // E not needed\n");
  }
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

  if (ls)
  {
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
  }
  // else it is not subtituded

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

/* call and generate if necessary a specialized kernel, if possible
 * the statement is bluntly modified "in place".
 * @return whether a substition was performed
 */
static void opencl_generate_special_kernel_ops(
  string module, dagvtx v, hash_table signatures,
  FILE * helper_file, FILE * opencl_file, set helpers)
{
  pips_debug(4, "considering statement %"_intFMT"\n", dagvtx_number(v));
  const freia_api_t * api = get_freia_api_vtx(v);
  intptr_t k00, k01, k02, k10, k11, k12, k20, k21, k22;
  if (!api ||
      !api->opencl.mergeable_kernel ||
      !freia_extract_kernel_vtx(v, true, &k00, &k01, &k02,
                                &k10, &k11, &k12, &k20, &k21, &k22))
    // do nothing
    return;

  // build an id number for this specialized version
  int number = (k00<<8) + (k01<<7) + (k02<<6) +
               (k10<<5) + (k11<<4) + (k12<<3) +
               (k20<<2) + (k21<<1) + k22;
  pips_assert("non-zero kernel!", number);

  // main_opencl_helper_<E8,D8,con>
  string prefix = strdup(cat(module, "_opencl_helper_", api->compact_name));
  // main_opencl_helper_E8_<0..511>
  string func_name = strdup(cat(prefix, "_", itoa(number)));

  entity specialized = local_name_to_top_level_entity(func_name);
  if (specialized == entity_undefined)
  {
    pips_debug(5, "generating %s\n", func_name);

    // we need to create the function
    dag one = make_dag(NIL, NIL, NIL);
    dagvtx vop = copy_dagvtx_norec(v);
    dag_append_vertex(one, vop);
    dag_outputs(one) = CONS(dagvtx, vop, NIL);
    //dag_compute_outputs(one, NULL, output_images, NIL, false);
    //dag_cleanup_other_statements(nd);

    // we just generate the function
    opencl_compile_mergeable_dag(module, one, NIL, prefix, number, NULL, NULL,
                                 helper_file, opencl_file, NULL, 0);

    // then create it... function name must be consistent with
    // what is done within previous function
    specialized = freia_create_helper_function(func_name, NIL);

    // record #outs for this helper, needed for cleaning
    hash_put(signatures, specialized, (void*) (_int) 1);
    set_add_element(helpers, helpers, specialized);

    // cleanup
    free_dag(one);
  }

  pips_assert("specialized function found", specialized!=entity_undefined);
  // directly call the function...
  call c = freia_statement_to_call(dagvtx_statement(v));
  list largs = call_arguments(c);

  int nargs = (int) gen_length(largs);
  pips_assert("3/5 arguments to function", nargs==3 || nargs==5);
  // for convolution, should check that we have a 3x3 kernel...
  call_function(c) = specialized;
  list third = CDR(CDR(largs));
  CDR(CDR(largs)) = NIL;
  gen_full_free_list(third);

  pips_debug(5, "statement %"_intFMT" specialized\n", dagvtx_number(v));
}

/* is v a constant kernel operation?
 */
static bool dagvtx_constant_kernel_p(const dagvtx v)
{
  const freia_api_t * api = get_freia_api_vtx(v);
  intptr_t val;
  return api->opencl.mergeable_kernel &&
    freia_extract_kernel_vtx(v, true, &val, &val, &val, &val,
                             &val, &val, &val, &val, &val);
}

static int compile_this_list(
  string module,
  list lvertices,
  list ls,
  string split_name,
  int n_cut,
  set global_remainings,
  hash_table signatures,
  FILE * helper_file, FILE * opencl,
  set helpers,
  set output_images,
  dag fulld,
  int stnb,
  int max_stnb)
{
  // actually build subdag if something to merge
  dag nd = make_dag(NIL, NIL, NIL);
  FOREACH(dagvtx, v, lvertices)
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

  return stnb;
}

static void migrate_statements(list lvertices, sequence sq, set dones)
{
  set stats = set_make(set_pointer);
  FOREACH(dagvtx, v, lvertices) {
    statement s = dagvtx_statement(v);
    if (s) set_add_element(stats, stats, s);
  }
  freia_migrate_statements(sq, stats, dones);
  set_union(dones, dones, stats);
  set_free(stats);
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
      compile_one_op = get_bool_property("HWAC_OPENCL_COMPILE_ONE_OPERATION"),
      generate_specialized_kernel =
        get_bool_property("HWAC_OPENCL_GENERATE_SPECIAL_KERNEL_OPS");
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

    // we try to aggregate some kernel ops to this merged task...
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
                // *all* its successors are arithmetic merged
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

    set merged = set_make(set_pointer);

    if (lnonmergeable)
    {
      // merge kernel operations with a common input from nonmergeable
      // hmmm... "nonmergeable" is really meant with arithmetic operations
      if (merge_kernels)
      {
        // if starting from sinks, the complexity is not good because we have
        // to rebuild predecessors over and over which requires scanning all
        // vertices...
        // it is a little better when starting the other way around...
        FOREACH(dagvtx, v, dag_vertices(d))
        {
          // detect constant-kernels in nonmergeable with a common input
          list okays = NIL;
          FOREACH(dagvtx, s, dagvtx_succs(v))
          {
            if (set_belong_p(nonmergeable, s) &&
                dagvtx_constant_kernel_p(s))
              okays = CONS(dagvtx, s, okays);
            else if (set_belong_p(mergeable, s) &&
                     dagvtx_is_measurement_p(s))
              // try to backtrack reductions as well?
              okays = CONS(dagvtx, s, okays);
          }

          if (gen_length(okays)>1) // yep, something to do!
          {
            pips_debug(5,
                       "merging %d common input const kernels & reductions\n",
                       (int) gen_length(okays));

            // fix statement connexity
            migrate_statements(okays, sq, dones);

            // let us merge and compile these operations
            stnb = compile_this_list(module, okays, ls, split_name, n_cut,
                                     global_remainings, signatures,
                                     helper_file, opencl, helpers,
                                     output_images, fulld, stnb, max_stnb);
            // this was another cut!
            n_cut++;

            // these are compiled, bye bye!
            FOREACH(dagvtx, s, okays)
            {
              set_add_element(merged, merged, s);
              if (set_belong_p(mergeable, s))
              {
                // dependency hack? is it needed?
                // int n = (int) dagvtx_number(s);
                // if (n>max_stnb) max_stnb = n;
                // cleanup mergeable
                set_del_element(mergeable, mergeable, s);
                gen_remove(&lmergeable, s);
                // also cleanup full dag
                dag_remove_vertex(d, s);
              }
              // non mergeable vertices will be removed later
            }
          }
          gen_free_list(okays);
        }
      }

      // BUG ??? this is too late for the just-aboved merged kernels
      FOREACH(dagvtx, v, lnonmergeable)
      {
        // keep track of previous which may have dependencies... hmmm...
        int n = (int) dagvtx_number(v);
        if (n>max_stnb) max_stnb = n;
      }

      // fix statement connexity...
      migrate_statements(lnonmergeable, sq, dones);

      // possibly compile specialized kernels
      if (generate_specialized_kernel)
      {
        FOREACH(dagvtx, v, lnonmergeable)
          if (!set_belong_p(merged, v))
            opencl_generate_special_kernel_ops(module, v, signatures,
                                               helper_file, opencl, helpers);
      }

      // cleanup initial dag??
      FOREACH(dagvtx, v, lnonmergeable)
        dag_remove_vertex(d, v);
      freia_hack_fix_global_ins_outs(fulld, d);

      gen_free_list(lnonmergeable), lnonmergeable = NIL;
      set_clear(nonmergeable);
      set_free(merged);

      n_cut++; // this was a cut, next cut...
    }

    // then mergeables
    if (lmergeable)
    {
      // fix statement connexity
      migrate_statements(lmergeable, sq, dones);

      // possibly compile
      if (gen_length(lmergeable)>1 || compile_one_op)
      {
        stnb = compile_this_list(module, lmergeable, ls, split_name, n_cut,
                                 global_remainings, signatures,
                                 helper_file, opencl, helpers,
                                 output_images, fulld, stnb, max_stnb);
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
