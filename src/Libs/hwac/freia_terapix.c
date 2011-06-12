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
#include "effects.h"
#include "ri-util.h"
#include "effects-util.h"
#include "properties.h"

#include "freia.h"
#include "freia_spoc_private.h"
#include "hwac.h"

/* Return the first/last available imagelet.
 * This ensures that the choice is deterministic.
 * Moreover, as first numbers are IO imagelets, this help putting outputs
 * in the right imagelet so as to avoid additionnal copies.
 */
static _int select_imagelet(set availables, int * nimgs, boolean first)
{
  _int choice = 0;
  // allocate if no images are available
  if (set_empty_p(availables))
  {
    pips_assert("can create new images", nimgs!=NULL);
    (*nimgs)++;
    choice = *nimgs;
  }
  else // search
  {
    SET_FOREACH(_int, i, availables)
    {
      if (choice==0) choice = i;
      if (first && (i<choice)) choice = i;
      if (!first && (i>choice)) choice = i;
    }
    set_del_element(availables, availables, (void*) choice);
  }
  pips_assert("some choice was made", choice>0);
  return choice;
}

/* @return the dead vertices (their output is dead) after computing v in d.
 */
static void compute_dead_vertices
  (set deads, const set computed, const dag d, const dagvtx v)
{
  list preds = dag_vertex_preds(d, v);
  set futured_computed = set_dup(computed);
  set_add_element(futured_computed, futured_computed, v);
  FOREACH(dagvtx, p, preds)
    if (list_in_set_p(dagvtx_succs(p), futured_computed))
      set_add_element(deads, deads, p);
  gen_free_list(preds);
  set_free(futured_computed);
}

/* tell whether the kernel is used on each of the 4 directions.
 */
static void erosion_optimization
  (dagvtx v, bool * north, bool * south, bool * west, bool * east)
{
  list largs = freia_get_vertex_params(v);
  pips_assert("one kernel", gen_length(largs)==1);
  // default result
  *north = true, *south = true, *west = true, *east = true;
  expression e = EXPRESSION(CAR(largs));
  if (expression_reference_p(e))
  {
    entity var = expression_variable(e);
    // ??? should check const...
    value val = entity_initial(var);
    if (value_expression_p(val))
    {
      expression ival = value_expression(val);
      if (brace_expression_p(ival))
      {
        list iargs = call_arguments(syntax_call(expression_syntax(ival)));
        pips_assert("must be a kernel...", gen_length(iargs)==9);

        // tell whether each kernel element is zero. If in doubt, count as 1.
        bool k00, k10, k20, k01, k11, k21, k02, k12, k22;
        intptr_t i = 0;
        k00 = expression_integer_value(EXPRESSION(CAR(iargs)), &i) && i==0;
        iargs = CDR(iargs);
        k10 = expression_integer_value(EXPRESSION(CAR(iargs)), &i) && i==0;
        iargs = CDR(iargs);
        k20 = expression_integer_value(EXPRESSION(CAR(iargs)), &i) && i==0;
        iargs = CDR(iargs);
        k01 = expression_integer_value(EXPRESSION(CAR(iargs)), &i) && i==0;
        iargs = CDR(iargs);
        k11 = expression_integer_value(EXPRESSION(CAR(iargs)), &i) && i==0;
        iargs = CDR(iargs);
        k21 = expression_integer_value(EXPRESSION(CAR(iargs)), &i) && i==0;
        iargs = CDR(iargs);
        k02 = expression_integer_value(EXPRESSION(CAR(iargs)), &i) && i==0;
        iargs = CDR(iargs);
        k12 = expression_integer_value(EXPRESSION(CAR(iargs)), &i) && i==0;
        iargs = CDR(iargs);
        k22 = expression_integer_value(EXPRESSION(CAR(iargs)), &i) && i==0;
        iargs = CDR(iargs);
        pips_assert("end of list reached", iargs==NIL);

        // summarize for each four directions
        *north = !(k00 && k10 && k20);
        *south = !(k02 && k12 && k22);
        *west = !(k00 && k01 && k02);
        *east = !(k20 && k21 && k22);
      }
    }
  }
}

// stupid hack, to have only one hash table for the 4 directions:
// as the key is a pointer, the alignment ensures that +0 to +3
// are distinct values thus should not clash one with another.
#define NORTH(v) ((void*) (((_int)v)+0))
#define SOUTH(v) ((void*) (((_int)v)+1))
#define WEST(v)  ((void*) (((_int)v)+2))
#define EAST(v)  ((void*) (((_int)v)+3))

/* update_erosions().
 * compute and store the imagelet erosion on vertex v output.
 */
static void update_erosions
  (const dag d, const dagvtx v, hash_table erosion)
{
  _int n = 0, s = 0, w = 0, e = 0;

  // compute most eroded imagelet
  const list preds = dag_vertex_preds(d, v);
  FOREACH(dagvtx, p, preds)
  {
    if ((_int)hash_get(erosion, NORTH(p))>n)
      n = (_int) hash_get(erosion, NORTH(p));
    if ((_int)hash_get(erosion, SOUTH(p))>s)
      s = (_int) hash_get(erosion, SOUTH(p));
    if ((_int)hash_get(erosion, WEST(p))>w)
      w = (_int) hash_get(erosion, WEST(p));
    if ((_int)hash_get(erosion, EAST(p))>e)
      e = (_int) hash_get(erosion, EAST(p));
  }
  gen_free_list(preds);

  // update with vertex erosion
  const freia_api_t * api = dagvtx_freia_api(v);

  // for erode/dilate, I look at the kernel, and if it is a
  // "const" with initial values { 000 XXX XXX } => north=0 and so on.
  // this is interesting on licensePlate, even if the zero
  // computations are still performed...
  bool north = true, south = true, west = true, east = true;
  if (api->terapix.north)
    erosion_optimization(v, &north, &south, &west, &east);

  if (north) n += api->terapix.north;
  if (south) s += api->terapix.south;
  if (west) w += api->terapix.west;
  if (east) e += api->terapix.east;

  // store results
  hash_put(erosion, NORTH(v), (void*) n);
  hash_put(erosion, SOUTH(v), (void*) s);
  hash_put(erosion, WEST(v), (void*) w);
  hash_put(erosion, EAST(v), (void*) e);
}

/* compute some measures about DAG d.
 * @return its length (depth)
 * width, aka maximum live produced image in a level by level
 * terapix computation cost (per row)
 * scheduling (could do a better job with a better scheduling?)
 * maximal erosion in all four directions
 */
static int dag_terapix_measures
  (const dag d, hash_table erosion,
   int * width, int * cost, int * nops,
   int * north, int * south, int * west, int * east)
{
  set processed = set_make(set_pointer);
  int dcost = 0, dlength = 0, dwidth = gen_length(dag_inputs(d)), dnops = 0;
  bool keep_erosion = erosion!=NULL;
  // vertex output -> NSWE erosion (v+0 to v+3 is N S W E)
  if (!keep_erosion) erosion = hash_table_make(hash_pointer, 0);

  FOREACH(dagvtx, in, dag_inputs(d))
    update_erosions(d, in, erosion);

  list lv;
  while ((lv = get_computable_vertices(d, processed, processed, processed)))
  {
    dlength++;
    int level_width = 0;
    FOREACH(dagvtx, v, lv)
    {
      const freia_api_t * api = dagvtx_freia_api(v);
      dcost += api->terapix.cost;
      // only count non null operations
      if (api->terapix.cost) dnops ++;
      if (api->arg_img_out) level_width++;
      update_erosions(d, v, erosion);
    }
    if (level_width>dwidth) dwidth = level_width;

    set_append_list(processed, lv);
    gen_free_list(lv);
  }

  // update width
  int nouts = gen_length(dag_outputs(d));
  if (nouts>dwidth) dwidth = nouts;

  // compute overall worth erosion
  int n=0, s=0, w=0, e=0;
  FOREACH(dagvtx, out, dag_outputs(d))
  {
    if ((_int)hash_get(erosion, NORTH(out))>n)
      n = (int) (_int) hash_get(erosion, NORTH(out));
    if ((_int)hash_get(erosion, SOUTH(out))>s)
      s = (int) (_int) hash_get(erosion, SOUTH(out));
    if ((_int)hash_get(erosion, WEST(out))>w)
      w = (int) (_int) hash_get(erosion, WEST(out));
    if ((_int)hash_get(erosion, EAST(out))>e)
      e = (int) (_int) hash_get(erosion, EAST(out));
  }

  // cleanup
  set_free(processed);
  if (!keep_erosion) hash_table_free(erosion);

  // return results
  *north = n, *south = s, *west = w, *east = e,
    *width = dwidth, *cost = dcost, *nops = dnops;
  return dlength;
}

/* fill in erosion hash table from dag d.
 */
static void dag_terapix_erosion(const dag d, hash_table erosion)
{
  int i = 0;
  dag_terapix_measures(d, erosion, &i, &i, &i, &i, &i, &i, &i);
}

/* @return the list of inputs to vertex v as imagelet numbers.
 */
static list /* of ints */ dag_vertex_pred_imagelets
  (const dag d, const dagvtx v, const hash_table allocation)
{
  list limagelets = NIL;
  FOREACH(entity, img, vtxcontent_inputs(dagvtx_content(v)))
  {
    dagvtx prod = dagvtx_get_producer(d, v, img);
    pips_assert("some producer found!", prod!=NULL);
    limagelets =
      gen_nconc(limagelets,
                CONS(int, (int)(_int) hash_get(allocation, prod), NIL));
  }
  return limagelets;
}

/* allocate bitfield to described used cells in global memory.
 */
static bool * terapix_gram_init(void)
{
  int row_size = get_int_property(trpx_gram_width);
  int col_size = get_int_property(trpx_gram_height);
  bool * gram = (bool *) malloc(sizeof(bool)*row_size*col_size);
  pips_assert("malloc ok", gram);
  for (int i=0; i<row_size*col_size; i++)
    gram[i] = false;
  return gram;
}

/* terapix allocate widthxheight in global memory
 * @return *x *y pointer to available memory
 */
static void terapix_gram_allocate
  (bool * used, int width, int height, int * x, int * y)
{
  int row_size = get_int_property(trpx_gram_width);
  int col_size = get_int_property(trpx_gram_height);
  for (int j = 0; j<col_size-height+1; j++)
  {
    for (int i = 0; i<row_size-width+1; i++)
    {
      boolean ok = true;
      for (int w = 0; ok && w<width; w++)
        for (int h = 0; ok && h<height; h++)
          ok &= !used[(i+w)+(j+h)*row_size];
      if (ok)
      {
        for (int w = 0; w<width; w++)
          for (int h = 0; h<height; h++)
            used[(i+w)+(j+h)*row_size] = true;
        *x = i;
        *y = j;
        return;
      }
    }
  }
  pips_internal_error("cannot find available memory for %dx%d", width, height);
}

#define IMG_PTR "imagelet_"

/* generate an image symbolic pointer (a name:-).
 */
static void terapix_image(string_buffer sb, int ff, int n)
{
  pips_assert("valid flip-flop", ff==0 || ff==1);
  pips_assert("valid image number", n!=0);
  if (n>0)
    sb_cat(sb, IMG_PTR, itoa(n));
  else
    sb_cat(sb, IMG_PTR "io_", itoa(-n), ff? "_1": "_0");
}

/* set a double buffered image argument.
 */
static void terapix_mcu_img(string_buffer code, int op, string ref, int n)
{
  sb_cat(code, "  mcu_macro[0][", itoa(op), "].", ref, " = ");
  terapix_image(code, 0, n);
  sb_cat(code, ";\n");
  sb_cat(code, "  mcu_macro[1][", itoa(op), "].", ref, " = ");
  terapix_image(code, 1, n);
  sb_cat(code, ";\n");
}

/* set an integer argument.
 */
static void terapix_mcu_int(string_buffer code, int op, string ref, int val)
{
  sb_cat(code, "  mcu_macro[0][", itoa(op), "].", ref);
  sb_cat(code, " = ", itoa(val), ";\n");
  sb_cat(code, "  mcu_macro[1][", itoa(op), "].", ref);
  sb_cat(code, " = ", itoa(val), ";\n");
}

/* set some value string argument.
 */
static void terapix_mcu_val(string_buffer code, int op, string r, string s)
{
  sb_cat(code, "  mcu_macro[0][", itoa(op), "].", r, " = ", s, ";\n");
  sb_cat(code, "  mcu_macro[1][", itoa(op), "].", r, " = ", s, ";\n");
}

/* set some prefixed value string argument.
 */
static void terapix_mcu_pval(string_buffer code, int op, string ref,
                             string p, string s)
{
  sb_cat(code, "  mcu_macro[0][", itoa(op), "].", ref,
         " = ", p, s, ";\n");
  sb_cat(code, "  mcu_macro[1][", itoa(op), "].", ref,
         " = ", p, s, ";\n");
}

/* copy some operator parameters in the global ram (aka gram).
 * the coordinates used are  (x_<name>, y_<name>).
 */
static void gram_param
  (string_buffer code, string_buffer decl,
   string name, dagvtx v, hash_table hparams,
   int width, int height, bool * used)
{
  int size = width*height;
  pips_assert("something to copy...", size>0);

  int x = 0, y = 0;
  terapix_gram_allocate(used, width, height, &x, &y);

  sb_cat(decl, "  // operation ", name, " parameters\n");
  sb_cat(decl, "  int16_t p_", name, "[", itoa(size), "];\n");
  sb_cat(decl, "  const int32_t x_", name, " = ", itoa(x), ";\n");
  sb_cat(decl, "  const int32_t y_", name, " = ", itoa(y), ";\n");

  sb_cat(code, "  // copy of operation ", name, " parameters\n");
  list largs = freia_get_vertex_params(v);
  pips_assert("some args...", gen_length(largs)>0);
  string p1 = hash_get(hparams, EXPRESSION(CAR(largs)));
  // copy code...
  switch (size)
  {
  case 1: // constant
  case 3: // threshold min/max/bin
    sb_cat(code, "  p_", name, "[0] = ", p1, ";\n");
    if (size==1) break;
    sb_cat(code, "  p_", name, "[1] = ",
           hash_get(hparams, EXPRESSION(CAR(CDR(largs)))), ";\n");
    sb_cat(code, "  p_", name, "[2] = ",
           hash_get(hparams, EXPRESSION(CAR(CDR(CDR(largs))))), ";\n");
    break;
  case 9: // kernel
    sb_cat(code,
           "  for(i=0; i<9; i++)\n"
           "    p_", name, "[i] = ", p1, "[i];\n");
    break;
  default:
    pips_internal_error("unexpected gram size");
  }

  sb_cat(code, "  gram.xoffset = x_", name, ";\n");
  sb_cat(code, "  gram.yoffset = y_", name, ";\n");
  sb_cat(code, "  gram.width = ", itoa(width), ";\n");
  sb_cat(code, "  gram.height = ", itoa(height), ";\n");
  sb_cat(code, "  gram.params = p_", name, ";\n");
  sb_cat(code, "  freia_mg_write_dynamic_param(&dyn_param);\n");
}

/* manage GRAM global memory to pass parameters.
 */
static void terapix_gram_management
  (string_buffer code, // generated code
   string_buffer decl, // generated declarations
   int op,             // operation
   const freia_api_t * api,
   const dagvtx v,     // current vertex
   hash_table hparams, // expression -> parameter...
   bool * used)        // current usage of GRAM
{
  if (!api->arg_misc_in) return;

  list largs = freia_get_vertex_params(v);
  string p1 = hash_get(hparams, EXPRESSION(CAR(largs)));

  // is it a new, never handled, parameter?
  bool initialize = !hash_defined_p(hparams, p1);
  // name suffix for variables...
  if (initialize) hash_put(hparams, p1, strdup(itoa(op)));
  string name = hash_get(hparams, p1);

  if (initialize)
  {
    switch (api->arg_misc_in)
    {
    case 3: // threshold
      gram_param(code, decl, name, v, hparams, 3, 1, used);
      break;
    case 1: // kernel or operation with a constant
      if (api->terapix.north)
        // let us say it is a kernel...
        gram_param(code, decl, name, v, hparams, 3, 3, used);
      else
        gram_param(code, decl, name, v, hparams, 1, 1, used);
      break;
    default:
      pips_internal_error("unexpected number of input image arguments");
    }
  }

  // is it always [xy]min3?
  terapix_mcu_pval(code, op, "xmin3", "x_", name);
  terapix_mcu_pval(code, op, "ymin3", "y_", name);
}

/* generate terapix code for
 * @param code, code stream being generated
 * @param decl, declaration stream being generated
 * @param op, operation number
 * @param api, actual freia operator called
 * @param used, array to keep track of what gram cells are used
 * @param hparam, expression to parameter mapping
 * @param v, dag vertex of the current operation
 * @param ins, list of image number inputs (i.e. operation arguments)
 * @param out, image number output for the operation
 */
static void terapix_macro_code
  (string_buffer code, string_buffer decl,
   int op, const freia_api_t * api, bool * used,
   hash_table hparams, const dagvtx v, const list ins, int out)
{
  // check image in/out consistency
  pips_assert("#ins ok", gen_length(ins)==api->arg_img_in);
  pips_assert("out ok", out? api->arg_img_out: !api->arg_img_out);

  switch (api->arg_img_in)
  {
  case 2:
    pips_assert("2 ins, alu operation...", out);
    terapix_mcu_img(code, op, "xmin1", INT(CAR(ins)));
    terapix_mcu_int(code, op, "ymin1", 0);
    terapix_mcu_img(code, op, "xmin2", INT(CAR(CDR(ins))));
    terapix_mcu_int(code, op, "ymin2", 0);
    terapix_mcu_img(code, op, "xmin3", out);
    terapix_mcu_int(code, op, "ymin3", 0);
    break;
  case 1:
    // alu: image op cst 1
    // threshold 3x1
    // erode/dilate 3x3
    // copy
    terapix_mcu_img(code, op, "xmin1", INT(CAR(ins)));
    terapix_mcu_int(code, op, "ymin1", 0);
    if (out) {
      terapix_mcu_img(code, op, "xmin2", out);
      terapix_mcu_int(code, op, "ymin2", 0);
    }
    terapix_gram_management(code, decl, op, api, v, hparams, used);
    break;
  case 0:
    pips_assert("no input, one output image", out);
    // const image generation... NSP
    terapix_mcu_img(code, op, "xmin???", out);
    terapix_mcu_int(code, op, "ymin???", 0);
    terapix_gram_management(code, decl, op, api, v, hparams, used);
    break;
  default:
    pips_internal_error("unexpected number of input images");
  }
  terapix_mcu_val(code, op, "iter1", "TERAPIX_PE_NUMBER");
  terapix_mcu_val(code, op, "iter2", "imagelet_size");
  terapix_mcu_val(code, op, "iter3", "0");
  terapix_mcu_val(code, op, "iter4", "0");
  terapix_mcu_val(code, op, "addrStart", api->terapix.ucode);
}

/* generate a terapix call for dag thedag.
 * the memory allocation is managed here.
 * however this function is dumb, the scheduling is just inherited as is...
 */
static void freia_terapix_call
  (const string module,
   const string fname_dag,
   string_buffer code,
   dag thedag,
   list /* of expression */ *params)
{
  // total number of imagelets used for computing the dag
  int n_imagelets = 0;
  // number of input images
  int n_ins = gen_length(dag_inputs(thedag));
  // number of output images
  int n_outs = gen_length(dag_outputs(thedag));
  // number of needed double buffers for I/Os.
  // this is also the number of I/O images
  int n_double_buffers = (n_ins>n_outs)? n_ins: n_outs; // max(#ins, #outs)

  pips_assert("some I/O images", n_double_buffers>0);

  // the memory will be decremented for "measures" data (reductions),
  // and then divided among imagelets
  int available_memory = get_int_property(trpx_mem_prop);

  string_buffer
    head = string_buffer_make(true),
    decl = string_buffer_make(true),
    body = string_buffer_make(true),
    dbio = string_buffer_make(true),
    tail = string_buffer_make(true);

  // array variable name in caller -> local kernel parameter
  // used to detect if a kernel is already available, so as to skip
  // its copy and share the generated parameter.
  hash_table hparams = hash_table_make(hash_pointer, 0);

  // number of arguments to generated function
  int nargs = 0;

  // get stats
  int length, width, cost, nops, n, s, w, e;
  length = dag_terapix_measures(thedag, NULL,
                                &width, &cost, &nops, &n, &s, &w, &e);

  int comm = get_int_property(trpx_dmabw_prop);

  // show stats in function's comments
  sb_cat(head, "\n/* FREIA terapix helper function for module ", module, "\n");
  sb_cat(head, " * ", itoa(n_ins), " input image", n_ins>1? "s": "");
  sb_cat(head, ", ", itoa(n_outs), " output image", n_outs>1? "s": "", "\n");
  sb_cat(head, " * ", itoa(nops), " image operations in dag\n");
  sb_cat(head, " * dag length is ", itoa(length));
  sb_cat(head, ", dag width is ", itoa(width), "\n");
  sb_cat(head, " * costs in cycles per imagelet row:\n");
  sb_cat(head, " * - computation: ", itoa(cost), "\n");
  sb_cat(head, " * - communication: ", itoa(comm*(n_ins+n_outs)), "\n");
  sb_cat(head, " */\n");

  // generate function declaration
  sb_cat(head, "freia_status ", fname_dag, "(");
  for (int i = 0; i<n_outs; i++)
    sb_cat(head, nargs++? ",": "", "\n  " FREIA_IMAGE "o", itoa(i));
  for (int i = 0; i<n_ins; i++)
    sb_cat(head, nargs++? ",": "", "\n  " FREIA_IMAGE "i", itoa(i));
  // other arguments to come...

  // corresponding helper call arguments
  list limg = NIL;
  FOREACH(dagvtx, voa, dag_outputs(thedag))
    limg = CONS(entity, vtxcontent_out(dagvtx_content(voa)), limg);
  FOREACH(dagvtx, via, dag_inputs(thedag))
    limg = CONS(entity, vtxcontent_out(dagvtx_content(via)), limg);
  limg = gen_nreverse(limg);

  sb_cat(decl,
         "{\n"
         "  // declarations:\n"
         "  freia_microcode mcode;\n"
         "  freia_op_param param;\n"
         "  freia_dynamic_param dyn_param;\n"
         "  terapix_gram gram;\n"
         "  int i;\n"
         "  freia_status ret;\n"
         "  // overall structure which describes the computation\n"
         "  terapix_mcu_instr mcu_instr;\n");

  sb_cat(body,
         "\n"
         "  // body:\n"
         "  // mcode param\n"
         "  mcode.raw = (void*) terapix_ucode_array;\n"
         "  mcode.size = TERAPIX_UCODE_SIZE_T;\n"
         "  freia_mg_write_microcode(&mcode);\n"
         "\n"
         "  // subimage operation\n"
         "  param.size = -1; // not used\n"
         "  param.raw = (void*) &mcu_instr;\n"
         "\n"
         "  // dyn_param contents\n"
         "  dyn_param.raw = &gram;\n"
         "  dyn_param.size = sizeof(terapix_gram);\n");

  // string_buffer head, decls, end, settings;

  // schedule to imagelet numbers as needed...
  // use a named pointer the value of which will be known later,
  // depending on the number of needed imagelets
  // operation -> imagelet number
  // the imagelet number is inverted if it is an I/O
  hash_table allocation = hash_table_make(hash_pointer, 0);
  set computed = set_make(set_pointer);

  // the GRAM initialization may be shared between helper calls?
  bool * used = terapix_gram_init();

  if (n_ins)
  {
    // ??? they should be given in the order of the arguments
    // when calling the runtime function.
    int n = 0;
    sb_cat(dbio, "  // inputs:\n");
    FOREACH(dagvtx, in, dag_inputs(thedag))
    {
      // update primary imagelet number
      n_imagelets++;
      set_add_element(computed, computed, in);
      // ??? stupid bug which filters undefined values, i.e. -16
      // I should really use a container...
      hash_put(allocation, in, (void*) (_int) -n_imagelets);

      string sn = strdup(itoa(n)), si = strdup(itoa(n_imagelets));

      // ??? tell that n_imagelets is an input
      sb_cat(dbio, "  // - imagelet ", si, " is i", sn, " for ",
             entity_user_name(vtxcontent_out(dagvtx_content(in))),
             "\n");

      sb_cat(dbio, "  tile_in[0][", sn, "].x = " IMG_PTR "io_", si, "_0;\n");
      sb_cat(dbio, "  tile_in[0][", sn, "].y = 0;\n");
      sb_cat(dbio, "  tile_in[1][", sn, "].x = " IMG_PTR "io_", si, "_1;\n");
      sb_cat(dbio, "  tile_in[1][", sn, "].y = 0;\n");
      free(sn);
      free(si);
      n++;
    }
    sb_cat(dbio, "\n");
  }
  else
  {
    sb_cat(dbio, "  // no input\n\n");
  }

  set avail_img = set_make(set_pointer);

  // complete if need be, there will be AT LEAST this number of images
  while (n_imagelets<n_double_buffers)
    set_add_element(avail_img, avail_img, (void*) (_int) ++n_imagelets);

  set deads = set_make(set_pointer);
  // newly created parameters at this round

  // generate code for every computation vertex
  int n_ops = 0;
  list vertices = gen_nreverse(gen_copy_seq(dag_vertices(thedag)));
  FOREACH(dagvtx, current, vertices)
  {
    // skip this vertex
    if (set_belong_p(computed, current))
      continue;
    if (dagvtx_other_stuff_p(current))
      continue;

    // compute freed images...
    set_clear(deads);
    compute_dead_vertices(deads, computed, thedag, current);

    vtxcontent vc = dagvtx_content(current);
    pips_assert("there is a statement",
                pstatement_statement_p(vtxcontent_source(vc)));
    statement s = pstatement_statement(vtxcontent_source(vc));
    call c = freia_statement_to_call(s);
    // int optype = dagvtx_optype(current);
    int opid = dagvtx_opid(current);
    const freia_api_t * api = get_freia_api(opid);
    pips_assert("freia api found", api!=NULL);

    if (api->terapix.memory)
    {
      available_memory -= api->terapix.memory;
      string saddr = strdup(itoa(available_memory));

      // computation
      sb_cat(body, "  // set measure ", api->compact_name, " at ", saddr, "\n");
      terapix_mcu_val(body, n_ops, "xmin3", saddr);
      terapix_mcu_val(body, n_ops, "ymin3", "0");

      // extraction
      sb_cat(tail, "  // get measure ", api->compact_name,
             " result from ", saddr, "\n");
      sb_cat(tail, "  // ??? not implemented yet...\n");
      free(saddr);
    }

    // if inplace, append freed images to availables
    if (api->terapix.inplace)
    {
      SET_FOREACH(dagvtx, v, deads)
      {
        _int img = (_int) hash_get(allocation, v);
        if (img<0) img=-img;
        set_add_element(avail_img, avail_img, (void*) img);
      }
    }

    // generate inS -> out computation
    // - code
    // imagelet inputs
    list ins = dag_vertex_pred_imagelets(thedag, current, allocation);
    sb_cat(body, "  // ", itoa(n_ops), ": ", api->compact_name, "(");
    if (ins)
    {
      // show input imagelet numbers
      int in_count=0;
      FOREACH(int, i, ins)
        sb_cat(body, in_count++? ",": "", itoa(i>0? i: -i));
    }
    sb_cat(body, ")");

    // imagelet output
    _int choice = 0;
    if (api->arg_img_out==1)
    {
      bool is_output = gen_in_list_p(current, dag_outputs(thedag));
      // SELECT one available
      choice = select_imagelet(avail_img, &n_imagelets, is_output);
      sb_cat(body, " -> ", itoa((int) choice));
      // there is a subtlety here, if no I/O image was available
      // then a copy will have to be inserted later on, see "PANIC".
      if (is_output && choice<=n_double_buffers) choice = -choice;
      hash_put(allocation, current, (void*) choice);
    }
    sb_cat(body, "\n");

    // update helper call arguments...
    *params = gen_nconc(*params,
                        freia_extract_params(opid, call_arguments(c),
                                             head, hparams, &nargs));

    if (api==hwac_freia_api(AIPO "copy") && choice==INT(CAR(ins)))
    {
      // skip in place copy, which may happen if the selected target
      // image buffer happens to be the same as the input.
      sb_cat(body, "  // in place copy skipped\n");
      n_ops--;
    }
    else
    {
      terapix_macro_code(body, decl, n_ops, api, used,
                         hparams, current, ins, choice);
    }

    gen_free_list(ins), ins=NIL;

    // if NOT inplace, append freed images to availables now
    if (!api->terapix.inplace)
    {
      SET_FOREACH(dagvtx, v, deads)
      {
        _int img = (_int) hash_get(allocation, v);
        if (img<0) img=-img;
        set_add_element(avail_img, avail_img, (void*) img);
      }
    }

    set_add_element(computed, computed, current);
    n_ops++;
  }

  // handle function image arguments
  freia_add_image_arguments(limg, params);

  if (n_outs)
  {
    int n = 0;
    sb_cat(dbio, "  // outputs:\n");
    FOREACH(dagvtx, out, dag_outputs(thedag))
    {
      int oimg = (int) (_int) hash_get(allocation, out);
      if (oimg<0) oimg=-oimg;
      if (oimg>n_double_buffers)
      {
        // PANIC:
        // if there is no available "IO" imagelet when an output is
        // produced, it will have to be put there with a copy later on.
        int old = oimg;
        oimg = select_imagelet(avail_img, NULL, true);
        pips_assert("IO imagelet found for output", oimg<=n_double_buffers);

        // generate copy code old -> oimg
        // hmmm... could not generate a test case where this is triggered...
        // the additional cost which should be reported?
        sb_cat(body, "  // output copy ", itoa(old));
        sb_cat(body, " -> ", itoa(oimg), "\n");
        list lic = CONS(int, old, NIL);
        // -oimg to tell the code generator that we are dealing with
        // a double buffered image...
        terapix_macro_code(body, decl, n_ops, hwac_freia_api(AIPO "copy"),
                           NULL, NULL, NULL, lic, -oimg);
        gen_free_list(lic);
        n_ops++;
      }
      // tell that oimg is an output
      // ??? tell that n_imagelets is an input
      string sn = strdup(itoa(n)), so = strdup(itoa(oimg));
      sb_cat(dbio, "  // - imagelet ", so);
      sb_cat(dbio, " is o", sn, " for ");
      sb_cat(dbio,
             (char*)entity_user_name(vtxcontent_out(dagvtx_content(out))),
             "\n");
      sb_cat(dbio, "  tile_out[0][", sn, "].x = " IMG_PTR"io_", so, "_0;\n");
      sb_cat(dbio, "  tile_out[0][", sn, "].y = 0;\n");
      sb_cat(dbio, "  tile_out[1][", sn, "].x = " IMG_PTR"io_", so, "_1;\n");
      sb_cat(dbio, "  tile_out[1][", sn, "].y = 0;\n");
      free(sn);
      free(so);
      n++;
    }
    sb_cat(dbio, "\n");
    sb_cat(body, "\n");
  }
  else
  {
    sb_cat(dbio, "  // no output\n\n");
  }

  // now I know how many imagelets are needed
  int total_imagelets = n_imagelets + n_double_buffers;
  int imagelet_rows = available_memory/total_imagelets; // round down

  // declarations when we know the number of operations
  // [2] for flip/flop double buffer handling
  sb_cat(decl, "  terapix_mcu_macrocode mcu_macro[2][", itoa(n_ops), "];\n");
  if (n_ins)
    sb_cat(decl, "  terapix_tile_info tile_in[2][", itoa(n_ins), "];\n");
  if (n_outs)
    sb_cat(decl, "  terapix_tile_info tile_out[2][", itoa(n_outs), "];\n");

  // computed values
  sb_cat(decl, "\n  // imagelets definitions:\n");
  sb_cat(decl, "  // - ", itoa(n_imagelets), " computation imagelets\n");
  sb_cat(decl, "  // - ", itoa(n_double_buffers), " double buffer imagelets\n");

  // this is really a MAXIMUM available size
  // the runtime can use that or less
  sb_cat(decl, "  int imagelet_size = ", itoa(imagelet_rows), ";\n");

  // generate imagelet pointers
  for (int i=1; i<=total_imagelets; i++)
  {
    sb_cat(decl, "  int " IMG_PTR, itoa(i), " = ");
    sb_cat(decl, itoa(imagelet_rows * (i-1)), ";\n");
  }

  // generate imagelet double buffer pointers
  // sb_cat(dbio, "  // double buffer management:\n");
  sb_cat(decl, "\n  // double buffer assignment\n");
  for (int i=1; i<=n_double_buffers; i++)
  {
    // sb_cat(dbio, "  // - buffer ", itoa(i), "/");
    // sb_cat(dbio, itoa(i+n_imagelets), "\n");

    sb_cat(decl, "  int " IMG_PTR "io_", itoa(i), "_0 = ");
    sb_cat(decl, IMG_PTR, itoa(i), ";\n");
    sb_cat(decl, "  int " IMG_PTR "io_", itoa(i), "_1 = ");
    sb_cat(decl, IMG_PTR, itoa(i+n_imagelets), ";\n");
  }

  // incorporate IO stuff
  string_buffer_append_sb(body, dbio);
  string_buffer_free(&dbio);

  // tell about imagelet erosion...
  // current output should be max(w,e) & max(n,s)
  sb_cat(body, "\n  // imagelet erosion for the computation\n");
  sb_cat(body, "  mcu_instr.borderTop    = ", itoa(n), ";\n");
  sb_cat(body, "  mcu_instr.borderBottom = ", itoa(s), ";\n");
  sb_cat(body, "  mcu_instr.borderLeft   = ", itoa(w), ";\n");
  sb_cat(body, "  mcu_instr.borderRight  = ", itoa(e), ";\n");
  sb_cat(body, "  mcu_instr.imagelet_height = imagelet_size;\n");
  sb_cat(body, "  mcu_instr.imagelet_width  = TERAPIX_PE_NUMBER;\n");

  sb_cat(body, "\n  // outputs\n");
  sb_cat(body, "  mcu_instr.nbout = ", itoa(n_outs), ";\n");
  if (n_outs)
    sb_cat(body,
           "  mcu_instr.out0 = tile_out[0];\n"
           "  mcu_instr.out1 = tile_out[1];\n");
  else
    sb_cat(body,
           "  mcu_instr.out0 = NULL;\n"
           "  mcu_instr.out1 = NULL;\n");

  sb_cat(body, "\n  // inputs\n");
  sb_cat(body, "  mcu_instr.nbin = ", itoa(n_ins), ";\n");
  if (n_ins)
    sb_cat(body,
           "  mcu_instr.in0 = tile_in[0];\n"
           "  mcu_instr.in1 = tile_in[1];\n");
  else
    sb_cat(body,
           "  mcu_instr.in0 = NULL;\n"
           "  mcu_instr.in1 = NULL;\n");

  sb_cat(body,
         "\n  // actual instructions\n"
         "  mcu_instr.nbinstr = ", itoa(n_ops), ";\n"
         "  mcu_instr.instr0   = mcu_macro[0];\n"
         "  mcu_instr.instr1   = mcu_macro[1];\n");


  // tell about imagelet size
  // ??? NOTE: the runtime *MUST* take care of possible in/out aliasing
  sb_cat(body, "\n"
         "  // call terapix runtime\n"
         "  ret = freia_cg_template_process(&param");
  for (int i=0; i<n_outs; i++)
    sb_cat(body, ", o", itoa(i));
  for (int i=0; i<n_ins; i++)
    sb_cat(body, ", i", itoa(i));
  sb_cat(body, ");\n");

  // ??? I must compute the total erosion
  // ??? I should check that something IS computed...

  string_buffer_append_sb(code, head);
  sb_cat(code, ")\n");
  string_buffer_append_sb(code, decl);
  string_buffer_append_sb(code, body);
  sb_cat(code, "\n");
  sb_cat(code, "  // extract measures\n");
  string_buffer_append_sb(code, tail);
  sb_cat(code, "\n  return ret;\n}\n\n");

  // cleanup computed vertices: they are REMOVED from the dag and "killed"
  // ??? should rather return them and the caller should to the cleaning?
  FOREACH(dagvtx, vr, vertices)
  {
    dag_remove_vertex(thedag, vr);
    if (set_belong_p(computed, vr))
    {
      if (pstatement_statement_p(vtxcontent_source(dagvtx_content(vr))))
        hwac_kill_statement(pstatement_statement
                            (vtxcontent_source(dagvtx_content(vr))));
      free_dagvtx(vr);
    }
  }
  // dag_compute_outputs(thedag);
  // cleanup
  gen_free_list(vertices), vertices = NIL;
  string_buffer_free(&head);
  string_buffer_free(&decl);
  string_buffer_free(&body);
  string_buffer_free(&tail);
  hash_table_free(allocation);
  // ??? free strings!
  hash_table_free(hparams);
  set_free(avail_img);
  set_free(computed);
  set_free(deads);
  free(used);
}

/* global variable used by the dagvtx_terapix_priority function,
 * because qsort does not allow to pass some descriptor.
 */
static hash_table erosion = NULL;

/* comparison function for sorting dagvtx in qsort,
 * this is deep voodoo, because the priority has an impact on
 * correctness? that should not be the case as only computations
 * allowed by dependencies are schedule.
 * tells v1 < (before) v2 => -1
 */
static int dagvtx_terapix_priority(const dagvtx * v1, const dagvtx * v2)
{
  pips_assert("global erosion is set", erosion!=NULL);

  // ??? should prioritize if more outputs?
  // ??? should prioritize inplace?
  // ??? should prioritize no erosion first? levels do that currrently?
  string why = "none";
  int result = 0;
  vtxcontent
    c1 = dagvtx_content(*v1),
    c2 = dagvtx_content(*v2);
  const freia_api_t
    * a1 = dagvtx_freia_api(*v1),
    * a2 = dagvtx_freia_api(*v2);

  // prioritize first scalar ops, measures and last copies
  // if there is only one of them
  if (vtxcontent_optype(c1)!=vtxcontent_optype(c2))
  {
    // scalars operations first to remove (scalar) dependences
    if (vtxcontent_optype(c1)==spoc_type_oth)
      result = -1, why = "scal";
    else if (vtxcontent_optype(c2)==spoc_type_oth)
      result = 1, why = "scal";
    // then measurements are put first
    else if (vtxcontent_optype(c1)==spoc_type_mes)
      result = -1, why = "mes";
    else if (vtxcontent_optype(c2)==spoc_type_mes)
      result = 1, why = "mes";
    // the copies are performed last...
    else if (vtxcontent_optype(c1)==spoc_type_nop)
      result = 1, why = "copy";
    else if (vtxcontent_optype(c2)==spoc_type_nop)
      result = -1, why = "copy";
    // idem with image generation...
    else if (vtxcontent_optype(c1)==spoc_type_alu &&
             vtxcontent_inputs(c1)==NIL)
      result = 1, why = "gen";
    else if (vtxcontent_optype(c2)==spoc_type_alu &&
             vtxcontent_inputs(c2)==NIL)
      result = -1, why = "gen";
    // ??? do inplace last
    // ??? or ONLY if there is a shared input?
    else if (a1->terapix.inplace && !a2->terapix.inplace)
      result = 1, why = "inplace";
    else if (!a1->terapix.inplace && a2->terapix.inplace)
      result = -1, why = "inplace";
  }

  // ??? priorise when an image is freed

  if (result==0 &&
      // is there an image output?
      vtxcontent_optype(c1)!=spoc_type_oth &&
      vtxcontent_optype(c1)!=spoc_type_mes &&
      vtxcontent_optype(c2)!=spoc_type_oth &&
      vtxcontent_optype(c2)!=spoc_type_mes)
  {
    ifdebug(6) {
      dagvtx_dump(stderr, "v1", *v1);
      dagvtx_dump(stderr, "v2", *v2);
    }
    pips_assert("erosion is defined",
                hash_defined_p(erosion, NORTH(*v1)) &&
                hash_defined_p(erosion, NORTH(*v2)));

    // try to conclude with erosions:
    // not sure about the right partial order to use...
    int e1 = (int)
      ((_int) hash_get(erosion, NORTH(*v1)) +
       (_int) hash_get(erosion, SOUTH(*v1)) +
       (_int) hash_get(erosion, WEST(*v1)) +
       (_int) hash_get(erosion, EAST(*v1))),
      e2 = (int)
      ((_int) hash_get(erosion, NORTH(*v2)) +
       (_int) hash_get(erosion, SOUTH(*v2)) +
       (_int) hash_get(erosion, WEST(*v2)) +
       (_int) hash_get(erosion, EAST(*v2)));

    pips_debug(6, "e1=%d, e2=%d\n", e1, e2);

    if (e1!=e2)
      result = e1-e2, why = "erosion";
  }

  // ??? I should look at in place?
  // ??? I should look at the number live uses?

  if (result==0)
  {
    // if not set by previous case, use other criterions
    int
      l1 = (int) gen_length(vtxcontent_inputs(c1)),
      l2 = (int) gen_length(vtxcontent_inputs(c2));

    // count non mesure successors:
    int nms1 = 0, nms2 = 0;

    FOREACH(dagvtx, vs1, dagvtx_succs(*v1))
      if (dagvtx_optype(vs1)!=spoc_type_mes) nms1++;

    FOREACH(dagvtx, vs2, dagvtx_succs(*v2))
      if (dagvtx_optype(vs2)!=spoc_type_mes) nms2++;

    if (l1!=l2 && (l1==0 || l2==0))
      // put image generators at the end, after any other computation
      result = l2-l1, why = "args";
    else if (nms1!=nms2 && l1==1 && l2==1)
      // the less successors the better? the rational is:
      // - mesures are handled before and do not have successors anyway,
      // - so this is about whether a result of an unary op is reused by
      //   two nodes, in which case it will just jam the pipeline, so
      //   try to put other computations before it. Note that mes
      //   successors do not really count, as the image is not lost.
      result = nms1 - nms2, why = "succs";
    else if (l1!=l2)
      // else ??? no effect on my validation.
      result = l2-l1, why = "args2";
    else if (vtxcontent_optype(c1)!=vtxcontent_optype(c2))
      // otherwise use the op types, which are somehow ordered
      // so that if all is well the pipe is filled in order.
      result = vtxcontent_optype(c1) - vtxcontent_optype(c2), why = "ops";
    else
      // if all else fails, rely on statement numbers.
      result = dagvtx_number(*v1) - dagvtx_number(*v2), why = "stats";
  }

  pips_debug(6, "%" _intFMT " %s %s %" _intFMT " %s (%s)\n",
             dagvtx_number(*v1), dagvtx_operation(*v1),
             result<0? ">": (result==0? "=": "<"),
             dagvtx_number(*v2), dagvtx_operation(*v2), why);

  pips_assert("total order", v1==v2 || result!=0);
  return result;
}

/* split a dag on scalar dependencies only, with a greedy heuristics.
 */
static list /* of dags */ split_dag_on_scalars(const dag initial)
{
  if (!single_image_assignement_p(initial))
    // well, it should work most of the time, so only a warning
    pips_user_warning("image reuse may result in subtly wrong code...\n");

  // ifdebug(1) pips_assert("initial dag ok", dag_consistent_p(initial));
  // if everything was removed by optimizations, there is nothing to do.
  if (dag_computation_count(initial)==0) return NIL;

  // work on a copy of the dag.
  dag dall = copy_dag(initial);
  list ld = NIL;
  set
    // current set of vertices to group
    current = set_make(set_pointer),
    // all vertices which are considered computed
    computed = set_make(set_pointer);

  do
  {
    list lcurrent = NIL, computables;
    set_clear(current);
    set_clear(computed);
    set_assign_list(computed, dag_inputs(dall));

    // GLOBAL
    pips_assert("erosion is clean", erosion==NULL);
    erosion = hash_table_make(hash_pointer, 0);
    dag_terapix_erosion(dall, erosion);

    // transitive closure
    while ((computables =
            get_computable_vertices(dall, computed, computed, current)))
    {
      ifdebug(7) {
        FOREACH(dagvtx, v, computables)
          dagvtx_dump(stderr, "computable", v);
      }

      gen_sort_list(computables,
                    (int(*)(const void*,const void*)) dagvtx_terapix_priority);

      // choose one while building the schedule?
      dagvtx first = DAGVTX(CAR(computables));
      gen_free_list(computables), computables = NIL;

      ifdebug(7)
        dagvtx_dump(stderr, "choice", first);

      set_add_element(current, current, first);
      set_add_element(computed, computed, first);
      lcurrent = gen_nconc(lcurrent, CONS(dagvtx, first, NIL));
    }

    // is there something?
    if (lcurrent)
    {
      // build extracted dag
      dag nd = make_dag(NIL, NIL, NIL);
      FOREACH(dagvtx, v, lcurrent)
      {
        pips_debug(7, "extracting node %" _intFMT "\n", dagvtx_number(v));
        dag_append_vertex(nd, copy_dagvtx_norec(v));
      }
      dag_compute_outputs(nd, NULL);
      dag_cleanup_other_statements(nd);

      ifdebug(7) {
        // dag_dump(stderr, "updated dall", dall);
        dag_dump(stderr, "pushed dag", nd);
      }

      // ??? hmmm... should not be needed?
      freia_hack_fix_global_ins_outs(initial, nd);

      // update global list of dags to return.
      ld = CONS(dag, nd, ld);

      // cleanup full dag for next round
      FOREACH(dagvtx, w, lcurrent)
        dag_remove_vertex(dall, w);

      gen_free_list(lcurrent), lcurrent = NIL;
    }
    hash_table_free(erosion), erosion = NULL;
  }
  while (set_size(current));

  free_dag(dall);
  return ld;
}

/* generate terapix code for this one dag, which should be already split.
 */
static void freia_trpx_compile_one_dag(
  string module,
  list /* of statements */ ls,
  dag d,
  string fname_fulldag,
  int n_split,
  int n_cut,
  set global_remainings,
  FILE * helper_file)
{
  ifdebug(4) {
    dag_consistency_asserts(d);
    dag_dump(stderr, "one_dag", d);
  }

  set remainings = set_make(set_pointer);
  set_append_vertex_statements(remainings, dag_vertices(d));

  // name_<number>_<split>[_<cut>]
  string fname_dag = strdup(cat(fname_fulldag, "_", itoa(n_split)));
  if (n_cut!=-1)
  {
    string s = strdup(cat(fname_dag, "_", itoa(n_cut)));
    free(fname_dag);
    fname_dag = s;
  }

  dag_dot_dump(module, fname_dag, d);

  // - output function in helper file
  list lparams = NIL;

  string_buffer code = string_buffer_make(true);
  freia_terapix_call(module, fname_dag, code, d, &lparams);
  string_buffer_to_file(code, helper_file);
  string_buffer_free(&code);

  // - and substitute its call...
  freia_substitute_by_helper_call(d, global_remainings, remainings,
                                  ls, fname_dag, lparams);

  // cleanup
  free(fname_dag), fname_dag = NULL;
}

/* would it seem interesting to split d?
 * @return the erosion up to which to split, or 0 of no split
 * should we also/instead consider the expected cost?
 */
static int cut_decision(dag d, hash_table erosion)
{
  int com_cost_per_row = get_int_property(trpx_dmabw_prop);
  int len, width, cost, nops, n, s, w, e;
  len = dag_terapix_measures(d, erosion, &width, &cost, &nops, &n, &s, &w, &e);
  int nins = gen_length(dag_inputs(d)), nouts = gen_length(dag_outputs(d));

  // if we assume that the imagelet size is quite large, say around 128
  // even with double buffers. The only reason to cut is because
  // of the erosion on the side which reduces the amount of valid data,
  // but there is really a point to do that only communications are still
  // masked by computations after splitting the dag...

  // first we compute a possible number of splits
  // computation cost = communication cost (in cycle per imagelet row)
  // communication cost = (nins + 2*width*n_splits + nouts) * cost_per_row
  // the width is taken as the expected number of images to extract and
  // reinject (hence 2*) if the dag is split.
  // this is really an approximation... indeed, nothing ensures that
  // the initial input is not still alive at the chosen cut?

  // for anr999 the gradient of depth 10 is just enough to cover the coms.
  // for lp, about 1(.2) split is suggested.

  double n_cuts = ((1.0*cost/com_cost_per_row)-nins-nouts)/(2.0*width);

  pips_debug(2, "cost=%d com_cost=%d nins=%d width=%d nouts=%d n_cuts=%f\n",
             cost, com_cost_per_row, nins, width, nouts, n_cuts);

  if (n_cuts < 1.0) return 0;

  // we also have to check that there is a significant erosion!
  // I first summarize the erosion to the max(n,s,e,w)
  // I could compute per direction, if necessary...
  int erode = n;
  if (s>erode) erode=s;
  if (e>erode) erode=e;
  if (w>erode) erode=w;

  // then we should decide...
  // there should be a *lot* of computations to amortize a split,
  // given that an erode/dilate costs about 15 cycles per row
  // there should be about 8 of them to amortize/hide one imagelet transfer,
  // whether as input or output.

  int cut = erode/((int)(n_cuts+1));
  return cut;
}

/* cut dag "d", possibly a subdag of "fulld", at "erosion" "cut"
 */
static dag cut_perform(dag d, int cut, hash_table erosion, dag fulld)
{
  pips_debug(2, "cutting with cut=%d\n", cut);
  pips_assert("something cut width", cut>0);

  // already warned while splitting on scalars
  // if (!single_image_assignement_p(initial))
    // well, it should work most of the time, so only a warning
    // pips_user_warning("image reuse may result in subtly wrong code...\n");

  set
    // current set of vertices to group
    current = set_make(set_pointer),
    // all vertices which are considered computed
    done = set_make(set_pointer);

  list lcurrent = NIL, computables;
  set_assign_list(done, dag_inputs(d));

  // transitive closure
  bool changed = true;
  while (changed &&
         (computables = get_computable_vertices(d, done, done, current)))
  {
    changed = false;
    FOREACH(dagvtx, v, computables)
    {
      // keep erosion up to cut
      // hmmm. what about \sigma_{d \in NSEW} erosion_d ?
      // would not work because the erosion only make sense if it is
      // the same for all imagelet, or said otherwise the erosion is
      // aligned to the worst case so that tiling can reasonnably take place.
      if ((((_int) hash_get(erosion, NORTH(v))) <= cut) &&
          (((_int) hash_get(erosion, SOUTH(v))) <= cut) &&
          (((_int) hash_get(erosion, EAST(v))) <= cut) &&
          (((_int) hash_get(erosion, WEST(v))) <= cut))
      {
        set_add_element(current, current, v);
        set_add_element(done, done, v);
        lcurrent = gen_nconc(lcurrent, CONS(dagvtx, v, NIL));
        changed = true;
      }
    }

    // cleanup
    gen_free_list(computables), computables = NIL;
  }

  pips_assert("some vertices where extracted", lcurrent!=NIL);

  // build extracted dag
  dag nd = make_dag(NIL, NIL, NIL);
  FOREACH(dagvtx, v, lcurrent)
  {
    // pips_debug(7, "extracting node %" _intFMT "\n", dagvtx_number(v));
    dag_append_vertex(nd, copy_dagvtx_norec(v));
  }
  dag_compute_outputs(nd, NULL);
  dag_cleanup_other_statements(nd);

  // cleanup full dag
  FOREACH(dagvtx, v, lcurrent)
    dag_remove_vertex(d, v);

  // ??? should not be needed?
  freia_hack_fix_global_ins_outs(fulld, nd);
  freia_hack_fix_global_ins_outs(fulld, d);

  ifdebug(1)
  {
    dag_consistency_asserts(nd);
    dag_consistency_asserts(d);
  }

  // cleanup
  gen_free_list(lcurrent), lcurrent = NIL;
  set_free(done);
  set_free(current);
  return nd;
}

/* do compile a list of statements for terapix
 * @param module, current module (function) name
 * @param ls, list of statements taken from the sequence
 * @param occs, occurences of images (image -> set of statements)
 * @param helper_file, file to which code is to be generated
 * @param number, number of this statement sequence in module
 */
void freia_trpx_compile_calls
(string module,
 list /* of statements */ ls,
 hash_table occs,
 FILE * helper_file,
 int number)
{
  // build DAG for ls
  pips_debug(3, "considering %d statements\n", (int) gen_length(ls));
  pips_assert("some statements", ls);

  list added_stats = NIL;
  dag fulld = build_freia_dag(module, ls, number, occs, &added_stats);

  string fname_fulldag = strdup(cat(module, HELPER, itoa(number)));

  // First, split only on scalar deps...
  // is it that simple? NO!
  // consider A -> B -> s -> C -> D
  //           \-> E -> F />
  // then ABEF / CD is chosen
  // although ABE / FCD and AB / EFCD would be also possible...
  list ld = split_dag_on_scalars(fulld);

  pips_debug(4, "dag initial split in %d dags\n", (int) gen_length(ld));

  string dag_cut = get_string_property(trpx_dag_cut);
  pips_assert("valid cutting strategy", trpx_dag_cut_is_valid(dag_cut));

  // globally remaining statements
  set global_remainings = set_make(set_pointer);
  set_assign_list(global_remainings, ls);

  int n_split = 0;
  FOREACH(dag, d, ld)
  {
    if (trpx_dag_cut_none_p(dag_cut))
    {
      // direct handling of the dag
      freia_trpx_compile_one_dag(module, ls, d, fname_fulldag, n_split, -1,
                                 global_remainings, helper_file);
    }
    else if (trpx_dag_cut_compute_p(dag_cut))
    {
      // try split dag into subdags with a rough computed strategy
      hash_table erosion = hash_table_make(hash_pointer, 0);
      int cut, n_cut = 0;

      // what about another strategy?
      // I can try every possible cuts and chose the best one,
      // that is to stop as soon as computation cost > communication cost?
      // or when costs are quite balanced in all cuts?
      // dag cutting strategy prop = none/computed/optimized?

      while ((cut = cut_decision(d, erosion)))
      {
        dag dc = cut_perform(d, cut, erosion, fulld);
        // generate code for cut
        freia_trpx_compile_one_dag(module, ls, dc, fname_fulldag, n_split,
                                   n_cut++, global_remainings, helper_file);
        // cleanup
        free_dag(dc);
        hash_table_clear(erosion);
      }
      freia_trpx_compile_one_dag(module, ls, d, fname_fulldag, n_split,
                                 n_cut++, global_remainings, helper_file);
      hash_table_free(erosion);
    }
    else if (trpx_dag_cut_enumerate_p(dag_cut))
      pips_internal_error("not implemented yet");
    else
      pips_internal_error("cannot get there");

    n_split++;
  }

  freia_insert_added_stats(ls, added_stats);
  added_stats = NIL;

  // full cleanup
  set_free(global_remainings), global_remainings = NULL;
  free(fname_fulldag), fname_fulldag = NULL;
  free_dag(fulld);
  FOREACH(dag, dc, ld)
    free_dag(dc);
  gen_free_list(ld);
}
