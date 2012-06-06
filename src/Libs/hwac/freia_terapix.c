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
#include "effects.h"
#include "ri-util.h"
#include "effects-util.h"
#include "properties.h"

#include "freia.h"
#include "freia_spoc_private.h"
#include "hwac.h"

/********************************************************** TERAPIX ANALYSES */

/* @return the dead vertices (their output is dead) after computing v in d.
 * ??? should it take care that an output node is never dead?
 */
static void compute_dead_vertices
  (set deads, const set computed, const dag d, const dagvtx v)
{
  list preds = dag_vertex_preds(d, v);
  set futured_computed = set_dup(computed);
  set_add_element(futured_computed, futured_computed, v);
  FOREACH(dagvtx, p, preds)
    if (// !gen_in_list_p(p, dag_outputs(d)) &&
        list_in_set_p(dagvtx_succs(p), futured_computed))
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
        bool k00, k10, k20, k01, k21, k02, k12, k22;
        intptr_t i = 0;
        k00 = expression_integer_value(EXPRESSION(CAR(iargs)), &i) && i==0;
        iargs = CDR(iargs);
        k10 = expression_integer_value(EXPRESSION(CAR(iargs)), &i) && i==0;
        iargs = CDR(iargs);
        k20 = expression_integer_value(EXPRESSION(CAR(iargs)), &i) && i==0;
        iargs = CDR(iargs);
        k01 = expression_integer_value(EXPRESSION(CAR(iargs)), &i) && i==0;
        iargs = CDR(iargs);
        //bool k11 = expression_integer_value(EXPRESSION(CAR(iargs)), &i) && i==0;
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
  if (freia_convolution_p(v)) // convolution special handling...
  {
    _int width, height;
    if (freia_convolution_width_height(v, &width, &height, false))
    {
      w+=width/2;
      e+=width/2;
      n+=height/2;
      s+=height/2;
    }
    // else simply ignore, should not be used anyway...
  }
  else if (api->terapix.north) // erode & dilate
  {
    bool north = true, south = true, west = true, east = true;
    erosion_optimization(v, &north, &south, &west, &east);
    if (north) n += api->terapix.north;
    if (south) s += api->terapix.south;
    if (west) w += api->terapix.west;
    if (east) e += api->terapix.east;
  }

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
  while ((lv = dag_computable_vertices(d, processed, processed, processed)))
  {
    dlength++;
    int level_width = 0;
    FOREACH(dagvtx, v, lv)
    {
      const freia_api_t * api = dagvtx_freia_api(v);
      if (freia_convolution_p(v)) // special handling...
      {
        _int w, h;
        if (freia_convolution_width_height(v, &w, &h, false))
          dcost += 8+(api->terapix.cost*w*h); // hmmm? 3x3 is 35?
        else
          dcost += 35; // au pif 3x3
      }
      else
        dcost += api->terapix.cost;
      // only count non null operations
      if (api->terapix.cost && api->terapix.cost!=-1) dnops ++;
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

/* @return the list of inputs to vertex v as imagelet numbers.
 */
static list /* of ints */ dag_vertex_pred_imagelets
  (const dag d, const dagvtx v, const hash_table allocation)
{
  list limagelets = NIL;
  FOREACH(entity, img, vtxcontent_inputs(dagvtx_content(v)))
  {
    dagvtx prod = dagvtx_get_producer(d, v, img, 0);
    pips_assert("some producer found!", prod!=NULL);
    limagelets =
      gen_nconc(limagelets,
                CONS(int, (int)(_int) hash_get(allocation, prod), NIL));
  }
  return limagelets;
}

/************************************************** GLOBAL MEMORY MANAGEMENT */

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
      bool ok = true;
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

/********************************** TERAPIX CODE GENERATION HELPER FUNCTIONS */

/* Return the first/last available imagelet, or create one if necessary
 * This ensures that the choice is deterministic.
 * Moreover, as first numbers are IO imagelets, this help putting outputs
 * in the right imagelet so as to avoid additionnal copies, if possible.
 */
static _int select_imagelet(set availables, int * nimgs, bool first)
{
  ifdebug(8) {
    pips_debug(8, "selecting first=%s\n", bool_to_string(first));
    set_fprint(stderr, "availables", availables, (gen_string_func_t) itoa);
  }

  _int choice = 0; // zero means no choice yet
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
  pips_debug(8, "choice is %"_intFMT"\n", choice);
  return choice;
}

#define IMG_PTR "imagelet_"
#define RED_PTR "reduction_"

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

/* copy some operator *parameters* in the global ram (aka gram).
 * the coordinates used are  (x_<name>, y_<name>).
 */
static void gram_param
  (string_buffer code, string_buffer decl,
   string name, dagvtx v, hash_table hparams,
   int width, int height, bool is_kernel, bool * used)
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
  if (is_kernel)
  {
    sb_cat(code,
           "  for(i=0; i<", itoa(size), "; i++)\n"
           "    p_", name, "[i] = ", p1, "[i];\n");
  }
  else
  {
    switch (size)
    {
    case 1: // constant
      sb_cat(code, "  p_", name, "[0] = ", p1, ";\n");
      break;
    case 3: // threshold min/max/bin
      sb_cat(code, "  p_", name, "[0] = ", p1, ";\n");
      sb_cat(code, "  p_", name, "[1] = ",
             hash_get(hparams, EXPRESSION(CAR(CDR(largs)))), ";\n");
      sb_cat(code, "  p_", name, "[2] = ",
             hash_get(hparams, EXPRESSION(CAR(CDR(CDR(largs))))), ";\n");
      break;
    default:
      pips_internal_error("unexpected gram size");
    }
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
   int op,             // operation number
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
    case 3: // convolution or threshold
      if (freia_convolution_p(v)) // convolution special case
      {
        _int w, h;
        freia_convolution_width_height(v, &w, &h, true);
        gram_param(code, decl, name, v, hparams, w, h, true, used);
      }
      else // threshold
        gram_param(code, decl, name, v, hparams, 3, 1, false, used);
      break;
    case 1: // kernel or operation with a constant
      if (api->terapix.north) // let us say it is a kernel...
        gram_param(code, decl, name, v, hparams, 3, 3, true, used);
      else
        gram_param(code, decl, name, v, hparams, 1, 1, false, used);
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
    int img1 = INT(CAR(ins)), img2 = INT(CAR(CDR(ins)));
    terapix_mcu_img(code, op, "xmin1", api->terapix.reverse? img2: img1);
    terapix_mcu_int(code, op, "ymin1", 0);
    terapix_mcu_img(code, op, "xmin2", api->terapix.reverse? img1: img2);
    terapix_mcu_int(code, op, "ymin2", 0);
    terapix_mcu_img(code, op, "xmin3", out);
    terapix_mcu_int(code, op, "ymin3", 0);
    // ??? needed for replace const... although arg 3 is used already
    // replace_const special argument management is handled directly elsewhere
    // terapix_gram_management(code, decl, op, api, v, hparams, used);
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
    terapix_mcu_img(code, op, "xmin1", out);
    terapix_mcu_int(code, op, "ymin1", 0);
    terapix_gram_management(code, decl, op, api, v, hparams, used);
    break;
  default:
    pips_internal_error("unexpected number of input images");
  }
  terapix_mcu_val(code, op, "iter1", "TERAPIX_PE_NUMBER");
  terapix_mcu_val(code, op, "iter2", "imagelet_size");
  if (freia_convolution_p(v)) // convolution special case hack
  {
    _int w, h;
    freia_convolution_width_height(v, &w, &h, true);
    // ??? should I use the parameters?
    // ??? or check their values?
    // ??? or remove them from the list as they are inlined?
    terapix_mcu_int(code, op, "iter3", (int) w);
    terapix_mcu_int(code, op, "iter4", (int) h);
  }
  else
  {
    terapix_mcu_val(code, op, "iter3", "0");
    terapix_mcu_val(code, op, "iter4", "0");
  }
  terapix_mcu_val(code, op, "addrStart", api->terapix.ucode);
}

/* @brief initialize a few rows at mem address with value val
 */
static void terapix_init_row(
  string_buffer decl,
  string_buffer code,
  string base,
  string suff,
  string mem,
  int nrow,
  string val,
  bool * used)
{
  // get one memory cell for the value
  int x = 0, y = 0;
  terapix_gram_allocate(used, 1, 1, &x, &y);

  // operation name
  string name = strdup(cat(base, "_", suff));

  // set the constant
  sb_cat(decl, "  // operation ", name, " initialization\n"
               "  int16_t p_", name, "[1];\n");
  sb_cat(decl, "  const int32_t x_", name, " = ", itoa(x), ";\n");
  sb_cat(decl, "  const int32_t y_", name, " = ", itoa(y), ";\n");

  sb_cat(code, "  // initializing  ", name, "\n"
               "  p_", name, "[0] = ", val, ";\n"
               "  gram.xoffset = x_", name, ";\n"
               "  gram.yoffset = y_", name, ";\n"
               "  gram.width = 1;\n"
               "  gram.height = 1;\n"
               "  gram.params = p_", name, ";\n"
               "  freia_mg_write_dynamic_param(&dyn_param);\n");

  // call the initialization
  sb_cat(code,
         "  // initialize memory for operation ", name, "\n"
         "  mem_init.xmin1 = ", mem, ";\n"
         "  mem_init.ymin1 = 0;\n"
         "  mem_init.xmin2 = 0;\n"
         "  mem_init.ymin2 = 0;\n"
         "  mem_init.xmin3 = 0;\n"
         "  mem_init.ymin3 = 0;\n"
         "  mem_init.iter1 = TERAPIX_PE_NUMBER;\n"
         "  mem_init.iter2 = ", itoa(nrow),";\n"
         "  mem_init.iter3 = 0;\n"
         "  mem_init.iter4 = 0;\n"
         "  mem_init.addrStart = TERAPIX_UCODE_SET_CONST;\n"
         "  param.size = sizeof(terapix_mcu_macrocode); // not used?\n"
         "  param.raw = (void*) (&mem_init);\n"
         "  ret |= freia_mg_work(&param);\n"
         "  ret |= freia_mg_end_work();\n");

  // cleanup
  free(name);
}

/* @brief initialize the memory at addr depending on the operation to perform
 * @param decl, added declarations are put there
 * @param body, generated code is put there
 * @param nop, current operation number
 * @param mem, memory symbolic x address
 * @param api, freia operation
 * @param used, current use of Global RAM (gram)
 */
static void terapix_initialize_memory(
  string_buffer decl,
  string_buffer body,
  int nop,
  string mem,
  const freia_api_t * api,
  bool * used)
{
  string op = api->compact_name;
  pips_assert("operation is a measure",
              same_string_p(op, "min") || same_string_p(op, "min!") ||
              same_string_p(op, "max") || same_string_p(op, "max!") ||
              same_string_p(op, "vol"));
  string sop = strdup(itoa(nop));

  // INT16 should be a property?

  if (same_string_p(op, "min") || same_string_p(op, "min!"))
    terapix_init_row(decl, body, sop, "val", mem, 1, "INT16_MAX", used);
  if (same_string_p(op, "max") || same_string_p(op, "max!"))
    terapix_init_row(decl, body, sop, "val", mem, 1, "INT16_MIN", used);
  if (same_string_p(op, "min!") || same_string_p(op, "max!"))
  {
    string memp1 = strdup(cat(mem,"+1"));
    terapix_init_row(decl, body, sop, "loc", memp1, 4, "0", used);
    free(memp1);
  }
  if (same_string_p(op, "vol"))
    terapix_init_row(decl, body, sop, "val", mem, 2, "0", used);

  free(sop);
}

/* @brief generate reduction extraction code
 */
static void terapix_get_reduction(
  string_buffer decl,
  string_buffer tail,
  int n_op,
  string mem,
  const freia_api_t * api)
{
  pips_assert("some results are expected", api->arg_misc_out>0);
  string sop = strdup(itoa(n_op));
  // I do not understand the underlying logic of these values
  string width = api->arg_misc_out==3? "5": "1";
  sb_cat(decl,
         "  // array for reduction ", sop, " extraction\n"
         "  int32_t red_", sop, "[", itoa(api->arg_misc_out), "];\n");
  sb_cat(tail,
         "  redter.xres = ", mem, ";\n"
         "  redter.yres = 0;\n"
         "  redter.width = ", width, ";\n"
         "  redter.height = TERAPIX_PE_NUMBER;\n"
         "  redter.result = (void*) red_", sop, ";\n"
         "  redter.macroid = ", api->terapix.ucode, ";\n"
         // just gessing that there must be a first input image
         // ??? we assume that all image are of the same size?!
         "  redter.imgwidth = i0->width;\n"
         "  redter.imgheight = i0->height;\n"
         "  redter.subimgwidth = TERAPIX_PE_NUMBER;\n"
         "  redter.subimgheight = imagelet_size;\n"
         "\n"
         "  ret |= freia_cg_read_reduction_results(&redres);\n"
         "\n");
  free(sop);
}

/*************************************************** TERAPIX CODE GENERATION */

/* generate a terapix call for dag thedag.
 * the memory allocation is managed here.
 * however this function is dumb, the scheduling is just inherited as is...
 * @return number of output images...
 */
static _int freia_terapix_call
  (const string module,
   const string fname_dag,
   string_buffer code,
   dag thedag,
   list /* of expression */ *params)
{
  // total number of imagelets used for computing the dag
  // will be updated later, implicitely derived from the scheduling
  int n_imagelets = 0;
  // number of input images
  int n_ins = gen_length(dag_inputs(thedag));
  // number of output images
  int n_outs = gen_length(dag_outputs(thedag));
  // number of needed double buffers for I/Os.
  // this is also the number of I/O images
  int n_double_buffers;

  if (trpx_overlap_io_p())
    n_double_buffers = n_ins+n_outs;
  else
    n_double_buffers = (n_ins>n_outs)? n_ins: n_outs; // max(#ins, #outs)

  pips_assert("some I/O images", n_double_buffers>0);

  // the memory will be decremented for "measures" data (reductions),
  // and then divided among imagelets
  int available_memory = get_int_property(trpx_mem_prop);

  string_buffer
    head = string_buffer_make(true),
    decl = string_buffer_make(true),
    init = string_buffer_make(true),
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

// integer property to string
#define ip2s(n) itoa(get_int_property(n))

  // show stats in function's comments
  sb_cat(head, "\n"
               "/* FREIA terapix helper function for module ", module, "\n");
  sb_cat(head, " *\n");
  // show terapix code generation parameters
  sb_cat(head, " * RAMPE    = ", ip2s(trpx_mem_prop), "\n");
  sb_cat(head, " * NPE      = ", ip2s(trpx_npe_prop), "\n");
  sb_cat(head, " * DMA BW   = ", ip2s(trpx_dmabw_prop), "\n");
  sb_cat(head, " * GRAM W   = ", ip2s(trpx_gram_width), "\n");
  sb_cat(head, " * GRAM H   = ", ip2s(trpx_gram_height), "\n");
  sb_cat(head, " * DAG CUT  = ", get_string_property(trpx_dag_cut), "\n");
  sb_cat(head, " * OVERLAP  = ", bool_to_string(trpx_overlap_io_p()), "\n");
  sb_cat(head, " * IMAGE H  = ", ip2s("FREIA_IMAGE_HEIGHT"), "\n");
  sb_cat(head, " * MAX SIZE = ", ip2s(trpx_max_size), "\n");
  sb_cat(head, " *\n");
  // show dag statistics
  sb_cat(head, " * ", itoa(n_ins), " input image", n_ins>1? "s": "");
  sb_cat(head, ", ", itoa(n_outs), " output image", n_outs>1? "s": "", "\n");
  sb_cat(head, " * ", itoa(nops), " image operations in dag\n");
  sb_cat(head, " * dag length is ", itoa(length));
  sb_cat(head, ", dag width is ", itoa(width), "\n");
  sb_cat(head, " * costs in cycles per imagelet row:\n");
  sb_cat(head, " * - computation: ", itoa(cost), "\n");
  // number of transfers depends on overlapping
  int n_trs = trpx_overlap_io_p()? (n_ins>n_outs? n_ins: n_outs): n_ins+n_outs;
  sb_cat(head, " * - communication: ", itoa(comm*n_trs), "\n");
  sb_cat(head, " */\n");

  // generate function declaration
  sb_cat(head, "freia_status ", fname_dag, "(");
  for (int i = 0; i<n_outs; i++)
    sb_cat(head, nargs++? ",": "", "\n  " FREIA_IMAGE "o", itoa(i));
  for (int i = 0; i<n_ins; i++)
    sb_cat(head, nargs++? ",": "", "\n  const " FREIA_IMAGE "i", itoa(i));
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
         "  int i;\n" // not always used...
         "  freia_status ret = FREIA_OK;\n"
         "  // data structures for reductions\n"
         "  terapix_mcu_macrocode mem_init;\n"
         "  freia_reduction_results redres;\n"
         "  terapix_reduction redter;\n"
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
         "  // dyn_param contents\n"
         "  dyn_param.raw = &gram;\n"
         "  dyn_param.size = sizeof(terapix_gram);\n"
         "\n"
         "  // redres contents\n"
         "  redres.raw = (void*) &redter;\n"
         "  redres.size = sizeof(terapix_reduction);\n"
         "\n");

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

  // currently available imagelets
  set avail_img = set_make(set_pointer);

  // output images are the first ones when I/O comms overlap
  if (trpx_overlap_io_p())
    while (n_imagelets<n_outs)
      set_add_element(avail_img, avail_img, (void*) (_int) ++n_imagelets);

  if (n_ins)
  {
    // ??? they should be given in the order of the arguments
    // when calling the runtime function.
    int n = 0;
    sb_cat(dbio, "\n  // inputs:\n");
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
    sb_cat(dbio, "\n  // no input\n\n");
  }

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

    // if inplace, append freed images to availables
    if (api->terapix.inplace)
    {
      SET_FOREACH(dagvtx, v, deads)
      {
        // but keep intermediate output images!
        if (!gen_in_list_p(v, dag_outputs(thedag)))
        {
          _int img = (_int) hash_get(allocation, v);
          if (img<0) img=-img;
          set_add_element(avail_img, avail_img, (void*) img);
        }
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
      // SELECT one available imagelet
      // if none is available, a new one is implicitely created
      choice = select_imagelet(avail_img, &n_imagelets, is_output);
      sb_cat(body, " -> ", itoa((int) choice));
      // there is a subtlety here, if no I/O image was available
      // then a copy will have to be inserted later on, see "PANIC".
      if (choice<=n_double_buffers) choice = -choice;
      hash_put(allocation, current, (void*) choice);
    }
    sb_cat(body, "\n");

    // update helper call arguments...
    *params = gen_nconc(*params,
                        freia_extract_params(opid, call_arguments(c),
                                             head, NULL, hparams, &nargs));

    // special case for replace_const, which needs a 4th argument
    if (same_string_p(api->compact_name, ":"))
    {
      sb_cat(body, "  // *special* set parameter for replace_const\n");
      terapix_mcu_int(body, n_ops, "xmin1", 0);
      terapix_mcu_int(body, n_ops, "ymin1", 0);
      terapix_mcu_int(body, n_ops, "xmin2", 0);
      terapix_mcu_int(body, n_ops, "ymin2", 0);
      terapix_gram_management(body, decl, n_ops, api, current, hparams, used);
      terapix_mcu_val(body, n_ops, "iter1", "TERAPIX_PE_NUMBER");
      terapix_mcu_int(body, n_ops, "iter2", 0);
      terapix_mcu_int(body, n_ops, "iter3", 0);
      terapix_mcu_int(body, n_ops, "iter4", 0);
      terapix_mcu_val(body, n_ops, "addrStart",
                      "TERAPIX_UCODE_SET_CONST_RAMREG");

      sb_cat(body, "  // now take care of actual operation\n");
      n_ops++;
    }

    if (api->terapix.memory)
    {
      string sop = strdup(itoa(n_ops));
      // reserve the necessary memory at the end of the segment
      available_memory -= api->terapix.memory;
      string mem = strdup(cat(RED_PTR, sop));
      sb_cat(init, "  int ", mem, " = ", itoa(available_memory), ";\n");

      // initialize the memory based on the measure operation
      terapix_initialize_memory(decl, body, n_ops, mem, api, used);

      // imagelet computation
      sb_cat(body, "  // set measure ", api->compact_name, " at ", mem, "\n");
      terapix_mcu_val(body, n_ops, "xmin2", mem);
      terapix_mcu_val(body, n_ops, "ymin2", "0");

      // should not be used, but just in case...
      terapix_mcu_val(body, n_ops, "xmin3", "0");
      terapix_mcu_val(body, n_ops, "ymin3", "0");

      // extraction
      sb_cat(tail, "  // get measure ", api->compact_name,
             " result from ", mem, "\n");
      terapix_get_reduction(decl, tail, n_ops, mem, api);

      sb_cat(tail, "  // assign reduction parameter",
             api->arg_misc_out>1? "s":"", "\n");
      int i = 0;
      FOREACH(expression, arg, freia_get_vertex_params(current))
      {
        string var = (string) hash_get(hparams, arg);
        // hmmm, kind of a hack to get the possibly needed cast
        string cast = strdup(api->arg_out_types[i]);
        string space = strchr(cast, ' ');
        if (space) *space = '\0';
        sb_cat(tail, "  *", var, " = (", cast, ") "
               "red_", sop, "[", itoa(i), "];\n");
        i++;
        free(cast);
      }
      free(mem);
      free(sop);
    }

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
        // but keep intermediate output images!
        if (!gen_in_list_p(v, dag_outputs(thedag)))
        {
          _int img = (_int) hash_get(allocation, v);
          if (img<0) img=-img;
          set_add_element(avail_img, avail_img, (void*) img);
        }
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
      // when not overlapping, any I/O image is fine
      // when overlapping, must be one of the first
      //   because the later ones are used in parallel as inputs
      if ((!trpx_overlap_io_p() && oimg>n_double_buffers) ||
          (trpx_overlap_io_p() && oimg>n_outs))
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
  sb_cat(decl, "  // flip flop macro code and I/Os\n");
  sb_cat(decl, "  terapix_mcu_macrocode mcu_macro[2][", itoa(n_ops), "];\n");
  if (n_ins)
    sb_cat(decl, "  terapix_tile_info tile_in[2][", itoa(n_ins), "];\n");
  if (n_outs)
    sb_cat(decl, "  terapix_tile_info tile_out[2][", itoa(n_outs), "];\n");

  // computed values
  sb_cat(decl, "  // imagelets definitions:\n");
  sb_cat(decl, "  // - ", itoa(n_imagelets), " computation imagelets\n");
  sb_cat(decl, "  // - ", itoa(n_double_buffers), " double buffer imagelets\n");

  // we may optimize the row size for a target image height, if available
  int image_height = FREIA_DEFAULT_HEIGHT;
  int vertical_border = n>s? n: s;
  int max_computed_size = imagelet_rows-2*vertical_border;
  // this is really a MAXIMUM available size that can be set from outside
  int max_size = get_int_property(trpx_max_size);

  if (image_height==0)
  {
    // what about vol(cst())?
    pips_assert("at least one image is needed!", n_ins||n_outs);
    // dynamic adjustment of the imagelet size
    sb_cat(decl,
      "  // dynamic optimal imagelet size computation\n"
      "  // this formula must match what the scheduler does!\n"
      "  int vertical_border = ", itoa(vertical_border), ";\n"
      // use first input image for the reference size, or default to output
      "  int image_height = ", n_ins? "i": "o", "0->heightWa;\n");
    sb_cat(decl,
      "  int max_computed_size = ", itoa(max_computed_size), ";\n"
      "  int n_tiles = (image_height+max_computed_size-1)/max_computed_size;\n"
      "  int imagelet_size =\n"
      "        ((image_height+n_tiles-1)/n_tiles)+2*vertical_border;\n");
    if (max_size)
    {
      sb_cat(decl,
             "  // max imagelet size requested..."
             "  int max_size = ", itoa(max_size), ";\n"
             "  if (imagelet_size>max_size)\n"
             "    imagelet_size = max_size;\n");
    }
  }
  else // assume the provided image_height
  {
    // we adjust statically the imagelet size so that we avoid recomputing
    // pixels... the formula must match whatever the scheduler does!
    // ??? hmmm... only for inner tiles
    // #tiles is ceil(height/computed)
    int n_tiles = (image_height+max_computed_size-1)/max_computed_size;
    // now we compute back the row size
    int optim_rows = ((image_height+n_tiles-1)/n_tiles)+2*vertical_border;
    pips_assert("optimized row size lower than max row size",
                optim_rows<=imagelet_rows && optim_rows>0);
    // now we set the value directly
    sb_cat(decl, "  // imagelet max size: ", itoa(imagelet_rows), "\n");
    imagelet_rows = optim_rows;

    // the runtime can use imagelet_rows or less
    sb_cat(decl, "  int imagelet_size = ",
           itoa(max_size?
                // max_size is defined, may use it if smaller than computed size
                (max_size<imagelet_rows? max_size: imagelet_rows):
                // max_size is not defined
                imagelet_rows), ";\n");
  }

  // generate imagelet pointers
  for (int i=1; i<=total_imagelets; i++)
  {
    sb_cat(decl, "  int " IMG_PTR, itoa(i), " = ");
    sb_cat(decl, itoa(imagelet_rows * (i-1)), ";\n");
  }
  // append reduction memory pointers
  sb_cat(decl, "\n");

  if (string_buffer_size(init)>0)
  {
    sb_cat(decl, "  // memory for reductions\n");
    string_buffer_append_sb(decl, init);
    sb_cat(decl, "\n");
  }
  string_buffer_free(&init);

  // generate imagelet double buffer pointers
  // sb_cat(dbio, "  // double buffer management:\n");
  sb_cat(decl, "  // double buffer assignment\n");
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
  sb_cat(body, "  // imagelet erosion for the computation\n");
  sb_cat(body, "  mcu_instr.borderTop    = ", itoa(n), ";\n");
  sb_cat(body, "  mcu_instr.borderBottom = ", itoa(s), ";\n");
  sb_cat(body, "  mcu_instr.borderLeft   = ", itoa(w), ";\n");
  sb_cat(body, "  mcu_instr.borderRight  = ", itoa(e), ";\n");
  sb_cat(body, "  mcu_instr.imagelet_height = imagelet_size;\n"
               "  mcu_instr.imagelet_width  = TERAPIX_PE_NUMBER;\n"
               "\n");

  sb_cat(body, "  // outputs\n"
               "  mcu_instr.nbout = ", itoa(n_outs), ";\n");
  if (n_outs)
    sb_cat(body,
           "  mcu_instr.out0 = tile_out[0];\n"
           "  mcu_instr.out1 = tile_out[1];\n");
  else
    sb_cat(body,
           "  mcu_instr.out0 = NULL;\n"
           "  mcu_instr.out1 = NULL;\n");

  sb_cat(body, "\n"
               "  // inputs\n"
               "  mcu_instr.nbin = ", itoa(n_ins), ";\n");
  if (n_ins)
    sb_cat(body,
           "  mcu_instr.in0 = tile_in[0];\n"
           "  mcu_instr.in1 = tile_in[1];\n");
  else
    sb_cat(body,
           "  mcu_instr.in0 = NULL;\n"
           "  mcu_instr.in1 = NULL;\n");

  sb_cat(body,
         "\n"
         "  // actual instructions\n"
         "  mcu_instr.nbinstr = ", itoa(n_ops), ";\n"
         "  mcu_instr.instr0   = mcu_macro[0];\n"
         "  mcu_instr.instr1   = mcu_macro[1];\n");

  // tell about imagelet size
  // NOTE: the runtime *MUST* take care of possible in/out aliasing
  sb_cat(body,
         "\n"
         "  // call terapix runtime\n"
         "  param.size = -1; // not used\n"
         "  param.raw = (void*) &mcu_instr;\n"
         "  ret |= freia_cg_template_process(&param");
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

  return n_outs;
}

/******************************************************************* ONE DAG */

/* generate terapix code for this one dag, which should be already split.
 * return the statement number of the helper insertion
 */
static int freia_trpx_compile_one_dag(
  string module,
  list /* of statements */ ls,
  dag d,
  string fname_fulldag,
  int n_split,
  int n_cut,
  set global_remainings,
  FILE * helper_file,
  set helpers,
  int stnb,
  hash_table signatures)
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

  dag_dot_dump(module, fname_dag, d, NIL, NIL);

  // - output function in helper file
  list lparams = NIL;

  string_buffer code = string_buffer_make(true);
  _int nout = freia_terapix_call(module, fname_dag, code, d, &lparams);
  string_buffer_to_file(code, helper_file);
  string_buffer_free(&code);

  // - and substitute its call...
  stnb = freia_substitute_by_helper_call(d, global_remainings, remainings,
                                         ls, fname_dag, lparams, helpers, stnb);

  // record (simple) signature
  hash_put(signatures, local_name_to_top_level_entity(fname_dag), (void*) nout);

  // cleanup
  free(fname_dag), fname_dag = NULL;

  return stnb;
}

/************************************************** TERAPIX DAG SCALAR SPLIT */

/* fill in erosion hash table from dag d.
 */
static void dag_terapix_erosion(const dag d, hash_table erosion)
{
  int i = 0;
  dag_terapix_measures(d, erosion, &i, &i, &i, &i, &i, &i, &i);
}

/* global variable used by the dagvtx_terapix_priority function,
 * because qsort does not allow to pass some descriptor.
 */
static hash_table erosion = NULL;

static void dag_terapix_reset_erosion(const dag d)
{
  pips_assert("erosion is allocated", erosion!=NULL);
  hash_table_clear(erosion);
  dag_terapix_erosion(d, erosion);
}

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
    // non implemented stuff
    if (!freia_aipo_terapix_implemented(a1))
      result = 1, why = "impl";
    else if (!freia_aipo_terapix_implemented(a2))
      result = -1, why = "impl";
    // scalars operations first to remove (scalar) dependences
    else if (vtxcontent_optype(c1)==spoc_type_oth)
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

/* @brief whether vertex is not implemented in terapix
 */
static bool not_implemented(dagvtx v)
{
  if (freia_convolution_p(v)) // special case
  {
    // skip if parametric
    _int w, h;
    return !freia_convolution_width_height(v, &w, &h, false);
  }
  return !freia_aipo_terapix_implemented(dagvtx_freia_api(v));
}

/* @brief whether dag is not implemented in terapix
 */
static bool terapix_not_implemented(dag d)
{
  FOREACH(dagvtx, v, dag_vertices(d))
    if (not_implemented(v))
      return true;
  return false;
}

/* @brief choose a vertex, avoiding non combinable stuff if the list is started
 */
static dagvtx choose_terapix_vertex(const list lv, bool started)
{
  pips_assert("list contains vertices", lv);
  if (started)
  {
    FOREACH(dagvtx, v, lv)
      if (!not_implemented(v))
        return v;
  }
  // just return the first vertex
  return DAGVTX(CAR(lv));
}

/*********************************************************** TERAPIX DAG CUT */

/* would it seem interesting to split d?
 * @return the erosion up to which to split, or 0 of no split
 * should we also/instead consider the expected cost?
 */
static int cut_decision(dag d, hash_table erosion)
{
  int com_cost_per_row = get_int_property(trpx_dmabw_prop);
  int  width, cost, nops, n, s, w, e;
  (void)dag_terapix_measures(d, erosion, &width, &cost, &nops, &n, &s, &w, &e);
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

  // compute number of cuts, that is the number of amortizable load/store
  // ??? maybe I should incorporate a margin?
  double n_cuts;

  // please note that these formula are somehow approximated and the results
  // may be proved wrong.
  if (trpx_overlap_io_p())
  {
    // number of image to communicate is MAX(#in,#out)
    int nimgs = nins>nouts? nins: nouts;
    // the overhead of a cut is one transfer
    n_cuts = ((1.0*cost/com_cost_per_row)-nimgs)/(1.0*width);
  }
  else
    n_cuts = ((1.0*cost/com_cost_per_row)-nins-nouts)/(2.0*width);

  pips_debug(2, "cost=%d com_cost=%d nins=%d width=%d nouts=%d n_cuts=%f\n",
             cost, com_cost_per_row, nins, width, nouts, n_cuts);

  if (n_cuts < 1.0) return 0;

  // we also have to check that there is a significant erosion!
  // I first summarize the erosion to the max(n,s,e,w)
  // grrr... C really lacks a stupid max/min function varyadic!
  // I could compute per direction, if necessary...
  int erode = n;
  if (s>erode) erode=s;
  if (e>erode) erode=e;
  if (w>erode) erode=w;

  // then we should decide...
  // there should be enough  computations to amortize a split,
  // given that an erode/dilate costs about 15 cycles per row
  // there should be about 2 of them to amortize/hide one imagelet transfer,
  // whether as input or output.

  int cut = erode/((int)(n_cuts+1));
  return cut;
}

/* cut dag "d", possibly a subdag of "fulld", at "erosion" "cut"
 */
static dag cut_perform(dag d, int cut, hash_table erodes, dag fulld,
                       const set output_images)
{
  pips_debug(2, "cutting with cut=%d\n", cut);
  pips_assert("something cut width", cut>0);

  set
    // current set of vertices to group
    current = set_make(set_pointer),
    // all vertices which are considered computed
    done = set_make(set_pointer);

  list lcurrent = NIL, computables;
  set_assign_list(done, dag_inputs(d));

  // GLOBAL
  pips_assert("erosion is clean", erosion==NULL);
  erosion = hash_table_make(hash_pointer, 0);
  dag_terapix_erosion(d, erosion);

  // transitive closure
  bool changed = true;
  while (changed &&
         (computables = dag_computable_vertices(d, done, done, current)))
  {
    // ensure determinism
    gen_sort_list(computables, (gen_cmp_func_t) dagvtx_terapix_priority);
    changed = false;
    FOREACH(dagvtx, v, computables)
    {
      // keep erosion up to cut
      // hmmm. what about \sigma_{d \in NSEW} erosion_d ?
      // would not work because the erosion only make sense if it is
      // the same for all imagelet, or said otherwise the erosion is
      // aligned to the worst case so that tiling can reasonnably take place.
      if ((((_int) hash_get(erodes, NORTH(v))) <= cut) &&
          (((_int) hash_get(erodes, SOUTH(v))) <= cut) &&
          (((_int) hash_get(erodes, EAST(v))) <= cut) &&
          (((_int) hash_get(erodes, WEST(v))) <= cut))
      {
        set_add_element(current, current, v);
        set_add_element(done, done, v);
        lcurrent = CONS(dagvtx, v, lcurrent);
        changed = true;
      }
    }

    // cleanup
    gen_free_list(computables), computables = NIL;
  }

  // cleanup GLOBAL
  hash_table_free(erosion), erosion = NULL;

  lcurrent = gen_nreverse(lcurrent);
  pips_assert("some vertices where extracted", lcurrent!=NIL);

  // build extracted dag
  dag nd = make_dag(NIL, NIL, NIL);
  FOREACH(dagvtx, v, lcurrent)
  {
    // pips_debug(7, "extracting node %" _intFMT "\n", dagvtx_number(v));
    dag_append_vertex(nd, copy_dagvtx_norec(v));
  }
  dag_compute_outputs(nd, NULL, output_images, NIL, false);
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

/*************************************************** TERAPIX HANDLE SEQUENCE */

static void migrate_statements(sequence sq, dag d, set dones)
{
  set stats = set_make(set_pointer);
  dag_statements(stats, d);
  freia_migrate_statements(sq, stats, dones);
  set_union(dones, dones, stats);
  set_free(stats);
}

/* do compile a list of statements for terapix
 * @param module, current module (function) name
 * @param ls, list of statements taken from the sequence
 * @param occs, occurences of images (image -> set of statements)
 * @param helper_file, file to which code is to be generated
 * @param number, number of this statement sequence in module
 * @return list of intermediate image to allocate
 */
list freia_trpx_compile_calls
(string module,
 dag fulld,
 sequence sq,
 list /* of statements */ ls,
 const hash_table occs,
 hash_table exchanges,
 const set output_images,
 FILE * helper_file,
 set helpers,
 int number)
{
  // build DAG for ls
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

  // dump final dag
  dag_dot_dump_prefix(module, "dag_cleaned_", number, fulld,
                      added_before, added_after);

  hash_table init = hash_table_make(hash_pointer, 0);
  list new_images = dag_fix_image_reuse(fulld, init, occs);

  string fname_fulldag = strdup(cat(module, "_terapix", HELPER, itoa(number)));

  // First, split only on scalar deps...
  // is it that simple? NO!
  // consider A -> B -> s -> C -> D
  //           \-> E -> F />
  // then ABEF / CD is chosen
  // although ABE / FCD and AB / EFCD would be also possible..

  pips_assert("erosion is clean", erosion==NULL);
  erosion = hash_table_make(hash_pointer, 0);
  list ld = dag_split_on_scalars(fulld,
                                 not_implemented,
                                 choose_terapix_vertex,
                                 (gen_cmp_func_t) dagvtx_terapix_priority,
                                 dag_terapix_reset_erosion,
                                 output_images);
  hash_table_free(erosion), erosion = NULL;

  pips_debug(4, "dag initial split in %d dags\n", (int) gen_length(ld));

  const char* dag_cut = get_string_property(trpx_dag_cut);
  pips_assert("valid cutting strategy", trpx_dag_cut_is_valid(dag_cut));

  // globally remaining statements
  set global_remainings = set_make(set_pointer);
  set_assign_list(global_remainings, ls);

  int n_split = 0;
  int stnb = -1;
  set dones = set_make(set_pointer);

  FOREACH(dag, d, ld)
  {
    // ??? should migrate beforehand?

    // skip if something is not implemented
    if (terapix_not_implemented(d))
      continue;

    if (dag_no_image_operation(d))
      continue;

    if (trpx_dag_cut_none_p(dag_cut))
    {
      migrate_statements(sq, d, dones);
      // direct handling of the dag
      stnb = freia_trpx_compile_one_dag(module, ls, d, fname_fulldag, n_split,
                       -1, global_remainings, helper_file, helpers, stnb, init);
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
        dag dc = cut_perform(d, cut, erosion, fulld, output_images);
        migrate_statements(sq, dc, dones);
        // generate code for cut
        stnb =
          freia_trpx_compile_one_dag(module, ls, dc, fname_fulldag, n_split,
                n_cut++, global_remainings, helper_file, helpers, stnb, init);
        // cleanup
        free_dag(dc);
        hash_table_clear(erosion);
      }
      migrate_statements(sq, d, dones);
      stnb = freia_trpx_compile_one_dag(module, ls, d, fname_fulldag, n_split,
                 n_cut++, global_remainings, helper_file, helpers, stnb, init);
      hash_table_free(erosion);
    }
    else if (trpx_dag_cut_enumerate_p(dag_cut))
      pips_internal_error("not implemented yet");
    else
      pips_internal_error("cannot get there");

    n_split++;
  }

  freia_insert_added_stats(ls, added_before, true);
  added_before = NIL;
  freia_insert_added_stats(ls, added_after, false);
  added_after = NIL;

  // full cleanup
  set_free(global_remainings), global_remainings = NULL;
  free(fname_fulldag), fname_fulldag = NULL;
  FOREACH(dag, dc, ld)
    free_dag(dc);
  gen_free_list(ld);

  // deal with new images
  list real_new_images =
    freia_allocate_new_images_if_needed(ls, new_images, occs, init, init);
  gen_free_list(new_images);
  hash_table_free(init);
  return real_new_images;
}
