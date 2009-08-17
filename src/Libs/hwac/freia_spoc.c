/*

  $Id$

  Copyright 1989-2009 MINES ParisTech

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

// help reduce code size:
#define T true
#define F false
#define cat concatenate

#define sb_cat string_buffer_cat
#define sb_app string_buffer_append

/* @return new allocated variable name using provided prefix.
 * *params is incremented as a side effect.
 */
static string get_var(string prefix, int * params)
{
  return strdup(cat(prefix, itoa((*params)++), NULL));
}

static void comment(string_buffer code, spoc_hardware_type hw,
		    dagvtx v, int stage, int side, bool flip)
{
  sb_cat(code, "  // ", what_operation(hw), " ",
	 itoa(dagvtx_number(v)), " ", dagvtx_operation(v), NULL);
  sb_cat(code, " stage ", itoa(stage), NULL);
  if (hw!=spoc_type_alu) sb_cat(code, " side ", itoa(side), NULL);
  if (hw==spoc_type_alu && flip) sb_app(code, " flipped");
  sb_app(code, "\n");
}

/* generate a configuration for the ALU hardware component.
 */
static void spoc_alu_conf
(spoc_alu_t alu,
 string_buffer head,
 string_buffer body,
 __attribute__ ((__unused__)) string_buffer tail,
 int stage,
 bool flip,
 int * cst,
 dagvtx orig)
{
  if (alu!=alu_unused) // should not be called? copy?
  {
    if (flip) // only flip if distinct
    {
      spoc_alu_t flipped_alu = get_spoc_alu_conf(alu)->flipped;
      if (alu!=flipped_alu)
	alu = flipped_alu;
      else
	flip = false;
    }
    const spoc_alu_op_t * aluop = get_spoc_alu_conf(alu);
    message_assert("alu operation found", alu==aluop->op);

    string s_stage = strdup(itoa(stage));
    comment(body, spoc_type_alu, orig, stage, -1, flip);
    sb_cat(body,
      "  si.alu[", s_stage, "][0].op = ", aluop->setting, ";\n", NULL);
    if (aluop->use_cst) {
      // string var = "c%d";
      string s_var = get_var("cst", cst);
      // (*decls) += ", int32_t " + var;
      sb_cat(head, ", int32_t ", s_var, NULL);
      // res += "spocinstr.alu[%stage][0].constant = " + var + ";\n";
      sb_cat(body,
	"  si.alu[", s_stage, "][0].constant = ", s_var, ";\n", NULL);
      free(s_var);
    }

    free(s_stage);
  }
}

/***************************************** POC & TH & MEASURE CONFIGURATIONS */

/* generate a configuration for a POC (morpho) hardware component.
 */
static void spoc_poc_conf
(spoc_poc_t poc,
 string_buffer head,
 string_buffer body,
 __attribute__ ((__unused__)) string_buffer tail,
 int stage,
 int side,
 int * cst,
 dagvtx orig)
{
  if (poc.op!=spoc_poc_unused) // should not be called?
  {
    string
      s_stag = strdup(itoa(stage)),
      s_side = strdup(itoa(side)),
      s_var = get_var("kern", cst);

    // int32_t * var (kernel)
    sb_cat(head, ", int32_t *", s_var, NULL);

    // spocinstr.poc[0][0].op = SPOC_POC_ERODE;
    comment(body, spoc_type_poc, orig, stage, side, false);
    sb_cat(body,
      "  si.poc[", s_stag, "][", s_side, "].op = ", NULL);
    if (poc.op==spoc_poc_erode)
      sb_app(body, "SPOC_POC_ERODE;\n");
    else if (poc.op==spoc_poc_dilate)
      sb_app(body, "SPOC_POC_DILATE;\n");
    else
      pips_internal_error("unexpected poc operation %d", poc.op);

    // spocinstr.poc[0][0].grid = SPOC_POC_8_CONNEX;
    sb_cat(body,
      "  si.poc[", s_stag, "][", s_side, "].grid = ", NULL);
    if (poc.connectivity==6)
      sb_app(body, "SPOC_POC_6_CONNEX;\n");
    else if (poc.connectivity==8)
      sb_app(body, "SPOC_POC_8_CONNEX;\n");
    else
      pips_internal_error("unexpected poc connectivity %d", poc.connectivity);

    // for(i=0 ; i<9 ; i++) spocparam.poc[0][0].kernel[i] = kernel[i];
    sb_cat(body,
      "  for(i=0 ; i<9 ; i++)\n"
      "    sp.poc[", s_stag, "][", s_side, "].kernel[i] = ",
	      s_var, "[i];\n", NULL);

    free(s_stag);
    free(s_side);
    free(s_var);
  }
}

/* generate a configuration for a threshold component.
 */
static void spoc_th_conf
(string_buffer head,
 string_buffer body,
 __attribute__ ((__unused__)) string_buffer tail,
 int stage,
 int side,
 int * cst,
 dagvtx orig)
{
  string
    s_inf = get_var("inf", cst),
    s_sup = get_var("sup", cst),
    s_bin = get_var("bin", cst),
    s_stag = strdup(itoa(stage)),
    s_side = strdup(itoa(side));

  // , int32_t cstX, int32_t cstY, bool binZ
  sb_cat(head, ", int32_t ", s_inf, ", int32_t ", s_sup, ", bool ", s_bin,
	 NULL);

  // spocinstr.th[0][0].op =
  //   (binarize==true) ? SPOC_TH_BINARIZE : SPOC_TH_NO_BINARIZE;
  // spocparam.th[0][0].boundmin = boundinf;
  // spocparam.th[0][0].boundmax = boundsup;
  comment(body, spoc_type_thr, orig, stage, side, false);
  sb_cat(body,
    "  si.th[", s_stag, "][", s_side, "].op =",
    "    (", s_bin, ")? SPOC_TH_BINARIZE : SPOC_TH_NO_BINARIZE;\n",
    "  sp.th[", s_stag, "][", s_side, "].boundmin = ", s_inf, ";\n",
    "  sp.th[", s_stag, "][", s_side, "].boundmax = ", s_sup, ";\n",
		    NULL);

  free(s_inf);
  free(s_sup);
  free(s_bin);
}

static int spoc_measure_n_params(spoc_measure_t measure)
{
  switch (measure)
  {
  case measure_none:
    return 0;
  case measure_max_coord:
  case measure_min_coord:
    return 3;
  default:
    return 1;
  }
}

/* there is no real configuration for the measures,
 * the issue is just to fetch them.
 */
static void spoc_measure_conf
(spoc_measure_t measure,
 string_buffer head,
 __attribute__ ((__unused__)) string_buffer body,
 string_buffer tail,
 int stage,
 int side,
 int * cst,
 dagvtx orig)
{
  int n = spoc_measure_n_params(measure);
  if (n==0) return;
  pips_assert("1 to 3 parameters", n>=1 && n<=3);

  string
    s_stage = strdup(itoa(stage)),
    s_side = strdup(itoa(side)),
    reduc = strdup(cat("reduc.measure[", s_stage, "][", s_side, "].", NULL));

  string
    v_1 = get_var("red", cst), v_2 = NULL, v_3 = NULL;
  if (n>=2) v_2 = get_var("red", cst);
  if (n>=3) v_3 = get_var("red", cst);

  sb_app(tail, "\n");
  comment(tail, spoc_type_mes, orig, stage, side, false);

  switch (measure) {
  case measure_max:
  case measure_min:
  case measure_max_coord:
  case measure_min_coord:
    sb_cat(head, ", int32_t * ", v_1, NULL);
    sb_cat(tail, "  *", v_1, " = (int32_t) ", reduc,
		      measure==measure_min||measure==measure_min_coord?
		      "minimum": "maximum", ";\n", NULL);
    if (n>1)
    {
      sb_cat(head, ", uint32_t * ", v_2, ", uint32_t * ", v_3, NULL);
      sb_cat(tail,
			"  *", v_2, " = (uint32_t) ", reduc,
			measure==measure_min||measure==measure_min_coord?
			"min": "max", "_coord_x;\n",
			"  *", v_3, " = (uint32_t) ", reduc,
			measure==measure_min||measure==measure_min_coord?
			"min": "max", "_coord_y;\n",
			NULL);
    }
    break;
  case measure_vol:
    sb_cat(head, ", int32_t * ", v_1, NULL);
    sb_cat(tail,
		      "  *", v_1, " = (int32_t) ", reduc, "volume;\n", NULL);
    break;
  default:
    pips_internal_error("unexpected measure %d", measure);
  }

  free(s_stage);
  free(s_side);
  free(reduc);
  free(v_1);
  if (v_2) free(v_2);
  if (v_3) free(v_3);
}

/************************************************************** HELPER NAMES */

#define HELPER "_helper_"
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

/* return the actual function call from a statement,
 * dealing with assign and return.
 */
static call freia_statement_to_call(statement s)
{
  // sanity check, somehow redundant
  instruction i = statement_instruction(s);
  pips_assert("is a call", instruction_call_p(i));
  call c = instruction_call(i);
  if (ENTITY_ASSIGN_P(call_function(c)))
  {
    list args = call_arguments(c);
    pips_assert("2 args", gen_length(args) == 2);
    syntax sy = expression_syntax(EXPRESSION(CAR(CDR(args))));
    pips_assert("expr is a call", syntax_call_p(sy));
    c = syntax_call(sy);
  }
  else if (ENTITY_C_RETURN_P(call_function(c)))
  {
    list args = call_arguments(c);
    pips_assert("one arg", gen_length(args)==1);
    syntax sy = expression_syntax(EXPRESSION(CAR(args)));
    pips_assert("expr is a call", syntax_call_p(sy));
    c = syntax_call(sy);
  }
  return c;
}

/* basic configuration generation for a stage, depending on hw description
 * @returns the number of arguments expected?
 * what about their type?
 */
static void basic_spoc_conf
  (string_buffer head,
   string_buffer body,
   string_buffer tail,
   int stage,
   int side,
   bool flip,
   int * cst, // numbering of arguments...
   const spoc_hw_t * conf,
   dagvtx orig)
{
  // only one call should be used for AIPO functions ?
  if (conf->used & spoc_poc_0)
    spoc_poc_conf(conf->poc[0], head, body, tail, stage, side, cst, orig);
  if (conf->used & spoc_alu)
    spoc_alu_conf(conf->alu, head, body, tail, stage, flip, cst, orig);
  if (conf->used & spoc_th_0)
    spoc_th_conf(head, body, tail, stage, side, cst, orig);
  if (conf->used & spoc_measure_0)
    spoc_measure_conf(conf->mes[0], head, body, tail, stage, side, cst, orig);
  sb_app(body, "\n");
}

/* returns an allocated expression list of the parameters only
 * (i.e. do not include the input & output images).
 */
static list /* of expression*/
get_parameters(const freia_api_t * api, list args)
{
  int skip = api->arg_img_in + api->arg_img_out;
  while (skip--) args = CDR(args);
  return gen_full_copy_list(args);
}

typedef struct {
  entity image;     // output
  dagvtx producer;  // NULL on input and output
  int stage;
  spoc_hardware_type level;  // nope | input < poc < alu < threshold < measure
  int side;  // 0 or 1
  bool flip; // whether alu op is flipped
}  op_schedule;

static void init_op_schedule(op_schedule * op, entity img, int side)
{
  op->image = img; // may be NULL?
  op->producer = NULL;
  op->stage = 0;
  op->level = spoc_type_inp;
  op->side = side;
  op->flip = false;
}

static int max_stage(const op_schedule * in0, const op_schedule * in1)
{
  if (in0->image)
    if (in1->image)
      // both
      return (in0->stage > in1->stage)? in0->stage: in1->stage;
    else
      // only in0
      return in0->stage;
  else
    if (in1->image)
      // only in1
      return in1->stage;
    else
      // none
      return 0;
}

static string dagvtx_to_string(const dagvtx);

static bool image_is_needed(entity img, dag d, set todo)
{
  bool needed = false;
  if (img)
  {
    needed = gen_in_list_p(img, dag_outputs(d));
    SET_FOREACH(dagvtx, v, todo)
      if (gen_in_list_p(img, vtxcontent_inputs(dagvtx_content(v))))
	needed = true;
    pips_debug(7, "\"%s\" is%s needed\n",
	       entity_local_name(img), needed? "": " not");
  }
  return needed;
}

static void print_op_schedule(FILE *, string, const op_schedule *);
static bool check_wiring_output(hash_table, int, int);

/* depending on available images (stage %d, level, side 0/1)
 * and vertex operation to perform , tell where to perform it.
 * some voodoo again...
 */
static void
where_to_perform_operation
  (const dagvtx op,
   op_schedule * in0,
   op_schedule * in1,
   dag computed, // current computed dag
   set todo,     // vertices still to be computed
   op_schedule * out,
   hash_table wiring)
{
  vtxcontent c = dagvtx_content(op);
  out->side = -100;

  ifdebug(6) {
    pips_debug(6, "scheduling (%" _intFMT " %s) %s\n",
	       dagvtx_number(op), dagvtx_operation(op),
	       vtxcontent_out(c)==entity_undefined? "":
	         entity_local_name(vtxcontent_out(c)));
    print_op_schedule(stderr, "in0", in0);
    print_op_schedule(stderr, "in1", in1);
  }

  spoc_hardware_type level = (spoc_hardware_type) vtxcontent_optype(c);

  if (level == spoc_type_mes)
  {
    // special case, because the image is not really consummed...
    pips_assert("one input image", gen_length(vtxcontent_inputs(c))==1);
    entity measured = ENTITY(CAR(vtxcontent_inputs(c)));
    pips_assert("input image available",
		measured==in0->image || measured==in1->image);
    // go to the end of the stage of the image producer, on same side...
    *out = (measured==in0->image)? *in0: *in1;
  }
  else
  {
    // how many images (arguments) are needed by the operation?
    int needed = (int) gen_length(vtxcontent_inputs(c));
    switch (needed)
    {
    case 0: // ALU CONST GENERATION ?
      pips_assert("alu const image generation", level==spoc_type_alu);
      // I must check that the alu is not already used?!
      // So I take a safe bet...
      out->stage = max_stage(in0, in1);
      if (in0->image && in0->stage == out->stage && in0->level > spoc_type_alu)
	out->stage++;
      if (in1->image && in1->stage == out->stage && in1->level > spoc_type_alu)
	out->stage++;
      break;
    case 1: // ANYTHING, may have to chose a side.
      {
	entity dep = ENTITY(CAR(vtxcontent_inputs(c)));
	pips_assert("dependence is available",
		    dep==in0->image || dep==in1->image);
	const op_schedule
	  * used = dep==in0->image? in0: in1,
	  * notused = used==in0? in1: in0;

	pips_debug(7, "using %s (%d %s %d)\n",
		   entity_local_name(used->image),
		   used->stage, what_operation(used->level), used->side);

	// default stage and side... should I check?
	out->stage = used->stage;
	out->side = used->side;

	if (level == spoc_type_nop)
	  // we have a copy... the operation is very light:-)
	  level = used->level;
	else
	  if (level <= used->level) out->stage++;

	// ??? if the input image is still needed, must include a cross
	if (image_is_needed(used->image, computed, todo) &&
	    spoc_type_alu < used->level && used->stage==out->stage)
	  out->stage++;

	// conflict
	if (notused->image &&
	    out->stage==notused->stage &&
	    out->level==notused->level)
	{
	  if (out->level==spoc_type_alu)
	    out->stage++;
	  else
	    out->side = 1 - notused->side;
	}

	if (out->side==-1)
	{
	  if (check_wiring_output(wiring, used->stage, 0))
	    out->side = 0;
	  else if (check_wiring_output(wiring, used->stage, 1))
	    out->side = 1;
	  else
	    pips_internal_error("no available path");
	}
      }
      break;
    case 2: // ALU
      // if same image is used as both operands?
      if (!in1->image) *in1 = *in0;
      if (!in0->image) *in0 = *in1;
      pips_assert("alu binary operation", level==spoc_type_alu);
      out->stage = max_stage(in0, in1);
      if (out->stage==in0->stage && in0->level >= spoc_type_alu)
	out->stage++;
      if (out->stage==in1->stage && in1->level >= spoc_type_alu)
	out->stage++;
      break;
    default:
      pips_internal_error("shoud not get there");
    }
  }

  // common settings
  out->level = level;
  if (out->level == spoc_type_alu)
    out->side = -1;
  if (vtxcontent_out(c)!=entity_undefined)
  {
    out->image = vtxcontent_out(c);
    out->producer = op;
  }
  // else it is a measure, don't change...
  ifdebug(6) print_op_schedule(stderr, "out", out);
}

// can I get out of stage on this side?
static bool
check_wiring_output(hash_table wiring, int stage, int side)
{
  int mux = side? 3: 0;
  _int index = stage*4+mux + 1;
  return !hash_defined_p(wiring, (void *) index);
}

static void
set_wiring(string_buffer code,
	   int stage, // 0 ..
	   int mux,   // 0 to 3
	   _int value, // 0 1
	   hash_table wiring)
{
  // must not use 0 & -1
  _int index = stage*4+mux + 1;
  if (hash_defined_p(wiring, (void *) index))
  {
    _int old = (_int) hash_get(wiring, (void *) index);
    if (old!=value)
    {
      string_buffer_to_file(code, stderr);
      pips_internal_error("stage %d mux %d: %" _intFMT " vs %" _intFMT "\n",
			  stage, mux, old, value);
    }
  }
  else
  {
    hash_put(wiring, (void *) index, (void *) value);
    string s_stage = strdup(itoa(stage)), s_mux = strdup(itoa(mux));
    sb_cat(code, "  si.mux[", s_stage, "][", s_mux, "].op = "
	   "SPOC_MUX_IN", value? "1": "0", ";\n", NULL);
    free(s_stage);
    free(s_mux);
  }
}

/* all possible wirings at one stage
 */
static void generate_wiring_stage
(string_buffer code,
 int stage,
 int in_side, // 0, 1, -1 for "alu"...
 int out_side,
 hash_table wiring)
{
  pips_assert("check sides",
	      in_side>=-1 && in_side<=1 && out_side>=0 && out_side<=1);
  pips_debug(7, "stage %d in %d -> out %d\n", stage, in_side, out_side);
  switch (in_side)
  {
  case -1:
    if (out_side) {
      set_wiring(code, stage, 1, 0, wiring);
      set_wiring(code, stage, 3, 0, wiring);
    }
    else {
      set_wiring(code, stage, 0, 1, wiring);
      set_wiring(code, stage, 2, 0, wiring);
    }
    break;
  case 0:
    if (out_side) {
      set_wiring(code, stage, 1, 0, wiring);
      set_wiring(code, stage, 3, 0, wiring);
    }
    else
      set_wiring(code, stage, 0, 0, wiring);
    break;
  case 1:
    if (out_side)
      set_wiring(code, stage, 3, 1, wiring);
    else {
      set_wiring(code, stage, 0, 1, wiring);
      set_wiring(code, stage, 2, 1, wiring);
    }
    break;
  default:
    pips_internal_error("should not get there...");
  }
}

/* generate wire code from in to out, record choices in wiring.
 */
static void generate_wiring
(string_buffer code,
 op_schedule * in,
 op_schedule * out,
 hash_table wiring)
{
  pips_debug(6, "%s [%d %s %d] -> [%d %s %d] %" _intFMT "\n",
	     entity_local_name(in->image),
	     in->stage, what_operation(in->level), in->side,
	     out->stage, what_operation(out->level), out->side,
	     dagvtx_number(out->producer));

  pips_assert("ordered schedule",
	      (in->stage < out->stage) ||
	      ((in->stage == out->stage) && (in->level <= out->level)));

  // code comment...
  string
    in_stage = strdup(itoa(in->stage)),
    out_stage = strdup(itoa(out->stage)),
    in_side = strdup(itoa(in->side)),
    out_side = strdup(itoa(out->side));

  sb_cat(code,
	 "  // ", entity_local_name(in->image),
	 " [", in_stage, " ", what_operation(in->level), NULL);
  if (in->level!=spoc_type_alu)
    sb_cat(code, " ", in_side, NULL);
  sb_cat(code,
	 "] -> [", out_stage, " ", what_operation(out->level), NULL);
  if (out->level!=spoc_type_alu)
    sb_cat(code, " ", out_side, NULL);
  sb_cat(code, "] ", itoa(dagvtx_number(out->producer)), " ",
	 dagvtx_operation(out->producer), "\n", NULL);
  free(in_stage); free(out_stage); free(in_side); free(out_side);

  // let us wire!
  if (in->stage==out->stage)
  {
    switch (in->level)
    {
      case spoc_type_inp:
      case spoc_type_poc:
	if (out->level<=spoc_type_alu)
	  sb_app(code, "  // nope\n");
	// ELSE thr or mes...
	else
	  generate_wiring_stage(code, in->stage, in->side, out->side, wiring);
	break;
      case spoc_type_alu:
	// out->level is after thr, so there is a side
	generate_wiring_stage(code, in->stage, -1, out->side, wiring);
	break;
      case spoc_type_thr:
      case spoc_type_mes:
      case spoc_type_out:
	// no choice
	pips_assert("same side", in->side==out->side);
	sb_app(code, "  // nope\n");
	break;
      case spoc_type_nop:
      default:
	pips_internal_error("shoud not get there");
    }

    // record operation flipping
    if (out->level == spoc_type_alu)
    {
      vtxcontent cout = dagvtx_content(out->producer);
      list lins = vtxcontent_inputs(cout);
      pips_assert("sanity", in->side>=0 && lins);
      switch (gen_length(lins))
      {
      case 2:
	if (in->side==0)
	  out->flip = !(in->image == ENTITY(CAR(lins)));
	else // in->side==1
	  out->flip = (in->image == ENTITY(CAR(lins)));
	break;
      case 1:
	out->flip = in->side==1;
	break;
      default:
      case 0:
	pips_internal_error("should not get there");
      }
    }
  }
  else // in->stage < out->stage
  {
    // let us do it with a recursion...
    int prefered;
    switch (in->level)
    {
    case spoc_type_inp:
    case spoc_type_poc:
    case spoc_type_alu:
      if (in->level==spoc_type_alu)
	// try direct, or default to 0
	prefered = (out->side>=0)? out->side: 0;
      else // try straight on first...
	prefered = in->side;
      // try both cases
      if (check_wiring_output(wiring, in->stage, prefered))
      {
	generate_wiring_stage(code, in->stage, in->side, prefered, wiring);
	// what if needed twice?
	in->stage++;
	in->side = prefered;
	in->level = spoc_type_inp;
	generate_wiring(code, in, out, wiring);
      }
      else if (check_wiring_output(wiring, in->stage, 1 - prefered))
      {
	generate_wiring_stage(code, in->stage, in->side, 1 - prefered, wiring);
	in->stage++;
	in->side = 1 - prefered;
	in->level = spoc_type_inp;
	generate_wiring(code, in, out, wiring);
      }
      else
	pips_internal_error("no available path...");
      break;
    case spoc_type_thr:
    case spoc_type_mes:
    case spoc_type_out:
      // no choice, skip to input of next stage
      in->stage++;
      in->level = spoc_type_inp;
      generate_wiring(code, in, out, wiring);
      break;
    case spoc_type_nop:
    default:
      pips_internal_error("shoud not get there");
    }
  }
}

/* build up final pipeline code from various pieces
 * @param module
 * @param function_name to be generated
 * @param code output
 * @param head input, function parameters
 * @param body input, pipeline stage initialization
 * @param tail input, getting back reduction results
 * @param n_im_* number of in and out paramters to function
 * @param some_reductions if there are
 * @param p_* variables sent and received from pipeline
 */
static void freia_spoc_code_buildup
(string module, string function_name, string_buffer code,
 const string_buffer head, const string_buffer body, const string_buffer tail,
 int n_im_out, int n_im_in, bool some_reductions,
 const string out0, const string out1, const string in0, const string in1)
{
  // function header: freia_error helper_function(freia_data_2d ...
  sb_cat(code,
	 "\n"
	 "// FREIA-SPoC helper function for module ", module, "\n"
	 "freia_status ", function_name, "(", NULL);
  // first, image arguments
  if (n_im_out>0) sb_app(code, FREIA_IMAGE "o0");
  if (n_im_out>1) sb_app(code, ", " FREIA_IMAGE "o1");
  if (n_im_out!=0) sb_app(code, ", ");
  if (n_im_in>0) sb_app(code, FREIA_IMAGE "i0");
  if (n_im_in>1) sb_app(code, ", " FREIA_IMAGE "i1");

  string_buffer_append_sb(code, head);

  // end of headers (some arguments may have been added by stages),
  // and begin the function body
  sb_app(code, ")\n{\n" FREIA_SPOC_DECL);
  string_buffer_append_sb(code, body);

  // generate actual call to the accelerator
  sb_cat(code,
	 FREIA_SPOC_CALL
	 "  // actual call of spoc hardware\n"
	 "  freia_cg_template_process_2i_2o",
	 "(op, param, ", out0, ", ", out1, ", ", in0, ", ", in1, ");\n",
	 NULL);

  if (some_reductions)
    sb_app(code,
	   "\n"
	   "  // get reductions\n"
	   "  freia_cg_read_reduction_results(&redres);\n");

  string_buffer_append_sb(code, tail);

  sb_app(code, "\n  return ret;\n}\n");
}

/* generate a SPoC pipeline from a single DAG for module.
 *
 * @param module current
 * @param helper name of function to generate
 * @param code output here
 * @param dpipe dag to consider
 * @param lparams parameters to call helper
 *
 * some side effects:
 * - if there is an overflow, dpipe updated with remaining vertices.
 * - accelerated statements are cleaned
 */
static void freia_spoc_pipeline
(string module, string helper, string_buffer code, dag dpipe, list * lparams)
{
  hash_table wiring = hash_table_make(hash_int, 128);
  list outs = gen_copy_seq(dag_outputs(dpipe));
  bool some_reductions = false;
  int pipeline_depth = get_int_property(spoc_depth_prop);

  pips_debug(3, "running on '%s' for %u functions\n",
	     module, (unsigned int) gen_length(dag_vertices(dpipe)));
  pips_assert("non empty dag", gen_length(dag_vertices(dpipe))>0);

  // generate a helper function in some file
  int cst = 0;
  string_buffer
    head = string_buffer_make(true),
    body = string_buffer_make(true),
    tail = string_buffer_make(true);

  // first arguments are out & in images
  int n_im_in = gen_length(dag_inputs(dpipe));

  // special case where the first operations use the same image twice
  int n_args_first_operation =
    gen_length(vtxcontent_inputs(dagvtx_content
				 (DAGVTX(CAR(dag_vertices(dpipe))))));

  pips_assert("0, 1, 2 input images", n_im_in>=0 && n_im_in<=2);

  // piped images in helper
  string p_in0 = "NULL", p_in1 = "NULL", p_out0 = "NULL", p_out1 = "NULL";
  // piped images in function
  entity a_in0 = NULL, a_in1 = NULL;

  op_schedule in0, in1, out;
  list /* of entity */ limg = NIL;

  // build input arguments/parameters/arguments
  if (n_im_in>=1)
  {
    a_in0 = ENTITY(CAR(dag_inputs(dpipe)));
    p_in0 = "i0";
    init_op_schedule(&in0, a_in0, 0);
    limg = CONS(entity, a_in0, NIL);
  }
  else
    init_op_schedule(&in0, NULL, 0);

  if (n_args_first_operation==2 && n_im_in==1)
  {
    // special case alu(i,i)...
    init_op_schedule(&in1, a_in0, 1);
    p_in1 = "i0";
  }
  else if (n_im_in>=2)
  {
    a_in1 = ENTITY(CAR(CDR(dag_inputs(dpipe))));
    p_in1 = "i1";
    init_op_schedule(&in1, a_in1, 1);
    limg = gen_nconc(limg, CONS(entity, a_in1, NIL));
  }
  else
    init_op_schedule(&in1, NULL, 1);

  list vertices = gen_nreverse(gen_copy_seq(dag_vertices(dpipe)));
  set todo = set_make(set_pointer);
  set_assign_list(todo, vertices);

  // keep track of current stage for comments, and to detect overflows
  int stage = -1;

  FOREACH(dagvtx, v, vertices)
  {
    vtxcontent vc = dagvtx_content(v);
    statement s = pstatement_statement(vtxcontent_source(vc));
    call c = freia_statement_to_call(s);
    string func = entity_local_name(call_function(c));
    const freia_api_t * api = hwac_freia_api(func);
    pips_assert("AIPO function found", api);

    set_del_element(todo, todo, v);

    // schedule...
    where_to_perform_operation(v, &in0, &in1, dpipe, todo, &out, wiring);

    if (out.stage >= pipeline_depth)
    {
      // pipeline is full, vi was not computed:
      pips_debug(3, "pipeline overflow at %" _intFMT "\n", dagvtx_number(v));
      set_add_element(todo, todo, v);
      break;
    }

    // needed parameters if any
    *lparams = gen_nconc(*lparams, get_parameters(api, call_arguments(c)));

    some_reductions |= vtxcontent_optype(vc)==spoc_type_mes;

    // there may be regressions if it is a little bit out of order...
    if (out.stage>stage)
    {
      stage = out.stage;
      sb_cat(body, "\n  // STAGE ", itoa(stage), "\n", NULL);
    }

    // generate wiring, which may change flip stage on ALU operation
    // special case if in0==in1 and is needed?
    if (in0.image && gen_in_list_p(in0.image, vtxcontent_inputs(vc)))
      generate_wiring(body, &in0, &out, wiring);

    if (in1.image && gen_in_list_p(in1.image, vtxcontent_inputs(vc)))
      generate_wiring(body, &in1, &out, wiring);

    // generate stage
    basic_spoc_conf(head, body, tail,
		    out.stage, out.side, out.flip, &cst, &(api->spoc), v);

    // update status of live images
    if (!image_is_needed(in0.image, dpipe, todo) || out.image==in0.image)
      in0 = out;
    else if (!image_is_needed(in1.image, dpipe, todo) || out.image==in1.image)
      in1 = out;
    else if (in0.image && in0.image == in1.image)
    {
      // we have the same image...
      if (out.side==in0.side)
	in0 = out;
      else if (out.side==in1.side)
	in1 = out;
      else
	pips_internal_error("should not get there (same image)?");
    }
    else
      pips_internal_error("should not get there (3 live images)...");
  }

  set computed = set_make(set_pointer);
  set_assign_list(computed, vertices);
  set_difference(computed, computed, todo);

  // pipeline interruption?
  if (!set_empty_p(todo))
  {
    // what are the effective outputs in the middle of the pipeline
    list new_outs = NIL;
    if (image_is_needed(in1.image, dpipe, todo))
      new_outs = CONS(entity, in1.image, new_outs);
    if (image_is_needed(in0.image, dpipe, todo))
      new_outs = CONS(entity, in0.image, new_outs);

    // gen_free_list(dag_outputs(dpipe));
    // dag_outputs(dpipe) = outs;
    gen_free_list(outs), outs = new_outs;
  }

  // cleanup dpipe of computed vertices
  FOREACH(dagvtx, vr, vertices)
  {
    if (set_belong_p(computed, vr))
    {
      dag_remove_vertex(dpipe, vr);
      hwac_kill_statement(pstatement_statement
		     (vtxcontent_source(dagvtx_content(vr))));
      free_dagvtx(vr);
    }
  }
  dag_set_inputs_outputs(dpipe);

  gen_free_list(vertices), vertices = NIL;
  set_free(computed), computed = NULL;

  // handle pipe outputs (may have been interrupted)
  int n_im_out = gen_length(outs);
  pips_assert("0, 1, 2 output images", n_im_out>=0 && n_im_out<=2);
  pips_assert("some input or output images", n_im_out || n_im_in);
  if (n_im_out==0)
  {
    sb_app(body, "\n  // no output image\n");
  }
  else if (n_im_out==1)
  {
    entity imout = ENTITY(CAR(outs));

    // put producer in in0
    if (in1.image==imout && in0.image!=imout)
      out = in0, in0 = in1, in1 = out;
    else if (in0.image==imout); // ok
    else pips_internal_error("result image is not alive");

    out.image = imout;
    out.producer = NULL;
    out.level = spoc_type_out;
    out.stage = in0.stage;
    out.side = in0.side>=0? in0.side: 0; // choose 0 if alu

    sb_cat(body, "\n"
	   "  // output image ", entity_local_name(imout),
	   " on ", itoa(out.side), "\n", NULL);
    generate_wiring(body, &in0, &out, wiring);

    if (out.side)
      p_out1 = "o0";
    else
      p_out0 = "o0";

    limg = CONS(entity, imout, limg);
  }
  else if (n_im_out==2)
  {
    entity
      out0 = ENTITY(CAR(outs)),
      out1 = ENTITY(CAR(CDR(outs)));
    int out0_side, out1_side;

    // make out0 match in0 and out1 match in1
    if (out0 != in0.image)
      out = in0, in0 = in1, in1 = out;
    pips_assert("results are available", out0==in0.image && out1==in1.image);

    // the more advanced image in the pipe decides the its output side
    if (in0.stage<in1.stage || (in0.stage==in1.stage && in0.level<in1.level))
    {
      out0_side = in0.side;
      if (out0_side==-1) out0_side = 0; // alu, choose!
      out1_side = 1 - out0_side;
    }
    else // in1 is more forward
    {
      out1_side = in1.side;
      if (out1_side==-1) out1_side = 1; // alu, choose!
      out0_side = 1 - out1_side;
    }

    pips_assert("2 images: one output on each side",
	(out0_side==0 && out1_side==1) || (out0_side==1 && out1_side==0));

    limg = CONS(entity, out0, CONS(entity, out1, limg));
    if (out1_side) // out1 on side 1, thus out0 side 0
      p_out0 = "o0", p_out1 = "o1";
    else // they are reversed
      p_out0 = "o1", p_out1 = "o0";

    int out_stage = (in0.stage>in1.stage)? in0.stage: in1.stage;

    sb_cat(body, "\n"
	   "  // output image ", entity_local_name(out0),
	   " on ", itoa(out0_side), NULL);
    sb_cat(body, " and image ", entity_local_name(out1),
	   " on ", itoa(out1_side), "\n", NULL);

    // know extract them.
    out.image = out0;
    out.producer = NULL;
    out.level = spoc_type_out;
    out.side = out0_side;
    out.stage = out_stage;
    generate_wiring(body, &in0, &out, wiring);

    out.image = out1;
    out.producer = NULL;
    out.level = spoc_type_out;
    out.side = out1_side;
    out.stage = stage;
    generate_wiring(body, &in1, &out, wiring);
  }
  else
    pips_internal_error("only 0 to 2 output images!");

  // handle function image arguments
  list largs = NIL;
  limg = gen_nreverse(limg);
  FOREACH(entity, e, limg)
    largs = CONS(expression, entity_to_expression(e), largs);
  gen_free_list(limg), limg = NIL;
  *lparams = gen_nconc(largs, *lparams);

  // build whole code from various pieces
  freia_spoc_code_buildup(module, helper, code, head, body, tail,
			  n_im_out, n_im_in, some_reductions,
			  p_out0, p_out1, p_in0, p_in1);

  // cleanup
  gen_free_list(outs);
  set_free(todo);
  hash_table_free(wiring);
  string_buffer_free(&head);
  string_buffer_free(&body);
  string_buffer_free(&tail);
}

/************************************************************** DAG BUILDING */

// debug
static void
print_op_schedule(FILE * out, const string name, const op_schedule * op)
{
  fprintf(out, "sched '%s' = %s (%" _intFMT " %s) %d %s %d %s\n",
	  name,
	  op->image? entity_local_name(op->image): "NULL",
	  dagvtx_number(op->producer),
	  dagvtx_operation(op->producer),
	  op->stage, what_operation(op->level), op->side,
	  (op->level==spoc_type_alu)? (op->flip? "/": "."): "");
}

/***************************************************************** MISC UTIL */

/* returns whether the entity is a freia API (AIPO) function.
 */
static bool entity_freia_api_p(entity f)
{
  // very partial...
  return strncmp(entity_local_name(f), AIPO, strlen(AIPO))==0;
}

static list fs_expression_list_to_entity_list(list /* of expression */ args)
{
  list /* of entity */ lent = NIL;
  FOREACH(expression, ex, args)
  {
    syntax s = expression_syntax(ex);
    pips_assert("is a ref", syntax_reference_p(s));
    reference r = syntax_reference(s);
    pips_assert("simple ref", reference_indices(r)==NIL);
    lent = CONS(entity, reference_variable(r), lent);
  }
  lent = gen_nreverse(lent);
  return lent;
}

/* extract first entity item from list.
 */
static entity extract_fist_item(list * lp)
{
  list l = gen_list_head(lp, 1);
  entity e = ENTITY(CAR(l));
  gen_free_list(l);
  return e;
}

/* append statement s to dag d
 */
static void dag_append_freia_call(dag d, statement s)
{
  call c = freia_statement_to_call(s);
  entity called = call_function(c);
  pips_assert("FREIA API call", entity_freia_api_p(called));
  const freia_api_t * api = hwac_freia_api(entity_local_name(called));
  pips_assert("some api", api!=NULL);
  list /* of entity */ args =
    fs_expression_list_to_entity_list(call_arguments(c));

  // extract arguments
  entity out = entity_undefined;
  pips_assert("one out image max for an AIPO", api->arg_img_out<=1);
  if (api->arg_img_out==1)
    out = extract_fist_item(&args);
  list ins = gen_list_head(&args, api->arg_img_in);
  //list outp = gen_list_head(&args, api->arg_misc_out);
  //list inp = gen_list_head(&args, api->arg_misc_in);
  //pips_assert("no more arguments", gen_length(args)==0);

  vtxcontent cont =
    make_vtxcontent(-1, 0, make_pstatement(s), ins, out, NIL, NIL);
  set_operation(api, &vtxcontent_optype(cont), &vtxcontent_opid(cont));

  dagvtx nv = make_dagvtx(cont, NIL);
  dag_append_vertex(d, nv);
  // ??? TODO update global outs later?
}

/* comparison function for sorting dagvtx in qsort,
 * this is voodoo...
 * tells v1 < v2 => -1
 */
static int dagvtx_priority(const dagvtx * v1, const dagvtx * v2)
{
  string why = "none";
  int result = 0;
  vtxcontent
    c1 = dagvtx_content(*v1),
    c2 = dagvtx_content(*v2);

  // prioritize first measures if any
  if (vtxcontent_optype(c1)!=vtxcontent_optype(c2))
  {
    if (vtxcontent_optype(c1)==spoc_type_mes)
      result = -1, why = "mes";
    else if (vtxcontent_optype(c2)==spoc_type_mes)
      result = 1, why = "mes";
  }

  if (result==0)
  {
    // other criterions
    int
      l1 = (int) gen_length(vtxcontent_inputs(c1)),
      l2 = (int) gen_length(vtxcontent_inputs(c2));
    if (l1!=l2)
      // the more images are needed, the earlier
      result = l2-l1, why = "args";
    else if (vtxcontent_optype(c1)!=vtxcontent_optype(c2))
      // otherwise use the op types, which are somehow ordered
      result = vtxcontent_optype(c1) - vtxcontent_optype(c2), why = "ops";
    else
      // rely on statements
      result = dagvtx_number(*v1) - dagvtx_number(*v2), why = "stats";
  }

  pips_debug(7, "%" _intFMT " %s %s %" _intFMT " %s (%s)\n",
	     dagvtx_number(*v1), dagvtx_operation(*v1),
	     result<0? "<": (result==0? "=": ">"),
	     dagvtx_number(*v2), dagvtx_operation(*v2), why);

  pips_assert("total order", v1==v2 || result!=0);
  return result;
}

/************************************************************ IMPLEMENTATION */

/* return the vertices which may be computed from the list of
 * available images, excluding vertices in exclude.
 * return a list for determinism.
 */
static list /* of dagvtx */
get_computable_vertices(dag d, set exclude, set /* of entity */ avails)
{
  list computable = NIL;

  // hmmm... should reverse the list to handle implicit dependencies?
  // where, there is an assert() to check that it does not happen.
  list lv = gen_nreverse(gen_copy_seq(dag_vertices(d)));

  FOREACH(dagvtx, v, lv)
  {
    vtxcontent c = dagvtx_content(v);

    // if all needed inputs are available
    if (list_in_set_p(vtxcontent_inputs(c), avails) &&
	!set_belong_p(exclude, v))
      computable = CONS(dagvtx, v, computable);

    // update availables: not needed under assert for no img reuse.
    // if (vtxcontent_out(c)!=entity_undefined)
    //  set_del_element(avails, avails, vtxcontent_out(c));
  }

  // cleanup
  gen_free_list(lv);
  return computable;
}

/* update sure/maybe set of live images after computing vertex v
 * that is the images that may be output from the pipeline.
 * this is voodoo...
 */
static void live_update(dagvtx v, set sure, set maybe)
{
  vtxcontent c = dagvtx_content(v);
  list ins = vtxcontent_inputs(c);
  int nins = gen_length(ins);
  entity out = vtxcontent_out(c);
  int nout = out!=entity_undefined? 1: 0;

  set all = set_make(set_pointer);
  set_union(all, sure, maybe);
  pips_assert("inputs are available", list_in_set_p(ins, all));
  set_free(all), all = NULL;

  switch (nins)
  {
  case 0:
    // no input image is used
    // maybe is unchanged
    // I guess it is not "NOP"...
    pips_assert("one output...", nout==1);
    set_union(maybe, maybe, sure);
    break;
  case 1:
    if (nout)
    {
      // 1 -> 1
      if (set_size(sure)==1 && !list_in_set_p(ins, sure))
      {
	set_assign_list(maybe, ins);
	set_union(maybe, maybe, sure);
      }
      else
      {
	set_append_list(maybe, ins);
      }
    }
    break;
  case 2:
    // any of the inputs may be kept
    set_assign_list(maybe, vtxcontent_inputs(c));
    break;
  default:
    pips_internal_error("unpexted number of inputs to vertex: %d", nins);
  }

  if (nout==1)
  {
    set_clear(sure);
    set_add_element(sure, sure, out);
  }
  // else sure is kept
  set_difference(maybe, maybe, sure);
}

/* returns an allocated set of vertices with live outputs.
 */
static set output_arcs(set vs)
{
  set out_nodes = set_make(set_pointer);
  // direct output nodes
  SET_FOREACH(dagvtx, v, vs)
  {
    if (vtxcontent_out(dagvtx_content(v))!=entity_undefined)
    {
      // this vertex produces an image
      bool is_needed = false;

      // it is needed if...
      list succs = dagvtx_succs(v);
      if (succs)
      {
	// some succs are not yet computed...
	FOREACH(dagvtx, vsucc, succs)
	  if (!set_belong_p(vs, vsucc))
	    is_needed = true;
      }
      else
	// or if there is no successor!?
	// ??? this is how outs are computed right now...
	is_needed = true;

      if (is_needed)
	set_add_element(out_nodes, out_nodes, v);
    }
  }
  return out_nodes;
}

/* how many output arcs from this set of vertices ?
 */
static int number_of_output_arcs(set vs)
{
  set out_nodes = output_arcs(vs);
  int n_arcs = set_size(out_nodes);
  set_free(out_nodes);
  return n_arcs;
}

/* return first vertex in the list which is compatible, or NULL if none.
 */
static dagvtx first_which_may_be_added
  (set current, // dagvtx
   list lv,     // dagvtx
   set sure,    // image entities
   __attribute__((unused)) set maybe)   // image entities
{
  set inputs = set_make(set_pointer);
  pips_assert("should be okay at first!", number_of_output_arcs(current)<=2);

  // output arcs from this subset
  // set outputs = output_arcs(current);

  FOREACH(dagvtx, v, lv)
  {
    pips_assert("not yet there", !set_belong_p(current, v));

    // when adding v to current:
    // the input acts are the one of outputs
    // plus the ones distinct inputs of v
    set_assign_list(inputs, vtxcontent_inputs(dagvtx_content(v)));

    if (set_size(inputs)==2 && !set_inclusion_p(sure, inputs))
      break;

    set_add_element(current, current, v);
    if (number_of_output_arcs(current) <= 2)
    {
      set_del_element(current, current, v);
      set_free(inputs);
      return v;
    }
    set_del_element(current, current, v);
  }
  // none found
  set_free(inputs);
  return NULL;
}

// DEBUG...
static string dagvtx_to_string(const dagvtx v)
{
  return itoa(dagvtx_number(v));
}

/* split dag dall into a list of pipelinable dags
 * which must be processed in that order (?)
 * side effect: dall is more or less consummed...
 */
static list /* of dags */ split_dag(dag dall)
{
  if (!single_image_assignement_p(dall))
    // well, it should work most of the time, so only a warning
    pips_user_warning("image reuse may result in subtly wrong code");

  int nvertices = gen_length(dag_vertices(dall));
  list ld = NIL, lcurrent = NIL;
  set
    current = set_make(set_pointer),
    removed = set_make(set_pointer),
    maybe = set_make(set_pointer),
    sure = set_make(set_pointer),
    avails = set_make(set_pointer);

  // well, there are not all available always!
  set_assign_list(maybe, dag_inputs(dall));
  set_assign(avails, maybe);
  int count = 0;

  do
  {
    pips_assert("argh...", count++<100);

    ifdebug(4) {
      pips_debug(4, "round %d:\n", count);
      set_fprint(stderr, "removed", removed,
		 (gen_string_func_t) dagvtx_to_string);
      set_fprint(stderr, "current", current,
		 (gen_string_func_t) dagvtx_to_string);
      set_fprint(stderr, "avails", avails,
		 (gen_string_func_t) entity_local_name);
      set_fprint(stderr, "maybe", maybe, (gen_string_func_t) entity_local_name);
      set_fprint(stderr, "sure", sure, (gen_string_func_t) entity_local_name);
    }

    list computables = get_computable_vertices(dall, removed, avails);
    gen_sort_list(computables,
		  (int(*)(const void*,const void*))dagvtx_priority);

    pips_assert("something must be computable if current is empty",
		computables || !set_empty_p(current));

    pips_debug(4, "%d computable vertices\n", (int) gen_length(computables));
    ifdebug(5) dagvtx_nb_dump(stderr, "computables", computables);

    // take the first one
    dagvtx vok = first_which_may_be_added(current, computables, sure, maybe);
    if (vok)
    {
      pips_debug(5, "extracting %" _intFMT "...\n", dagvtx_number(vok));
      set_add_element(current, current, vok);
      lcurrent = CONS(dagvtx, vok, lcurrent);
      set_add_element(removed, removed, vok);
      live_update(vok, sure, maybe);
      set_union(avails, sure, maybe);
    }

    // no stuff vertex can be added, or it was the last one
    if (!vok || set_size(removed)==nvertices)
    {
      // ifdebug(5)
      // set_fprint(stderr, "closing current", current, )
      pips_debug(5, "closing current...\n");

      lcurrent = gen_nreverse(lcurrent);
      // close current and build a deterministic dag...
      dag nd = make_dag(NIL, NIL, NIL);
      FOREACH(dagvtx, v, lcurrent)
      {
	// ??? can/should I do that?
	dag_remove_vertex(dall, v);
	dag_append_vertex(nd, v);
      }
      dag_set_inputs_outputs(dall);
      dag_set_inputs_outputs(nd);

      ifdebug(7) {
	dag_dump(stderr, "updated dall", dall);
	dag_dump(stderr, "pushed dag", nd);
      }

      ld = CONS(dag, nd, ld);

      gen_free_list(lcurrent), lcurrent = NIL;
      set_clear(current);
      // recompute image sets
      set_clear(sure);
      set_clear(maybe);
      // dall
      set_append_list(maybe, dag_inputs(dall));
      // and already extracted dags
      FOREACH(dag, d, ld)
      {
	FOREACH(dagvtx, v, dag_vertices(d))
	{
	  vtxcontent c = dagvtx_content(v);
	  set_append_list(maybe, vtxcontent_inputs(c));
	  if (vtxcontent_out(c)!=entity_undefined)
	    set_add_element(maybe, maybe, vtxcontent_out(c));
	}
      }
      set_assign(avails, maybe);
    }

    gen_free_list(computables);
  }
  while (dag_vertices(dall));

  // cleanup
  pips_assert("current empty list", !lcurrent);
  set_free(current);
  set_free(removed);
  set_free(sure);
  set_free(maybe);
  set_free(avails);

  pips_debug(5, "returning %d dags\n", (int) gen_length(ld));
  return gen_nreverse(ld);
}

/* @return whether to optimize AIPO call to function for SPoC.
*/
static bool freia_spoc_optimise(entity called)
{
  string fname = entity_local_name(called);
  return !same_string_p(fname, "freia_aipo_convolution") &&
    !same_string_p(fname, "freia_aipo_cast") &&
    !same_string_p(fname, "freia_aipo_fast_correlation");
}

/* returns whether the statement is a FREIA call.
 */
static bool statement_freia_call_p(statement s)
{
  // very partial as well
  instruction i = statement_instruction(s);
  if (instruction_call_p(i)) {
    call c = instruction_call(i);
    if (entity_freia_api_p(call_function(c)) &&
	// ??? should be take care later?
	freia_spoc_optimise(call_function(c)))
      return true;
    else if (ENTITY_ASSIGN_P(call_function(c))) {
      list la = call_arguments(c);
      pips_assert("2 arguments to assign", gen_length(la));
      syntax op2 = expression_syntax(EXPRESSION(CAR(CDR(la))));
      if (syntax_call_p(op2))
	return entity_freia_api_p(call_function(syntax_call(op2)))
	  // ??? later?
	  && freia_spoc_optimise(call_function(syntax_call(op2)));
    }
    else if (ENTITY_C_RETURN_P(call_function(c))) {
      list la = call_arguments(c);
      if (gen_length(la)==1) {
	syntax op = expression_syntax(EXPRESSION(CAR(la)));
	if (syntax_call_p(op))
	  return entity_freia_api_p(call_function(syntax_call(op)))
	    // ??? later?
	    && freia_spoc_optimise(call_function(syntax_call(op)));
      }
    }
  }
  return false;
}

static void set_append_vertex_statements(set s, list lv)
{
  FOREACH(dagvtx, v, lv)
    set_add_element(s, s,
        pstatement_statement(vtxcontent_source(dagvtx_content(v))));
}

/* generate helpers for statements in ls of module
 * output resulting functions in helper
 * @param module
 * @param ls list of statements for the dag
 * @param helper output file
 * @param number current helper dag count
 */
static void
freia_spoc_compile_calls
(string module, list /* of statements */ ls, FILE * helper_file, int number)
{
  pips_debug(3, "considering %d statements\n", (int) gen_length(ls));
  pips_assert("some statements", ls);

  // build full dag
  dag fulld = make_dag(NIL, NIL, NIL);
  FOREACH(statement, s, ls)
    dag_append_freia_call(fulld, s);
  dag_set_inputs_outputs(fulld);
  ifdebug(3) dag_dump(stderr, "fulld", fulld);

  string dag_name = strdup(cat("dag_", itoa(number), NULL));
  dag_dot_dump(module, dag_name, fulld);
  free(dag_name), dag_name = NULL;

  // remove copies if possible...
  dag_remove_useless_copies(fulld);

  dag_name = strdup(cat("dag_cleaned_", itoa(number), NULL));
  dag_dot_dump(module, dag_name, fulld);
  free(dag_name), dag_name = NULL;

  string fname_fulldag = strdup(cat(module, HELPER, itoa(number), NULL));

  // split dag in one-pipe dags.
  list ld = split_dag(fulld);
  pips_assert("fulld dag is empty", gen_length(dag_vertices(fulld))==0);

  // globally remaining statements
  set global_remainings = set_make(set_pointer);
  set_assign_list(global_remainings, ls);

  int n_pipes = 0;
  FOREACH(dag, d, ld)
  {
    set remainings = set_make(set_pointer);
    set_append_vertex_statements(remainings, dag_vertices(d));

    // generate a new helper function for dag d
    string fname_dag = strdup(cat(fname_fulldag, "_", itoa(n_pipes++), NULL));

    ifdebug(4) dag_dump(stderr, "d", d);
    dag_dot_dump(module, fname_dag, d);

    // one logical pipeline may be split because of overflow
    // the dag is updated to reflect that as a code generation side effect
    int split = 0;
    while (dag_vertices(d))
    {
      string fname_split = strdup(cat(fname_dag, "_", itoa(split++), NULL));
      list /* of expression */ lparams = NIL;

      string_buffer code = string_buffer_make(true);
      freia_spoc_pipeline(module, fname_split, code, d, &lparams);
      string_buffer_to_file(code, helper_file);
      string_buffer_free(&code);

      // statements that are not yet computed in d...
      set not_dones = set_make(set_pointer), dones = set_make(set_pointer);
      FOREACH(dagvtx, vs, dag_vertices(d))
	set_add_element(not_dones, not_dones,
		pstatement_statement(vtxcontent_source(dagvtx_content(vs))));
      set_difference(dones, remainings, not_dones);
      set_difference(remainings, remainings, dones);
      set_difference(global_remainings, global_remainings, dones);

      // replace first statement of dones in ls (so last in sequence)
      FOREACH(statement, sc, ls)
      {
	pips_debug(5, "in statement %" _intFMT "\n", statement_number(sc));

	if (set_belong_p(dones, sc))
	{
	  instruction old = statement_instruction(sc);
	  pips_debug(5, "sustituting...\n");
	  // substitute by call to helper
	  entity helper = make_empty_subroutine(fname_split); // or function?
	  call c = make_call(helper, lparams);
	  if (instruction_call_p(old) &&
	      ENTITY_C_RETURN_P(call_function(instruction_call(old))))
	  {
	    call ret = instruction_call(old);
	    gen_full_free_list(call_arguments(ret));
	    call_arguments(ret) = CONS(expression, call_to_expression(c), NIL);
	  }
	  else
	  {
	    statement_instruction(sc) = call_to_instruction(c);
	    free_instruction(old);
	  }
	  break;
	}
      }

      set_free(not_dones), not_dones = NULL;
      set_free(dones), dones = NULL;
      free(fname_split), fname_split = NULL;
    }

    set_free(remainings), remainings = NULL;
    free(fname_dag), fname_dag = NULL;
  }

  // no, copy statements are not done...
  // pips_assert("all statements done", set_empty_p(global_remainings));

  // cleanup
  set_free(global_remainings), global_remainings = NULL;
  free(fname_fulldag), fname_fulldag = NULL;
  free_dag(fulld);
  FOREACH(dag, dc, ld)
    free_dag(dc);
  gen_free_list(ld);
}

typedef struct {
  list /* of list of statements */ seqs;
} freia_spoc_info;

/** consider a sequence */
static bool sequence_flt(sequence sq, freia_spoc_info * fsip)
{
  list /* of statements */ ls = NIL;
  pips_debug(9, "considering sequence...\n");
  FOREACH(statement, s, sequence_statements(sq))
  {
    if (statement_freia_call_p(s))
      // ??? and possibly some other conditions...
      ls = CONS(statement, s, ls);
    else {
      // something else, interruption of the call sequence
      if (ls!=NIL) {
	ls = gen_nreverse(ls);
	fsip->seqs = CONS(list, ls, fsip->seqs);
	ls = NIL;
      }
    }
  }

  // end of sequence reached
  if (ls!=NIL) {
    ls = gen_nreverse(ls);
    fsip->seqs = CONS(list, ls, fsip->seqs);
    ls = NIL;
  }

  return true;
}

string freia_spoc_compile(string module, statement mod_stat)
{
  freia_spoc_info fsi;
  fsi.seqs = NIL;

  // collect freia api functions...
  // is it necessarily in a sequence?
  gen_context_multi_recurse(mod_stat, &fsi,
			    sequence_domain, sequence_flt, gen_null,
			    NULL);

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
