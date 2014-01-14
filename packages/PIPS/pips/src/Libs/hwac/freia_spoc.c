/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

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

#include "linear.h"

#include "ri.h"
#include "effects.h"
#include "ri-util.h"
#include "effects-util.h"
#include "properties.h"

#include "freia_spoc_private.h"
#include "hwac.h"

// help reduce code size:
#define T true
#define F false

static void comment(string_buffer code, spoc_hardware_type hw,
		    dagvtx v, int stage, int side, bool flip)
{
  sb_cat(code, "  // ", what_operation(hw), " ",
	 itoa(dagvtx_number(v)), " ", dagvtx_operation(v));
  sb_cat(code, " stage ", itoa(stage));
  if (hw!=spoc_type_alu) sb_cat(code, " side ", itoa(side));
  if (hw==spoc_type_alu && flip) sb_cat(code, " flipped");
  sb_cat(code, "\n");
}

/********************************************************* ALU CONFIGURATION */

// value to set operation, in same order as enum declaration
static const spoc_alu_op_t ALU_OP[] = {
  { alu_unused, alu_unused, NULL, F, F, F },
  // ADD
  { alu_add, alu_add, "SPOC_ALU_ADD_IN0_IN1", F, T, T },
  { alu_add_0cst, alu_add_1cst, "SPOC_ALU_ADD_IN0_CONST", T, T, F },
  { alu_add_1cst, alu_add_0cst, "SPOC_ALU_ADD_IN1_CONST", T, F, T },
  // ADDSAT
  { alu_addsat, alu_addsat, "SPOC_ALU_ADDSAT_IN0_IN1", F, T, T },
  { alu_addsat_0cst, alu_addsat_1cst, "SPOC_ALU_ADDSAT_IN0_CONST", T, T, F },
  { alu_addsat_1cst, alu_addsat_0cst, "SPOC_ALU_ADDSAT_IN1_CONST", T, F, T },
  // SUB
  { alu_sub_01, alu_sub_10, "SPOC_ALU_SUB_IN0_IN1", F, T, T },
  { alu_sub_10, alu_sub_01, "SPOC_ALU_SUB_IN1_IN0", F, T, T },
  { alu_sub_0cst, alu_sub_1cst, "SPOC_ALU_SUB_IN0_CONST", T, T, F },
  { alu_sub_1cst, alu_sub_0cst, "SPOC_ALU_SUB_IN1_CONST", T, F, T },
  { alu_sub_cst0, alu_sub_cst1, "SPOC_ALU_SUB_CONST_IN0", T, T, F },
  { alu_sub_cst1, alu_sub_cst0, "SPOC_ALU_SUB_CONST_IN1", T, F, T },
  // SUBSAT
  { alu_subsat_01, alu_subsat_10, "SPOC_ALU_SUBSAT_IN0_IN1", F, T, T },
  { alu_subsat_10, alu_subsat_01, "SPOC_ALU_SUBSAT_IN1_IN0", F, T, T },
  { alu_subsat_0cst, alu_subsat_1cst, "SPOC_ALU_SUBSAT_IN0_CONST", T, T, F },
  { alu_subsat_1cst, alu_subsat_0cst, "SPOC_ALU_SUBSAT_IN1_CONST", T, F, T },
  { alu_subsat_cst0, alu_subsat_cst1, "SPOC_ALU_SUBSAT_CONST_IN0", T, T, F },
  { alu_subsat_cst1, alu_subsat_cst0, "SPOC_ALU_SUBSAT_CONST_IN1", T, F, T },
  // ABSSUB
  { alu_abssub, alu_abssub, "SPOC_ALU_ABSSUB_IN0_IN1", F, T, T },
  { alu_abssub_0cst, alu_abssub_1cst, "SPOC_ALU_ABSSUB_IN0_CONST", T, T, F },
  { alu_abssub_1cst, alu_abssub_0cst, "SPOC_ALU_ABSSUB_IN1_CONST", T, F, T },
  // MUL
  { alu_mul, alu_mul, "SPOC_ALU_MUL_IN0_IN1", F, T, T },
  { alu_mul_0cst, alu_mul_1cst, "SPOC_ALU_MUL_IN0_CONST", T, T, F },
  { alu_mul_1cst, alu_mul_0cst, "SPOC_ALU_MUL_IN1_CONST", T, F, T },
  // DIV
  { alu_div_01, alu_div_10, "SPOC_ALU_DIV_IN0_IN1", F, T, T },
  { alu_div_10, alu_div_01, "SPOC_ALU_DIV_IN1_IN0", F, T, T },
  { alu_div_0cst, alu_div_1cst, "SPOC_ALU_DIV_IN0_CONST", T, T, F },
  { alu_div_1cst, alu_div_0cst, "SPOC_ALU_DIV_IN1_CONST", T, F, T },
  { alu_div_cst0, alu_div_cst1, "SPOC_ALU_DIV_CONST_IN0", T, T, F },
  { alu_div_cst1, alu_div_cst0, "SPOC_ALU_DIV_CONST_IN1", T, F, T },
  // LOG2
  { alu_log2_0, alu_log2_1, "SPOC_ALU_LOG2_IN0", F, T, F },
  { alu_log2_1, alu_log2_0, "SPOC_ALU_LOG2_IN1", F, F, T },
  // INF
  { alu_inf_01, alu_inf_01, "SPOC_ALU_INF_IN0_IN1", F, T, T },
  { alu_inf_0cst, alu_inf_1cst, "SPOC_ALU_INF_IN0_CONST", T, T, F },
  { alu_inf_1cst, alu_inf_0cst, "SPOC_ALU_INF_IN1_CONST", T, F, T },
  // SUP
  { alu_sup_01, alu_sup_01, "SPOC_ALU_SUP_IN0_IN1", F, T, T },
  { alu_sup_0cst, alu_sup_1cst, "SPOC_ALU_SUP_IN0_CONST", T, T, F },
  { alu_sup_1cst, alu_sup_0cst, "SPOC_ALU_SUP_IN1_CONST", T, F, T },
  // AND
  { alu_and, alu_and, "SPOC_ALU_AND_IN0_IN1", F, T, T },
  { alu_and_0cst, alu_and_1cst, "SPOC_ALU_AND_IN0_CONST", T, T, F },
  { alu_and_1cst, alu_and_0cst, "SPOC_ALU_AND_IN1_CONST", T, F, T },
  // OR
  { alu_or, alu_or, "SPOC_ALU_OR_IN0_IN1", F, T, T },
  { alu_or_0cst, alu_or_1cst, "SPOC_ALU_OR_IN0_CONST", T, T, F },
  { alu_or_1cst, alu_or_0cst, "SPOC_ALU_OR_IN1_CONST", T, F, T },
  // XOR
  { alu_xor, alu_xor, "SPOC_ALU_XOR_IN0_IN1", F, T, T },
  { alu_xor_0cst, alu_xor_1cst, "SPOC_ALU_XOR_IN0_CONST", T, T, F },
  { alu_xor_1cst, alu_xor_0cst, "SPOC_ALU_XOR_IN1_CONST", T, F, T },
  // NOT
  { alu_not_0, alu_not_1, "SPOC_ALU_NOT_IN0", F, T, F },
  { alu_not_1, alu_not_0, "SPOC_ALU_NOT_IN1", F, F, T },
  // MISC
  { alu_copy_cst, alu_copy_cst, "SPOC_ALU_COPY_CONST", T, F, F },
  // hmmm, the input images are exchanged from AIPO to SPOC?
  { alu_repcst_0, alu_repcst_1, "SPOC_ALU_REPLACE_IN1_IN0_CONST", T, T, T },
  { alu_repcst_1, alu_repcst_0, "SPOC_ALU_REPLACE_IN0_IN1_CONST", T, T, T }
  // and so on?
};

static const spoc_alu_op_t *get_spoc_alu_conf(spoc_alu_t alu)
{
  return &ALU_OP[alu];
}

/* generate a configuration for the ALU hardware component.
 */
static void spoc_alu_conf
  (spoc_alu_t alu,
   string_buffer body,
   __attribute__ ((__unused__)) string_buffer tail,
   int stage,
   bool flip,
   dagvtx orig,
   hash_table hp)
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
    sb_cat(body, "  si.alu[", s_stage, "][0].op = ", aluop->setting, ";\n");
    if (aluop->use_cst)
    {
      expression arg = EXPRESSION(CAR(freia_get_vertex_params(orig)));
      string s_var = hash_get(hp, arg);
      // res += "spocinstr.alu[%stage][0].constant = " + var + ";\n";
      sb_cat(body, "  sp.alu[", s_stage, "][0].constant = ", s_var, ";\n");
    }

    free(s_stage);
  }
}

/***************************************** POC & TH & MEASURE CONFIGURATIONS */

/* generate a configuration for a POC (morpho) hardware component.
 */
static void spoc_poc_conf
  (spoc_poc_t poc,
   string_buffer body,
   __attribute__ ((__unused__)) string_buffer tail,
   int stage,
   int side,
   dagvtx orig,
   hash_table hp) // expression -> parameter name
{
  if (poc.op!=spoc_poc_unused) // should not be called?
  {
    string
      s_stag = strdup(itoa(stage)),
      s_side = strdup(itoa(side));

    // only one argument is expected, the kernel
    expression arg = EXPRESSION(CAR(freia_get_vertex_params(orig)));
    string s_var = hash_get(hp, arg);

    // spocinstr.poc[0][0].op = SPOC_POC_ERODE;
    comment(body, spoc_type_poc, orig, stage, side, false);
    sb_cat(body, "  si.poc[", s_stag, "][", s_side, "].op = ");
    if (poc.op==spoc_poc_erode)
      sb_cat(body, "SPOC_POC_ERODE;\n");
    else if (poc.op==spoc_poc_dilate)
      sb_cat(body, "SPOC_POC_DILATE;\n");
    else if (poc.op==spoc_poc_conv)
      sb_cat(body, "SPOC_POC_CONV;\n");
    else
      pips_internal_error("unexpected poc operation %d", poc.op);

    // spocinstr.poc[0][0].grid = SPOC_POC_8_CONNEX;
    sb_cat(body, "  si.poc[", s_stag, "][", s_side, "].grid = ");
    if (poc.connectivity==6)
      sb_cat(body, "SPOC_POC_6_CONNEX;\n");
    else if (poc.connectivity==8)
      sb_cat(body, "SPOC_POC_8_CONNEX;\n");
    else
      pips_internal_error("unexpected poc connectivity %d", poc.connectivity);

    // for(i=0 ; i<9 ; i++) spocparam.poc[0][0].kernel[i] = kernel[i];
    sb_cat(body,
           "  for(i=0 ; i<9 ; i++)\n"
           "    sp.poc[", s_stag, "][", s_side, "].kernel[i] = ",
           s_var, "[i];\n");

    free(s_stag);
    free(s_side);
  }
}

/* generate a configuration for a threshold component.
 */
static void spoc_th_conf
  (string_buffer body,
   __attribute__ ((__unused__)) string_buffer tail,
   int stage,
   int side,
   dagvtx orig,
   hash_table hp)
{
  list largs = freia_get_vertex_params(orig);
  pips_assert("3 arguments to threshold", gen_length(largs)==3);
  expression
    e_inf = EXPRESSION(CAR(largs)),
    e_sup = EXPRESSION(CAR(CDR(largs))),
    e_bin = EXPRESSION(CAR(CDR(CDR(largs))));
  string
    s_inf = hash_get(hp, e_inf),
    s_sup = hash_get(hp, e_sup),
    s_bin = hash_get(hp, e_bin),
    s_stag = strdup(itoa(stage)),
    s_side = strdup(itoa(side));

  // spocinstr.th[0][0].op =
  //   (binarize==true) ? SPOC_TH_BINARIZE : SPOC_TH_NO_BINARIZE;
  // spocparam.th[0][0].boundmin = boundinf;
  // spocparam.th[0][0].boundmax = boundsup;
  comment(body, spoc_type_thr, orig, stage, side, false);
  sb_cat(body,
         "  si.th[", s_stag, "][", s_side, "].op = ",
         s_bin, "? SPOC_TH_BINARIZE : SPOC_TH_NO_BINARIZE;\n",
         "  sp.th[", s_stag, "][", s_side, "].boundmin = ", s_inf, ";\n",
         "  sp.th[", s_stag, "][", s_side, "].boundmax = ", s_sup, ";\n");

  free(s_stag);
  free(s_side);
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
   __attribute__ ((__unused__)) string_buffer body,
   string_buffer tail,
   int stage,
   int side,
   dagvtx orig,
   hash_table hp)
{
  int n = spoc_measure_n_params(measure);
  if (n==0) return;
  pips_assert("1 to 3 parameters", n>=1 && n<=3);

  string
    s_stage = strdup(itoa(stage)),
    s_side = strdup(itoa(side)),
    reduc = strdup(cat("reduc.measure[", s_stage, "][", s_side, "]."));

  list largs = freia_get_vertex_params(orig);

  pips_assert("argument list length is okay", (int) gen_length(largs)==n);

  string
    v_1 = hash_get(hp, EXPRESSION(CAR(largs))),
    v_2 = NULL, v_3 = NULL;
  if (n>=2) v_2 = hash_get(hp, EXPRESSION(CAR(CDR(largs))));
  if (n>=3) v_3 = hash_get(hp, EXPRESSION(CAR(CDR(CDR(largs)))));

  sb_cat(tail, "\n");
  comment(tail, spoc_type_mes, orig, stage, side, false);

  switch (measure) {
  case measure_max:
  case measure_min:
  case measure_max_coord:
  case measure_min_coord:
    sb_cat(tail, "  *", v_1, " = (int32_t) ", reduc,
	   measure==measure_min||measure==measure_min_coord?
	   "minimum": "maximum", ";\n");
    if (n>1)
    {
      sb_cat(tail, "  *", v_2, " = (uint32_t) ", reduc,
             measure==measure_min||measure==measure_min_coord?
             "min": "max", "_coord_x;\n",
             "  *", v_3, " = (uint32_t) ", reduc,
             measure==measure_min||measure==measure_min_coord?
             "min": "max", "_coord_y;\n");
    }
    break;
  case measure_vol:
    sb_cat(tail, "  *", v_1, " = (int32_t) ", reduc, "volume;\n");
    break;
  default:
    pips_internal_error("unexpected measure %d", measure);
  }

  free(s_stage);
  free(s_side);
  free(reduc);
}

/* basic configuration generation for a stage, depending on hw description
 * @returns the number of arguments expected?
 * what about their type?
 */
static void basic_spoc_conf
  (spoc_hardware_type op,
   string_buffer body,
   string_buffer tail,
   int stage,
   int side,
   bool flip,
   const spoc_hw_t * conf,
   dagvtx orig,
   hash_table hp)
{
  // only one call should be used for AIPO functions ?
  switch (op) {
  case spoc_type_poc:
    spoc_poc_conf(conf->poc[0], body, tail, stage, side, orig, hp);
    break;
  case spoc_type_alu:
    spoc_alu_conf(conf->alu, body, tail, stage, flip, orig, hp);
    break;
  case spoc_type_thr:
    spoc_th_conf(body, tail, stage, side, orig, hp);
    break;
  case spoc_type_mes:
    spoc_measure_conf(conf->mes[0], body, tail, stage, side, orig, hp);
    break;
  case spoc_type_inp:
  case spoc_type_nop:
    // copy, no code
    break;
  default:
    pips_internal_error("unexpected op type %s", what_operation(op));
  }
  if (op!=spoc_type_nop)
    sb_cat(body, "\n");
}

/* a data structure to describe a schedule for an operation
 */
typedef struct {
  entity image;     // output of this operation
  dagvtx producer;  // possibly NULL on input and output
  int stage;        // stage of production
  spoc_hardware_type level;  // nope | input < poc < alu < threshold < measure
  int side;  // 0 or 1
  bool flip; // whether alu op is flipped
  bool used; // whether the image has been used by an output
  bool just_used; // idem but at this stage
  // where this was used, to detect some particular cases?
  int used_stage;
  spoc_hardware_type used_level;
}  op_schedule;

static void init_op_schedule(op_schedule * op, dagvtx v, int side)
{
  op->image = v? dagvtx_image(v): NULL;
  op->producer = v;
  op->stage = 0;
  op->level = spoc_type_inp;
  op->side = side;
  op->flip = false;
  op->used = false;
  op->just_used = false;
  op->used_stage = -1; // never used
  op->used_level = spoc_type_inp;
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

/* is image needed?
 */
static bool image_is_needed(dagvtx prod, dag d, set todo)
{
  bool needed = false;
  if (prod)
  {
    if (gen_in_list_p(prod, dag_outputs(d)))
      needed = true;
    else
    {
      SET_FOREACH(dagvtx, v, todo)
      {
        list preds = dag_vertex_preds(d, v);
        needed = gen_in_list_p(prod, preds);
        gen_free_list(preds), preds = NIL;
        if (needed)
          break;
      }
    }
    pips_debug(7, "\"%"_intFMT"\" is%s needed\n",
               dagvtx_number(prod), needed? "": " not");
  }
  return needed;
}

// debug
static void
print_op_schedule(FILE * out, const string name, const op_schedule * op)
{
  fprintf(out, "sched '%s' = %s (%" _intFMT " %s) %d:%s:%d %s %sused (%d:%s)\n",
          name,
          op->image? entity_local_name(op->image): "NULL",
          dagvtx_number(op->producer),
          dagvtx_operation(op->producer),
          op->stage, what_operation(op->level), op->side,
          (op->level==spoc_type_alu)? (op->flip? "/": "."): "",
          op->used? "": "not ",
          op->used_stage, what_operation(op->used_level));
}

static bool check_mux_availibity(hash_table wiring, int stage, int mux)
{
  _int index = 2 * (stage*4+mux) + 1; // odds
  return !hash_defined_p(wiring, (void *) index);
}

static void set_mux(hash_table wiring, int stage, int mux)
{
  _int index = 2 * (stage*4+mux) + 1; // odds
  hash_put(wiring, (void *) index, (void *) index);
}

/* can I get out of stage on this side?
 */
static bool check_wiring_output(hash_table wiring, int stage, int side)
{
  return check_mux_availibity(wiring, stage, side? 3: 0);
}

static _int component_index(int stage, int level, int side)
{
  pips_assert("spoc component", level>=spoc_type_poc && level<= spoc_type_mes);
  if (level==spoc_type_alu) side=0;
  return 2 * (2 * (stage*4+level) + side) + 2; // even
}

static void set_component(hash_table wiring, int stage, int level, int side)
{
  _int index = component_index(stage, level, side);
  hash_put(wiring, (void *) index, (void *) index);
  // scratch everything on the path before the used component
  if (stage>0 && level==spoc_type_poc)
    set_component(wiring, stage-1, spoc_type_mes, side);
  if (level==spoc_type_mes)
    set_component(wiring, stage, spoc_type_thr, side);
}

static bool available_component
  (hash_table wiring, int stage, int level, int side)
{
  _int index = component_index(stage, level, side);
  return !hash_defined_p(wiring, (void *) index);
}

/* return the first stage after stage/level with both paths available
 */
static int find_first_crossing(hash_table wiring, int stage, int level)
{
  if (level>spoc_type_alu)
    stage++;
  while (!(check_wiring_output(wiring, stage, 0) &&
	   check_wiring_output(wiring, stage, 1)))
    stage++;
  return stage;
}

/* return the stage & side of the first component available
 * after (start_stage, level, side) which is the source of the used image,
 * for an operation of type target_level
 */
static void find_first_available_component(
  hash_table wiring,
  int start_stage, int level, int side, // image comes from
  int target_level,
  bool crossing, // whether to include a crossing on the path...
  int * pstage, int * pside)
{
  int stage = start_stage;
  int preferred, other;
  if (side>=0) preferred = side, other = 1 - side;
  // side==-1, alu...
  else preferred = 0, other = 1;
  if (target_level==spoc_type_alu)
    preferred=0, other=0;

  if (target_level<=level)
    stage++;

  if (crossing)
  {
    stage = find_first_crossing(wiring, stage, level);
    if (target_level<spoc_type_alu)
      stage++;
  }

  bool pok,
    skip_first_other =
    (stage==start_stage && target_level<spoc_type_alu) ||
    (stage==start_stage+1 && target_level<spoc_type_alu && level>spoc_type_alu);

  while (!(pok=available_component(wiring, stage, target_level, preferred)) &&
	 (skip_first_other ||
	  !available_component(wiring, stage, target_level, other)))
  {
    stage++;
    skip_first_other = false;
  }

  // hmmm...
  if (pok && level<=spoc_type_alu &&
      stage==start_stage+1 && target_level<spoc_type_alu)
  {
    if (!check_wiring_output(wiring, start_stage, preferred))
    {
      pips_assert("symmetric component must be available",
		  available_component(wiring, stage, target_level, other));
      pok = false;
    }
  }

  // return result
  *pstage = stage;
  *pside = pok? preferred: other;
}

/* generate wiring code for mux if necessary.
 * @param code generated
 * @param stage stage under consideration
 * @param mux multiplexer to set
 * @param value status for the multiplexer
 * @param wiring already performed wiring to check for double settings.
 */
static void
set_wiring(string_buffer code,
	   int stage, // 0 ..
	   int mux,   // 0 to 3
	   _int value, // 0 1
	   hash_table wiring)
{
  pips_debug(8, "%d %d %"_intFMT"\n", stage, mux, value);

  pips_assert("mux is available", check_mux_availibity(wiring, stage, mux));

  // code
  string s_stage = strdup(itoa(stage)), s_mux = strdup(itoa(mux));
  sb_cat(code, "  si.mux[", s_stage, "][", s_mux, "].op = "
	 "SPOC_MUX_IN", value? "1": "0", ";\n");
  free(s_stage);
  free(s_mux);

  // record
  set_mux(wiring, stage, mux);
}

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
  out->used = false;
  out->used_stage = -1;
  out->used_level = spoc_type_inp;
  in0->just_used = false;
  in1->just_used = false;

  ifdebug(6) {
    pips_debug(6, "scheduling (%" _intFMT " %s) %s\n",
	       dagvtx_number(op), dagvtx_operation(op),
	       vtxcontent_out(c)==entity_undefined? "":
	       entity_local_name(vtxcontent_out(c)));
    print_op_schedule(stderr, "in0", in0);
    print_op_schedule(stderr, "in1", in1);
  }

  spoc_hardware_type level = (spoc_hardware_type) vtxcontent_optype(c);

  // measurements is special case, because the image is not consummed...
  if (level == spoc_type_mes)
  {
    pips_assert("one input image", gen_length(vtxcontent_inputs(c))==1);
    entity measured = ENTITY(CAR(vtxcontent_inputs(c)));
    pips_assert("input image available",
		measured==in0->image || measured==in1->image);
    // go to the end of the stage of the image producer,
    // on same side...
    if (in0->image == in1->image)
    {
      // we have the same image, chose closest to mes, and prefer in0
      if (in0->level>=in1->level)
        *out = *in0, in0->used = (in0->level!=spoc_type_mes),
          in0->just_used = true, in1->used = false;
      else
        *out = *in1, in1->used = (in0->level!=spoc_type_mes),
          in1->just_used = true, in0->used = false;
    }
    else // take the available one, whatever
    {
      if (measured==in0->image)
        *out = *in0, in0->used = (in0->level!=spoc_type_mes),
          in0->just_used = true, in1->used = false;
      else
        *out = *in1, in1->used = (in0->level!=spoc_type_mes),
          in1->just_used = true, in0->used = false;
    }
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
      // ??? argh... the image may have been cleaned of not used any more...
      out->stage = max_stage(in0, in1);

      // hmmm... I should also check that the component is not already used...
      if (in0->image && in0->stage == out->stage && in0->level >= spoc_type_alu)
        out->stage++;
      if (in1->image && in1->stage == out->stage && in1->level >= spoc_type_alu)
        out->stage++;

      // ensure there is at least one path for the output
      while (!(check_wiring_output(wiring, out->stage, 0) ||
	       check_wiring_output(wiring, out->stage, 1)))
        out->stage++;

      // two paths are required if one of the other image is still needed.
      if ((in0->image && image_is_needed(in0->producer, computed, todo)) ||
          (in1->image && image_is_needed(in1->producer, computed, todo)))
      {
        while (! (check_wiring_output(wiring, out->stage, 0) &&
                  check_wiring_output(wiring, out->stage, 1)))
          out->stage++;
      }

      in0->used = false, in1->used = false;
      break;
    case 1: // ANYTHING, may have to chose a side.
    {
      entity dep = ENTITY(CAR(vtxcontent_inputs(c)));
      pips_assert("dependence is available",
                  dep==in0->image || dep==in1->image);
      op_schedule * used, * notused;

      // where is the needed image? put it in used
      if (in0->image == in1->image)
      {
        // if the same image is available,
        // chose the not already used one
        if (in0->used)
          used = in1, notused = in0;
        else if (in0->used)
          used = in0, notused = in1;
        // or the one which is already engaged on a side?
        else if (in0->level >= spoc_type_thr && level <= spoc_type_alu)
          used = in0, notused = in1;
        else if (in1->level >= spoc_type_thr && level <= spoc_type_alu)
          used = in1, notused = in0;
        // or the earliest one, or in0 by default
        else if (in1->stage<in0->stage ||
                 (in1->stage==in0->stage && in1->level < in0->level))
          used = in1, notused = in0;
        else
          used = in0, notused = in1;
      }
      else // just take the right one
      {
        used = dep==in0->image? in0: in1;
        notused = dep==in0->image? in1: in0;
      }

      pips_debug(7, "using %s (%d %s %d)\n",
                 entity_local_name(used->image),
                 used->stage, what_operation(used->level), used->side);

      // default stage and side...
      out->stage = used->stage;
      out->side = used->side;
      used->used = true;
      used->just_used = true;

      bool used_image_still_needed =
        image_is_needed(used->producer, computed, todo);

      // handle copy... if it is still there, it must be needed,
      // so just put it where it is available for later use.
      // ??? output copies should not be handled by the pipeline!
      if (level == spoc_type_nop)
      {
        // go to the end of the local path
        switch (used->level)
        {
        case spoc_type_inp:
        case spoc_type_poc:
          level = spoc_type_poc;
          break;
        case spoc_type_alu:
          level = spoc_type_alu;
          break;
        case spoc_type_thr:
        case spoc_type_mes:
          level = spoc_type_poc;
          out->stage++;
          break;
        default:
          pips_internal_error("unexpected spoc type %d", used->level);
        }

        // we need to put some space to extract the other image on another path
        if (used_image_still_needed)
        {
          if (level==spoc_type_poc)
          {
            // switch side if straight path is not available?
            if (!check_wiring_output(wiring, out->stage, out->side))
              out->side = 1 - out->side;
            pips_assert("available path",
                        check_wiring_output(wiring, out->stage, out->side));
            out->stage++;
          }
          else if (level==spoc_type_alu)
          {
            // choose one side, prefer 0
            if (check_wiring_output(wiring, out->stage, 0))
              out->side = 0;
            else if (check_wiring_output(wiring, out->stage, 1))
              out->side = 1;
            else
              pips_internal_error("no available path for copy of needed image");
            out->stage++;
            level = spoc_type_poc;
          }
          else
            pips_internal_error("should not get there");
        }
        else
          pips_debug(8, "copied image %s is not needed further\n",
                     entity_local_name(used->image));
        break;
      }

      pips_assert("not a copy", level!=spoc_type_nop);

      // ??? hmmm... an ALU used the operation???
      find_first_available_component
        (wiring,
         // source (level may be shifted for ALU/morph issue, see freia_50
         used->stage,
         used->used_stage!=-1 &&
         used->used_level==spoc_type_alu &&
         level==spoc_type_poc? spoc_type_poc: used->level, used->side,
         level, used_image_still_needed && used->image!=notused->image,
         &out->stage, &out->side);

      // tell which image was used
      used->used = true;
      notused->used = false;

      // record last use
      used->used_stage = out->stage;
      used->used_level = level;
      break;
    }
    case 2: // ALU
      // what if same image is used as both operands?
      if (!in1->image) *in1 = *in0;
      if (!in0->image) *in0 = *in1;
      pips_assert("alu binary operation", level==spoc_type_alu);
      out->stage = max_stage(in0, in1);
      // update input image status
      in0->used = true, in0->just_used = true;
      in1->used = true, in1->just_used = true;
      if (out->stage==in0->stage && in0->level >= spoc_type_alu)
        out->stage++;
      if (out->stage==in1->stage && in1->level >= spoc_type_alu)
        out->stage++;
      // including where last used
      in0->used_stage = out->stage;
      in0->used_level = level;
      in1->used_stage = out->stage;
      in1->used_level = level;
      break;
    default:
      pips_internal_error("shoud not get there");
    }
  }

  // common settings
  out->level = level;
  if (out->level == spoc_type_alu)
    out->side = -1;
  else if (out->side==-1)
  {
    // must set a size if not already chosen
    int stage = out->level<spoc_type_alu? out->stage-1: out->stage;
    if (check_wiring_output(wiring, stage, 0))
      out->side = 0;
    else if (check_wiring_output(wiring, stage, 1))
      out->side = 1;
    else
      pips_internal_error("no available path");
  }

  if (vtxcontent_out(c)!=entity_undefined)
  {
    out->image = vtxcontent_out(c);
    out->producer = op;
  }
  // else it is a measure, don't change...
  ifdebug(6) print_op_schedule(stderr, "out", out);
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
      set_wiring(code, stage, 1, 1, wiring);
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
         " [", in_stage, " ", what_operation(in->level));
  if (in->level!=spoc_type_alu)
    sb_cat(code, " ", in_side);
  sb_cat(code, "] -> [", out_stage, " ", what_operation(out->level));
  if (out->level!=spoc_type_alu)
    sb_cat(code, " ", out_side);
  sb_cat(code, "] ", itoa(dagvtx_number(out->producer)), " ",
         dagvtx_operation(out->producer), "\n");
  free(in_stage); free(out_stage); free(in_side); free(out_side);

  // let us wire!
  if (in->stage==out->stage)
  {
    switch (in->level)
    {
    case spoc_type_inp:
    case spoc_type_poc:
      if (out->level<=spoc_type_poc)
        pips_assert("same side if early", in->side==out->side);
      if (out->level<=spoc_type_alu)
        sb_cat(code, "  // nope\n");
      // ELSE thr or mes...
      else
      {
        generate_wiring_stage(code, in->stage, in->side, out->side, wiring);
        if (in->level == spoc_type_inp)
        {
          // the poc was passed by wiring...
          in->level = spoc_type_poc;
          set_component(wiring, in->stage, in->level, in->side);
        }
      }
      break;
    case spoc_type_alu:
      // out->level is after thr, so there is a side
      if (out->side!=-1)
        generate_wiring_stage(code, in->stage, -1, out->side, wiring);
      // else: X alu -> W alu, possibly because it operation is a copy
      break;
    case spoc_type_thr:
    case spoc_type_mes:
    case spoc_type_out:
      // no choice
      pips_assert("same side", in->side==out->side);
      sb_cat(code, "  // nope\n");
      break;
    case spoc_type_nop:
    default:
      pips_internal_error("shoud not get there");
    }

    // record operation flipping
    if (out->level == spoc_type_alu &&
        dagvtx_optype(out->producer)==spoc_type_alu) // skip copy
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
        out->flip = (in->side==1);
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
    op_schedule saved = *in;
    switch (in->level)
    {
    case spoc_type_inp:
    case spoc_type_poc:
      set_component(wiring, in->stage, spoc_type_poc, in->side);
    case spoc_type_alu:
      // chose prefered side
      if (in->level==spoc_type_alu)
        // try direct, or default to 0
        prefered = (out->side>=0)? out->side: 0;
      else // take the right side as soon as possible?
        // ??? what if the path is not possible later?
        prefered = (out->side==-1)? in->side: out->side;

      // try both cases
      if (check_wiring_output(wiring, in->stage, prefered))
      {
        generate_wiring_stage(code, in->stage, in->side, prefered, wiring);
        set_component(wiring, in->stage, spoc_type_mes, prefered);
        in->stage++;
        in->side = prefered;
        in->level = spoc_type_inp;
        generate_wiring(code, in, out, wiring);
      }
      else if (check_wiring_output(wiring, in->stage, 1 - prefered))
      {
        generate_wiring_stage(code, in->stage, in->side, 1 - prefered, wiring);
        set_component(wiring, in->stage, spoc_type_mes, 1 - prefered);
        in->stage++;
        in->side = 1 - prefered;
        in->level = spoc_type_inp;
        generate_wiring(code, in, out, wiring);
      }
      else
        pips_internal_error("no available path...");

      // we may still use the result for something else... others?
      if (saved.level<=spoc_type_alu && out->level<=spoc_type_poc &&
          // restore to the previous stage
          saved.stage==out->stage-1)
      {
        pips_debug(7, "restoring previous schedule...\n");
        *in = saved;
      }
      if (in->level == spoc_type_inp)
        in->level = spoc_type_poc; // the poc was passed by wiring
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
(string module,
 string function_name,
 // buffers which hold different parts of the code
 string_buffer code,
 const string_buffer head,
 const string_buffer body,
 const string_buffer tail,
 // number of images in & out
 int n_im_out, int n_im_in,
 // parts which may be needed
 bool some_reductions, bool some_kernels,
 // input/output parameters
 const string out0, const string out1, const string in0, const string in1)
{
  // function header: freia_error helper_function(freia_data_2d ...
  sb_cat(code,
         "\n"
         "// FREIA-SPoC helper function for module ", module, "\n"
         "freia_status ", function_name, "(\n");
  // first, image arguments
  if (n_im_out>0) sb_cat(code, "  " FREIA_IMAGE "o0");
  if (n_im_out>1) sb_cat(code, ",\n  " FREIA_IMAGE "o1");
  if (n_im_out!=0 && n_im_in>0) sb_cat(code, ",\n");
  if (n_im_in>0) sb_cat(code, "  const " FREIA_IMAGE "i0");
  if (n_im_in>1) sb_cat(code, ",\n  const " FREIA_IMAGE "i1");

  string_buffer_append_sb(code, head);

  // end of headers (some arguments may have been added by stages),
  // and begin the function body
  sb_cat(code, ")\n{\n" FREIA_SPOC_DECL);
  if (some_reductions)
    sb_cat(code,
           "  spoc_reduction reduc;\n"
           "  freia_reduction_results redres;\n");
  if (some_kernels) sb_cat(code, "  int i;\n");
  sb_cat(code, "\n");

  sb_cat(code,
         "  // init pipe to nop\n"
         "  spoc_init_pipe(&si, &sp, ", itoa(FREIA_DEFAULT_BPP), ");\n"
         "\n");

  string_buffer_append_sb(code, body);

  // generate actual call to the accelerator
  sb_cat(code, FREIA_SPOC_CALL_START);
  if (some_reductions)
    sb_cat(code, FREIA_SPOC_CALL_REDUC);
  sb_cat(code, FREIA_SPOC_CALL_END
         "  // actual call of spoc hardware\n"
         "  freia_cg_template_process_2i_2o",
         "(&param, ", out0, ", ", out1, ", ", in0, ", ", in1, ");\n",
         NULL);

  if (some_reductions)
    sb_cat(code,
           "\n"
           "  // get reductions\n"
           "  freia_cg_read_reduction_results(&redres);\n");

  string_buffer_append_sb(code, tail);

  sb_cat(code, "\n  return ret;\n}\n");
}

/* tell whether the image produced by prod is definitely consummed by v
 * given the global dag d and the set of vertex still to be computed todo.
 */
static bool is_consummed_by_vertex(dagvtx prod, dagvtx v, dag d, set todo)
{
  bool consummed;
  // no producer, no problem
  if (!prod)
    consummed = true;
  // the image is a global output, it must be kept
  else if (gen_in_list_p(prod, dag_outputs(d)))
    consummed = false;
  // the image is consummed by v
  else if (gen_in_list_p(v, dagvtx_succs(prod)))
  {
    set succs = set_make(set_pointer);
    set_assign_list(succs, dagvtx_succs(prod));
    set_del_element(succs, succs, v);
    // other todos expect this image?
    consummed = !set_intersection_p(succs, todo);
    set_free(succs);
  }
  else
    consummed = false;
  pips_debug(7, "%"_intFMT" is %sconsummed by %"_intFMT"\n",
             dagvtx_number(prod), consummed? "": "not ", dagvtx_number(v));
  return consummed;
}

/* generate a SPoC pipeline from a single DAG for module.
 *
 * @param module current
 * @param helper name of function to generate
 * @param code output here
 * @param dpipe dag to consider
 * @param lparams parameters to call helper
 * @return number of output images
 *
 * some side effects:
 * - if there is an overflow, dpipe updated with remaining vertices.
 * - accelerated statements are cleaned
 *
 * I should have really done an allocation on a infinite pipeline,
 * and then cut it, instead of this half measure to handle overflows,
 * but that would change a lot of things to go back.
 */
static _int freia_spoc_pipeline
(string module, string helper, string_buffer code, dag dpipe, list * lparams,
 const set output_images)
{
  hash_table wiring = hash_table_make(hash_int, 128);
  list outs = NIL;
  bool some_reductions = false, some_kernels = false;
  int pipeline_depth = get_int_property(spoc_depth_prop);

  pips_debug(3, "running on '%s' for %d operations\n",
             module, (int) (gen_length(dag_vertices(dpipe)) -
                            gen_length(dag_inputs(dpipe))));
  pips_assert("non empty dag", gen_length(dag_vertices(dpipe))>0);

  // build list of output entities from the pipe
  FOREACH(dagvtx, v, dag_outputs(dpipe))
    outs = CONS(entity, vtxcontent_out(dagvtx_content(v)), outs);
  outs = gen_nreverse(outs);

  // generate a helper function in some file
  int cst = 0;
  string_buffer
    head = string_buffer_make(true),
    body = string_buffer_make(true),
    tail = string_buffer_make(true);

  // first arguments are out & in images
  int n_im_in = gen_length(dag_inputs(dpipe));
  pips_assert("0, 1, 2 input images", n_im_in>=0 && n_im_in<=2);

  // piped images in helper
  string p_in0 = "NULL", p_in1 = "NULL", p_out0 = "NULL", p_out1 = "NULL";
  // piped images in function
  entity a_in0 = NULL, a_in1 = NULL;
  dagvtx v_in0 = NULL;

  op_schedule in0, in1, out;
  list /* of entity */ limg = NIL;

  int live_images = 0;

  // build input arguments/parameters/arguments
  if (n_im_in>=1)
  {
    v_in0 = DAGVTX(CAR(dag_inputs(dpipe)));
    a_in0 = vtxcontent_out(dagvtx_content(v_in0));
    p_in0 = "i0";
    init_op_schedule(&in0, v_in0, 0);
    limg = CONS(entity, a_in0, NIL);
    live_images++;
  }
  else
    init_op_schedule(&in0, NULL, 0);

  if (n_im_in==1 && gen_length(dagvtx_succs(v_in0))>1)
  {
    // help scheduling by providing the first input twice.
    init_op_schedule(&in1, v_in0, 1);
    p_in1 = "i0";
    live_images++;
  }
  else if (n_im_in>=2)
  {
    dagvtx v_in1 = DAGVTX(CAR(CDR(dag_inputs(dpipe))));
    a_in1 = vtxcontent_out(dagvtx_content(v_in1));
    p_in1 = "i1";
    init_op_schedule(&in1, v_in1, 1);
    limg = gen_nconc(limg, CONS(entity, a_in1, NIL));
    live_images++;
  }
  else
    init_op_schedule(&in1, NULL, 1);

  list vertices = gen_nreverse(gen_copy_seq(dag_vertices(dpipe)));
  set todo = set_make(set_pointer);
  FOREACH(dagvtx, vi, dag_inputs(dpipe))
    gen_remove(&vertices, vi);
  set_assign_list(todo, vertices);

  set skipped = set_make(set_pointer);

  // expression/variable -> parameter string name
  hash_table hparams = hash_table_make(hash_pointer, 0);

  // keep track of current stage for comments, and to detect overflows
  int stage = -1;

  FOREACH(dagvtx, v, vertices)
  {
    pips_debug(3, "considering vertex %"_intFMT"\n", dagvtx_number(v));

    // skip inputs
    if (dagvtx_number(v)==0) break;

    // on overflow...
    if (!set_empty_p(skipped))
    {
      // check for dependencies on skipped vertices...
      bool toskip = false;
      SET_FOREACH(dagvtx, sk, skipped)
      {
        // skip because of missing deps
        if (gen_in_list_p(v, dagvtx_succs(sk))) {
          pips_debug(7, "skipping %"_intFMT", deps on %"_intFMT"\n",
                     dagvtx_number(v), dagvtx_number(sk));
          toskip = true;
          break;
        }
      }
      // or no available output path, none created by consumming an image
      if (!toskip && dagvtx_succs(v) && live_images==2 &&
          !(is_consummed_by_vertex(in0.producer, v, dpipe, todo) ||
            is_consummed_by_vertex(in1.producer, v, dpipe, todo)))
      {
        pips_debug(7, "skipping %"_intFMT", no path\n",  dagvtx_number(v));
        toskip = true;
      }
      if (toskip) {
        set_add_element(skipped, skipped, v);
        continue;
      }
    }

    pips_debug(7, "dealing with vertex %" _intFMT ", %d to go, %d path\n",
	       dagvtx_number(v), set_size(todo), live_images);

    vtxcontent vc = dagvtx_content(v);
    pips_assert("there is a statement",
		pstatement_statement_p(vtxcontent_source(vc)));
    statement s = pstatement_statement(vtxcontent_source(vc));
    call c = freia_statement_to_call(s);
    // extract the api information now, because of scheduling side
    // effects on out so that out.provider may be changed on measures
    int optype = dagvtx_optype(v), opid = dagvtx_opid(v);
    const freia_api_t * api = get_freia_api(opid);
    pips_assert("AIPO function found", api!=NULL);

    set_del_element(todo, todo, v);

    // schedule...
    where_to_perform_operation(v, &in0, &in1, dpipe, todo, &out, wiring);

    // detect pipeline overflow
    if (out.stage >= pipeline_depth)
    {
      // pipeline is full, vi was not computed:
      pips_debug(3, "pipeline overflow on %" _intFMT "\n", dagvtx_number(v));
      set_add_element(todo, todo, v);

      // now, there may be other vertices which are computable...
      // but the splitting put there after this one in the vertex list
      // hmmm...
      set_add_element(skipped, skipped, v);
    }
    else // the vertex is added to the pipeline...
    {
      // record
      set_component(wiring, out.stage, out.level, out.side);

      // get needed parameters if any
      *lparams = gen_nconc(*lparams,
         freia_extract_params(opid, call_arguments(c), head, NULL,
                              hparams, &cst));

      some_reductions |= vtxcontent_optype(vc)==spoc_type_mes;
      some_kernels |= vtxcontent_optype(vc)==spoc_type_poc;

      // there may be regressions if it is a little bit out of order...
      // add a comment each time a new stage is started.
      if (out.stage!=stage)
      {
        stage = out.stage;
        sb_cat(body, "\n  // STAGE ", itoa(stage), "\n");
      }

      // generate wiring, which may change flip stage on ALU operation
      // special case if in0==in1 is needed? no, rely on just_used
      if (in0.just_used)
        generate_wiring(body, &in0, &out, wiring);

      if (in1.just_used)
        generate_wiring(body, &in1, &out, wiring);

      // generate stage code
      basic_spoc_conf(optype, body, tail, out.stage, out.side, out.flip,
                      &(api->spoc), v, hparams);

      bool
        in0_needed = image_is_needed(in0.producer, dpipe, todo),
        in1_needed = image_is_needed(in1.producer, dpipe, todo);

      // first, keep where it is...
      if (in0.stage==out.stage && in0.level==out.level && in0.side==out.side)
        in0 = out;
      else if (in1.stage==out.stage &&
               in1.level==out.level && in1.side==out.side)
        in1 = out;
      // else overwrite not needed used image
      else if (!in0_needed && in0.used)
        in0 = out;
      else if (!in1_needed && in1.used)
        in1 = out;
      // else, overwrite same image if not needed
      else if (!in0_needed || out.image==in0.image)
      {
        if (in1.image)
          in0 = out;
        else
          in1 = out; // rather keep it the available slot?
      }
      else if (!in1_needed || out.image==in1.image)
      {
        if (in0.image)
          in1 = out;
        else
          in0 = out; // idem
      }
      // the image is available twice which is needed
      else if (in0.image && in0.image == in1.image)
      {
        // we have the same image, overwrite the used one
        if (in0.used)
          in0 = out;
        else if (in1.used)
          in1 = out;
        // if none where used, overwrite the one on the current side
        // ??? in0.side == -1 && in1.side==-1 ?
        else if (in0.side == out.side)
          in0 = out;
        else if (in1.side == out.side)
          in1 = out;
        else if (out.side==-1)
          // just choose one?
          in0 = out;
        else
          pips_internal_error("should not get there (same image)?");
      }
      else
      {
        pips_assert("overflow mode", !set_empty_p(skipped));
        // ??? hmmm, we get there on overflow, if an input image is
        // still to be used, but 2 computed images are extracted.
        // what we are in effect doing is to modify the schedule,
        // possibly breaking the 2 live image property in the process,
        // in order to attempt to fill in the pipe a little more...
        if (out.image)
        {
          if (in0.just_used)
          {
            in0 = out;
            in0_needed = true;
          }
          else if (in1.just_used)
          {
            in1 = out;
            in1_needed = true;
          }
          else
            // not sure of the conditions above to choose between in0 and in1
            pips_internal_error("computed image not extracted...");
        }
      }

      // anyway, we must clean unuseful variables, because
      // the scheduling  on image entities to check deps (still ?)
      if (!in0_needed && in0.producer!=out.producer)
        init_op_schedule(&in0, NULL, 0);
      if (!in1_needed && in1.producer!=out.producer)
        init_op_schedule(&in1, NULL, 0);

      // update count of live images for overflow mode
      live_images = 0;
      if (image_is_needed(in0.producer, dpipe, todo)) live_images++;
      if (image_is_needed(in1.producer, dpipe, todo)) live_images++;
    }
  }

  set computed = set_make(set_pointer);
  set_assign_list(computed, vertices);
  set_difference(computed, computed, todo);

  // pipeline interruption?
  if (!set_empty_p(todo))
  {
    // what are the effective outputs in the middle of the pipeline
    list new_outs = NIL;
    if (image_is_needed(in1.producer, dpipe, todo) &&
        !gen_in_list_p(in1.producer, dag_inputs(dpipe)))
      new_outs = CONS(entity, in1.image, new_outs);
    if (image_is_needed(in0.producer, dpipe, todo) &&
        !gen_in_list_p(in0.producer, dag_inputs(dpipe)))
      // should also check that in0!=in1?
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
      if (pstatement_statement_p(vtxcontent_source(dagvtx_content(vr))))
        hwac_kill_statement(pstatement_statement
                            (vtxcontent_source(dagvtx_content(vr))));
      free_dagvtx(vr);
    }
  }

  // hmmm???
  dag_compute_outputs(dpipe, NULL, output_images, NIL, false);

  gen_free_list(vertices), vertices = NIL;
  set_free(computed), computed = NULL;

  // handle pipe outputs (may have been interrupted)
  int n_im_out = gen_length(outs);
  pips_assert("0, 1, 2 output images", n_im_out>=0 && n_im_out<=2);
  pips_assert("some input or output images", n_im_out || n_im_in);
  if (n_im_out==0)
  {
    sb_cat(body, "\n  // no output image\n");
  }
  else if (n_im_out==1)
  {
    entity imout = ENTITY(CAR(outs));

    // put producer in in0
    if (in1.image==in0.image)
    {
      pips_assert("image found", in0.image==imout);
      // choose the latest
      if (in1.stage>in0.stage || (in1.stage==in0.stage && in1.level>in0.level))
        out = in0, in0 = in1, in1 = out;
    }
    // else just take the right one
    else if (in1.image==imout && in0.image!=imout)
      out = in0, in0 = in1, in1 = out;
    else if (in0.image==imout)
      ; // ok
    else
      pips_internal_error("result image is not alive");

    out.image = imout;
    out.producer = NULL;
    out.level = spoc_type_out;
    out.stage = in0.stage;
    out.side = in0.side>=0? in0.side: 0; // choose 0 by default if alu

    // switch if side is not available
    if (in0.side<0 && !check_wiring_output(wiring, out.stage, out.side))
      out.side = 1 - out.side;

    sb_cat(body, "\n"
	   "  // output image ", entity_local_name(imout),
	   " on ", itoa(out.side), "\n");
    generate_wiring(body, &in0, &out, wiring);

    // do not trust the default initialisation of the paths
    in0 = out;
    out.stage = pipeline_depth-1;
    sb_cat(body, "\n  // fill in to the end...\n");
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

    pips_assert("output two results in two variables", out0!=out1);

    // make out0 match in0 and out1 match in1
    if (out0 != in0.image)
      out = in0, in0 = in1, in1 = out;
    pips_assert("results are available", out0==in0.image && out1==in1.image);

    pips_debug(7, "out0 %s out1 %s in0(%d.%d.%d) %s in1(%d.%d.%d) %s\n",
	       entity_local_name(out0), entity_local_name(out1),
	       in0.stage, in0.level, in0.side, entity_local_name(in0.image),
	       in1.stage, in1.level, in1.side, entity_local_name(in1.image));

    // the more advanced image in the pipe decides its output side
    bool in0_advanced = in0.stage>in1.stage ||
      (in0.stage==in1.stage && in0.level>in1.level);

    if (in0_advanced)
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
	   " on ", itoa(out0_side));
    sb_cat(body, " and image ", entity_local_name(out1),
	   " on ", itoa(out1_side), "\n");

    // extract the first one.
    out.image = out0;
    out.producer = NULL;
    out.level = spoc_type_out;
    out.side = out0_side;
    out.stage = out_stage;
    generate_wiring(body, &in0, &out, wiring);

    // do not trust the default initialisation of the pipeline
    in0 = out;
    out.stage = pipeline_depth-1;
    sb_cat(body, "\n  // fill in to the end...\n");
    generate_wiring(body, &in0, &out, wiring);

    // extract the seconde one
    sb_cat(body, "\n");
    out.image = out1;
    out.producer = NULL;
    out.level = spoc_type_out;
    out.side = out1_side;
    out.stage = out_stage;
    generate_wiring(body, &in1, &out, wiring);

    // do not trust the default initialisation of the pipeline
    in1 = out;
    out.stage = pipeline_depth-1;
    sb_cat(body, "\n  // fill in to the end...\n");
    generate_wiring(body, &in1, &out, wiring);
  }
  else
    pips_internal_error("only 0 to 2 output images!");

  // handle function image arguments
  freia_add_image_arguments(limg, lparams);

  // build whole code from various pieces
  freia_spoc_code_buildup(module, helper, code, head, body, tail,
                          n_im_out, n_im_in, some_reductions, some_kernels,
                          p_out0, p_out1, p_in0, p_in1);

  // cleanup
  gen_free_list(outs);
  set_free(todo);
  set_free(skipped);
  hash_table_free(wiring);
  string_buffer_free(&head);
  string_buffer_free(&body);
  string_buffer_free(&tail);

  return n_im_out;
}

/***************************************************************** MISC UTIL */

/* current list of computable that may be used to know about the
 * global context when comparing to vertices in "dagvtx_spoc_priority".
 */
static list dagvtx_spoc_priority_computables = NIL;
// idem for vertices computed in current pipe
static list dagvtx_spoc_priority_current = NIL;

#if 0
/* return the number of poc operations which have the same input node
 * as ref from the global list, including the ref itself.
 */
static int poc_count_same_inputs(const dagvtx ref)
{
  pips_assert("global list is defined", dagvtx_spoc_priority_computables);
  pips_assert("vertex is in list",
              gen_in_list_p(ref, dagvtx_spoc_priority_computables));
  pips_assert("vertex is a poc operation", dagvtx_optype(ref)==spoc_type_poc);

  dagvtx src = DAGVTX(CAR(vtxcontent_inputs(dagvtx_content(ref))));
  int n = 0;

  FOREACH(dagvtx, v, dagvtx_spoc_priority_computables)
    if (dagvtx_optype(v)==spoc_type_poc &&
        DAGVTX(CAR(vtxcontent_inputs(dagvtx_content(v)))) == src)
      n++;

  FOREACH(dagvtx, v, dagvtx_spoc_priority_current)
    if (dagvtx_optype(v)==spoc_type_poc &&
        DAGVTX(CAR(vtxcontent_inputs(dagvtx_content(v)))) == src)
      n++;

  pips_assert("at least myself", n>0);
  return n;
}
#endif

/* ? = Morpho(I); ? = ALU(I, ?);
 */
static bool erode_alu_shared_p(vtxcontent c1, vtxcontent c2)
{
  return
    vtxcontent_optype(c1)==spoc_type_poc &&
    vtxcontent_optype(c2)==spoc_type_alu &&
    gen_length(vtxcontent_inputs(c2))==2 &&
    gen_in_list_p(ENTITY(CAR(vtxcontent_inputs(c1))), vtxcontent_inputs(c2));
}

/* comparison function for sorting dagvtx in qsort,
 * this is deep voodoo, because the priority has an impact on
 * correctness? that should not be the case as only computations
 * allowed by dependencies are schedule.
 * non implemented functions are pushed late.
 * tells v1 < v2  =>  -1  =>  v1 BEFORE v2
 */
static int dagvtx_spoc_priority(const dagvtx * v1, const dagvtx * v2)
{
  string why = "none";
  int result = 0;
  vtxcontent
    c1 = dagvtx_content(*v1),
    c2 = dagvtx_content(*v2);

  // prioritize first scalar ops, measures and last copies
  // if there is only one of them
  if (vtxcontent_optype(c1)!=vtxcontent_optype(c2))
  {
    // non implemented operations
    if (vtxcontent_optype(c1)==spoc_type_sni)
      result = 1, why = "not";
    else if (vtxcontent_optype(c2)==spoc_type_sni)
      result = -1, why = "not";
    // then scalars operations first to remove (scalar) dependences
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
    // special morpho/alu case with a shared input image
    else if (erode_alu_shared_p(c1, c2))
      result = 1, why = "shared";
    else if (erode_alu_shared_p(c2, c1))
      result = -1, why = "shared";
  }

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

    // the decision process is mostly based on the number of inputs
    // images to the two compared operations.
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
      /*
        // unsuccessful attempt for freia_43
      if (vtxcontent_optype(c1)==spoc_type_poc)
    {
      int
        same1 = poc_count_same_inputs(*v1),
        same2 = poc_count_same_inputs(*v2);
      if (same1!=same2)
        // chose the one with more inputs
        result = same1-same2, why = "pocinputs";
      else
        // same as last case
        result = dagvtx_number(*v1) - dagvtx_number(*v2), why = "stats";
    }
    else */
      // if all else fails, rely on statement numbers.
      result = dagvtx_number(*v1) - dagvtx_number(*v2), why = "stats";
  }

  pips_debug(7, "%" _intFMT " %s %s %" _intFMT " %s (%s)\n",
	     dagvtx_number(*v1), dagvtx_operation(*v1),
	     result<0? ">": (result==0? "=": "<"),
	     dagvtx_number(*v2), dagvtx_operation(*v2), why);

  pips_assert("total order", v1==v2 || result!=0);
  return result;
}

/************************************************************ IMPLEMENTATION */

/* update sure/maybe set of live images after computing vertex v
 * that is the images that may be output from the pipeline.
 * this is voodoo...
 */
static void live_update(dag d, dagvtx v, set sure, set maybe)
{
  vtxcontent c = dagvtx_content(v);

  // skip "other" statements
  if (dagvtx_other_stuff_p(v))
    return;

  list ins = vtxcontent_inputs(c);
  int nins = gen_length(ins);
  entity out = vtxcontent_out(c);
  int nout = out!=entity_undefined? 1: 0;
  list preds = dag_vertex_preds(d, v);

  set all = set_make(set_pointer);
  set_union(all, sure, maybe);
  pips_assert("inputs are available", list_in_set_p(preds, all));
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
      if (set_size(sure)==1 && !list_in_set_p(preds, sure))
      {
        set_assign_list(maybe, preds);
        set_union(maybe, maybe, sure);
      }
      else
      {
        set_append_list(maybe, preds);
      }
    }
    break;
  case 2:
    // any of the inputs may be kept
    set_assign_list(maybe, preds);
    break;
  default:
    pips_internal_error("unpexted number of inputs to vertex: %d", nins);
  }

  if (nout==1)
  {
    set_clear(sure);
    set_add_element(sure, sure, v);
  }
  // else sure is kept
  set_difference(maybe, maybe, sure);

  // cleanup
  gen_free_list(preds), preds = NIL;
}

/* returns an allocated set of vertices with live outputs.
 */
static set output_arcs(dag d, set vs)
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
      if (gen_in_list_p(v, dag_outputs(d)))
        is_needed = true;
      else if (succs)
      {
        // some succs are not yet computed...
        FOREACH(dagvtx, vsucc, succs)
          if (!set_belong_p(vs, vsucc))
            is_needed = true;
      }
      // else there is no successor!?

      if (is_needed)
        set_add_element(out_nodes, out_nodes, v);
    }
  }
  return out_nodes;
}

/* how many output arcs from this set of vertices?
 */
static int number_of_output_arcs(dag d, set vs)
{
  set out_nodes = output_arcs(d, vs);
  int n_arcs = set_size(out_nodes);
  set_free(out_nodes);
  return n_arcs;
}

/* @return whether it is a 3x3 convolution
 */
static bool convolution_33(dagvtx v)
{
  if (freia_convolution_p(v)) {
    _int w, h;
    if (freia_convolution_width_height(v, &w, &h, false) && w==3 && h==3)
      return true;
  }
  return false;
}

/* does this dag contains a spoc non implemented operation?
 */
static bool dag_spoc_not_implemented(dag d)
{
  FOREACH(dagvtx, v, dag_vertices(d))
    if (dagvtx_optype(v)==spoc_type_sni ||
        // special handling of convolution
        (freia_convolution_p(v) && !convolution_33(v)))
      return true;
  return false;
}

/* return first vertex in the list which is compatible, or NULL if none.
 */
static dagvtx first_which_may_be_added
  (dag dall,
   set current, // of dagvtx
   list lv,     // of candidate dagvtx
   set sure,    // of dagvtx
   set maybe)   // image entities
{
  dagvtx chosen = NULL;
  set inputs = set_make(set_pointer);
  set current_output = output_arcs(dall, current);
  int n_outputs = set_size(current_output);
  pips_assert("should be okay at first!", n_outputs<=2);

  pips_debug(8, "#outputs = %d\n", n_outputs);

  // output arcs from this subset
  // set outputs = output_arcs(current);

  FOREACH(dagvtx, v, lv)
  {
    pips_debug(8, "considering vertex %"_intFMT"\n", dagvtx_number(v));
    pips_assert("not yet there", !set_belong_p(current, v));

    // some intermediate stuff
    if (dagvtx_optype(v)==spoc_type_oth)
    {
      chosen = v;
      break;
    }

    // no image is produce, so no image output is added...
    if (!dagvtx_image(v))
    {
      pips_assert("is a mesure", dagvtx_optype(v)==spoc_type_mes);
      chosen = v;
      break;
    }

    list preds = dag_vertex_preds(dall, v);
    set_assign_list(inputs, preds);
    pips_debug(8, "#inputs = %d\n", set_size(inputs));
    int npreds = gen_length(preds);
    gen_free_list(preds), preds = NIL;

    //pips_debug(7, "vertex %"_intFMT": %d preds\n", dagvtx_number(v), npreds);

    if (npreds==2) // binary alu operations...
    {
      if (!set_inclusion_p(sure, inputs))
        continue;
      // two lives, but the var reuse will break the pipe by hidding
      // the other live which is not used by the computation of v.
      if (n_outputs==2 && set_size(inputs)==1)
        continue;

      // another more general case from freia_68
      set all_lives = set_make(hash_pointer);
      set_intersection(all_lives, inputs, maybe);
      set_union(all_lives, current_output, all_lives);
      int n_lives = set_size(all_lives);
      set_free(all_lives);

      // cannot do, we would go over the limit...
      if (n_lives>2)
        continue;
    }

    set_add_element(current, current, v);
    int narcs_with_v = number_of_output_arcs(dall, current);
    set_del_element(current, current, v);

    if (narcs_with_v <= 2)
    {
      chosen = v;
      break;
    }
  }
  // none found
  set_free(inputs);
  return chosen;
}

/* split dag dall into a list of pipelinable dags
 * which must be processed in that order (?)
 * side effect: dall is more or less consummed...
 */
static list /* of dags */ split_dag(dag initial, const set output_images)
{
  // this may happen if an input image is also an output image...
  // pips_assert("no image reuse", single_image_assignement_p(initial));
  if (!single_image_assignement_p(initial))
    pips_user_warning("still some image reuse...\n");

  // ifdebug(1) pips_assert("initial dag ok", dag_consistent_p(initial));
  // if everything was removed by optimizations, there is nothing to do.
  if (dag_computation_count(initial)==0) return NIL;

  dag dall = copy_dag(initial);
  int nvertices = gen_length(dag_vertices(dall));
  list ld = NIL, lcurrent = NIL;
  set
    current = set_make(set_pointer),
    computed = set_make(set_pointer),
    maybe = set_make(set_pointer),
    sure = set_make(set_pointer),
    avails = set_make(set_pointer);

  // well, there are not all available always!
  set_assign_list(maybe, dag_inputs(dall));
  // set_assign(avails, maybe);
  set_assign(computed, maybe);
  int count = 0;

  do
  {
    count++;
    set_union(avails, sure, maybe);

    ifdebug(4) {
      pips_debug(4, "round %d:\n", count);
      set_fprint(stderr, "computed", computed,
                 (gen_string_func_t) dagvtx_to_string);
      set_fprint(stderr, "current", current,
                 (gen_string_func_t) dagvtx_to_string);
      set_fprint(stderr, "avails", avails,
                 (gen_string_func_t) dagvtx_to_string);
      set_fprint(stderr, "maybe", maybe, (gen_string_func_t) dagvtx_to_string);
      set_fprint(stderr, "sure", sure, (gen_string_func_t) dagvtx_to_string);
    }

    list computables = dag_computable_vertices(dall, computed, avails, current);
    dagvtx_spoc_priority_computables = computables;
    dagvtx_spoc_priority_current = lcurrent;
    gen_sort_list(computables,
		  (int(*)(const void*,const void*)) dagvtx_spoc_priority);
    dagvtx_spoc_priority_computables = NIL;
    dagvtx_spoc_priority_current = NIL;

    pips_assert("something must be computable if current is empty",
                computables || !set_empty_p(current));

    pips_debug(4, "%d computable vertices\n", (int) gen_length(computables));
    ifdebug(5) dagvtx_nb_dump(stderr, "computables", computables);

    // take the first one possible in the pipe, if any
    dagvtx vok =
      first_which_may_be_added(dall, current, computables, sure, maybe);

    if (vok && (dagvtx_optype(vok)==spoc_type_sni ||
                (freia_convolution_p(vok) && !convolution_33(vok)))
        && lcurrent)
      // extract non implemented nodes alone only!
      vok = NULL;

    if (vok)
    {
      pips_debug(5, "extracting %" _intFMT "...\n", dagvtx_number(vok));
      set_add_element(current, current, vok);
      lcurrent = CONS(dagvtx, vok, lcurrent);
      set_add_element(computed, computed, vok);
      live_update(dall, vok, sure, maybe);
      // set_union(avails, sure, maybe);
    }

    // no stuff vertex can be added, or it was the last one,
    // or it is not implemented and thus to be extracted "alone"
    if (!vok || set_size(computed)==nvertices ||
        (vok && (dagvtx_optype(vok)==spoc_type_sni ||
                 (freia_convolution_p(vok) && !convolution_33(vok)))))
    {
      // ifdebug(5)
      // set_fprint(stderr, "closing current", current, )
      pips_debug(5, "closing current...\n");
      pips_assert("current not empty", !set_empty_p(current));

      // gen_sort_list(lcurrent, (gen_cmp_func_t) dagvtx_ordering);
      lcurrent = gen_nreverse(lcurrent);
      // close current and build a deterministic dag...
      dag nd = make_dag(NIL, NIL, NIL);
      FOREACH(dagvtx, v, lcurrent)
      {
        pips_debug(7, "extracting node %" _intFMT "\n", dagvtx_number(v));
        dag_append_vertex(nd, copy_dagvtx_norec(v));
      }
      dag_compute_outputs(nd, NULL, output_images, NIL, false);
      dag_cleanup_other_statements(nd);

      ifdebug(7) {
        // dag_dump(stderr, "updated dall", dall);
        dag_dump(stderr, "pushed dag", nd);
      }

      // update global list of dags to return.
      ld = CONS(dag, nd, ld);

      // cleanup
      gen_free_list(lcurrent), lcurrent = NIL;
      set_clear(current);
      set_clear(sure);
      set_assign(maybe, computed);
    }

    gen_free_list(computables);
  }
  while (set_size(computed)!=nvertices);

  // checks and cleanup
  pips_assert("current empty list", !lcurrent);
  pips_assert("all vertices were computed", set_size(computed)==nvertices);
  set_free(current);
  set_free(computed);
  set_free(sure);
  set_free(maybe);
  set_free(avails);
  free_dag(dall);

  pips_debug(5, "returning %d dags\n", (int) gen_length(ld));
  return gen_nreverse(ld);
}

/* generate helpers for statements in ls of module
 * output resulting functions in helper, which may be empty in some cases.
 * @param module
 * @param ls list of statements for the dag (in reverse order)
 * @param helper output file
 * @param helpers created functions
 * @param number current helper dag count
 * @return list of intermediate images to allocate
 */
list freia_spoc_compile_calls
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

  // remove copies and duplicates if possible...
  // ??? maybe there should be an underlying transitive closure? not sure.
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

  string fname_fulldag = strdup(cat(module, "_spoc", HELPER, itoa(number)));

  // split dag in one-pipe dags.
  list ld = split_dag(fulld, output_images);

  // nothing to do!
  // it would have been interesting not to create thehelper file in this case.
  // if (ld==NIL) return;

  FOREACH(dag, dfix, ld)
    freia_hack_fix_global_ins_outs(fulld, dfix);

  // globally remaining statements
  set global_remainings = set_make(set_pointer);
  set_assign_list(global_remainings, ls);

  string_buffer code = string_buffer_make(true);

  int n_spoc_calls = 0;
  int n_pipes = 0;
  int stnb = -1;

  set stats = set_make(set_pointer), dones = set_make(set_pointer);

  FOREACH(dag, d, ld)
  {
    // skip non implemented stuff
    if (dag_spoc_not_implemented(d))
      continue;

    // fix connectivity
    dag_statements(stats, d);
    freia_migrate_statements(sq, stats, dones);
    set_union(dones, dones, stats);

    set remainings = set_make(set_pointer);
    set_append_vertex_statements(remainings, dag_vertices(d));
    // remove special inputs nodes
    // FOREACH(dagvtx, inp, dag_inputs(d))
    // set_del_element(remainings, remainings, inp);

    // generate a new helper function for dag d
    string fname_dag = strdup(cat(fname_fulldag, "_", itoa(n_pipes++)));

    ifdebug(4) dag_dump(stderr, "d", d);
    dag_dot_dump(module, fname_dag, d, NIL, NIL);

    // one logical pipeline may be split because of overflow
    // the dag is updated to reflect that as a code generation side effect
    int split = 0;
    while (dag_vertices(d))
    {
      // fix internal ins/outs, that are tempered with by split & overflows
      freia_hack_fix_global_ins_outs(fulld, d);

      ifdebug(6) {
        pips_debug(4, "dag for split %d\n", split);
        dag_dump(stderr, "d", d);
      }

      string fname_split = strdup(cat(fname_dag, "_", itoa(split++)));
      list /* of expression */ lparams = NIL;

      _int nout = freia_spoc_pipeline(module, fname_split, code, d, &lparams,
                                      output_images);
      stnb = freia_substitute_by_helper_call(d, global_remainings, remainings,
                                      ls, fname_split, lparams, helpers, stnb);
      // record (simple) signature
      hash_put(init, local_name_to_top_level_entity(fname_split), (void*) nout);
      free(fname_split), fname_split = NULL;
    }

    n_spoc_calls += split;
    fprintf(helper_file, "// split %d: %d cut%s\n",
            n_pipes-1, split, split>1? "s": "");

    set_free(remainings), remainings = NULL;
    free(fname_dag), fname_dag = NULL;
  }

  set_free(stats);
  set_free(dones);

  fprintf(helper_file, "// # SPOC calls: %d\n", n_spoc_calls);

  freia_insert_added_stats(ls, added_before, true);
  added_before = NIL;

  freia_insert_added_stats(ls, added_after, false);
  added_after = NIL;

  string_buffer_to_file(code, helper_file);
  string_buffer_free(&code);

  // cleanup
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
