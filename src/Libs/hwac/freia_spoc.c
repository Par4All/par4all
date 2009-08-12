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

#include "freia_spoc_private.h"

// help reduce code size:
#define T true
#define F false
#define cat concatenate

// out of order functions
static void dag_remove_vertex(dag, dagvtx);
static int hwac_freia_api_index(string);

/**************************************************************** FREIA AIPO */

typedef enum {
  spoc_type_none,
  spoc_type_alu,
  spoc_type_poc,
  spoc_type_thr,
  spoc_type_mes
} spoc_type;

/* @return a string which describes the type of operation
 * (i.e. the hardware used by the function).
 */
static string what_operation(_int type)
{
  switch (type)
  {
  case spoc_type_none: return "none";
  case spoc_type_alu: return "alu";
  case spoc_type_poc: return "poc";
  case spoc_type_mes: return "measure";
  case spoc_type_thr: return "threshold";
  default: return "unexpected value...";
  }
}

/* ??? beurk: I keep the operation as two ints for code regeneration.
 */
static void set_operation(freia_api_t * api, _int * type, _int * id)
{
  pips_assert("no type set", *type == spoc_type_none);

  // set type which is enough for staging?
  if (api->spoc.used & spoc_alu)
    *type = spoc_type_alu;
  else if (api->spoc.used & spoc_poc_0)
    *type = spoc_type_poc;
  else if (api->spoc.used & spoc_th_0)
    *type = spoc_type_thr;
  else if (api->spoc.used & spoc_measure_0)
    *type = spoc_type_mes;
  else
    *type = spoc_type_none; // e.g. for copy? (cast? convol? ...)

  // set details
  *id = hwac_freia_api_index(api->function_name);
}

#define NO_POC { { spoc_poc_unused, 0 }, { spoc_poc_unused, 0 } }
#define NO_MES { measure_none, measure_none }

// types used by AIPO parameters
#define TY_INT "int32_t"
#define TY_PIN "int32_t *"
#define TY_UIN "uint32_t"
#define TY_PUI "uint32_t *"

#define NO_PARAM { NULL, NULL, NULL }
// this may be moved into some "hwac_freia.[hc]"
/* !!! there are some underlying assumptions when using this structure:
 * only one of ALU/POC/MES is used per AIPO.
 * only the first of two POC or MES is used per AIPO.
 * ??? conversions between 8/16 bits images are definitely not handled here.
 */
static freia_api_t FREIA_AIPO_API[] = {
  {
    // ARITHMETIC
    // binary
    "freia_aipo_add", 1, 2, 0, 0, NO_PARAM, NO_PARAM,
    // spoc_hw_t
    {
      // parts used...
      spoc_input_0|spoc_input_1|spoc_output_0|spoc_alu,
      // poc[2]
      NO_POC,
      // alu
      alu_add,
      // global measures
      NO_MES
    }
  },
  { "freia_aipo_sub", 1, 2, 0, 0, NO_PARAM, NO_PARAM,
    { spoc_input_0|spoc_input_1|spoc_output_0|spoc_alu,
      NO_POC, alu_sub_01, NO_MES }
  },
  { "freia_aipo_mul", 1, 2, 0, 0, NO_PARAM, NO_PARAM,
    { spoc_input_0|spoc_input_1|spoc_output_0|spoc_alu,
      NO_POC, alu_mul, NO_MES }
  },
  { "freia_aipo_div", 1, 2, 0, 0, NO_PARAM, NO_PARAM,
    { spoc_input_0|spoc_input_1|spoc_output_0|spoc_alu,
      NO_POC, alu_div_01, NO_MES }
  },
  { "freia_aipo_addsat", 1, 2, 0, 0, NO_PARAM, NO_PARAM,
    { spoc_input_0|spoc_input_1|spoc_output_0|spoc_alu,
      NO_POC, alu_addsat, NO_MES }
  },
  { "freia_aipo_subsat", 1, 2, 0, 0, NO_PARAM, NO_PARAM,
    { spoc_input_0|spoc_input_1|spoc_output_0|spoc_alu,
      NO_POC, alu_subsat_01, NO_MES }
  },
  { "freia_aipo_absdiff", 1, 2, 0, 0, NO_PARAM, NO_PARAM,
    { spoc_input_0|spoc_input_1|spoc_output_0|spoc_alu,
      NO_POC, alu_abssub_01, NO_MES }
  },
  { "freia_aipo_inf", 1, 2, 0, 0, NO_PARAM, NO_PARAM,
    { spoc_input_0|spoc_input_1|spoc_output_0|spoc_alu,
      NO_POC, alu_inf_01, NO_MES }
  },
  { "freia_aipo_sup", 1, 2, 0, 0, NO_PARAM, NO_PARAM,
    { spoc_input_0|spoc_input_1|spoc_output_0|spoc_alu,
      NO_POC, alu_sup_01, NO_MES }
  },
  { "freia_aipo_and", 1, 2, 0, 0, NO_PARAM, NO_PARAM,
    { spoc_input_0|spoc_input_1|spoc_output_0|spoc_alu,
      NO_POC, alu_and, NO_MES }
  },
  { "freia_aipo_or", 1, 2, 0, 0, NO_PARAM, NO_PARAM,
    { spoc_input_0|spoc_input_1|spoc_output_0|spoc_alu,
      NO_POC, alu_or, NO_MES }
  },
  { "freia_aipo_xor", 1, 2, 0, 0, NO_PARAM, NO_PARAM,
    { spoc_input_0|spoc_input_1|spoc_output_0|spoc_alu,
      NO_POC, alu_xor, NO_MES }
  },
  // unary
  { "freia_aipo_not", 1, 1, 0, 0, NO_PARAM, NO_PARAM,
    { spoc_input_0|spoc_output_0|spoc_alu, NO_POC, alu_not_0, NO_MES }
  },
  { "freia_aipo_add_const", 1, 1, 0, 1, NO_PARAM, { TY_INT, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_alu, NO_POC, alu_add_0cst, NO_MES }
  },
  { "freia_aipo_inf_const", 1, 1, 0, 1, NO_PARAM, { TY_INT, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_alu, NO_POC, alu_inf_0cst, NO_MES }
  },
  { "freia_aipo_sup_const", 1, 1, 0, 1, NO_PARAM, { TY_INT, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_alu, NO_POC, alu_sup_0cst, NO_MES }
  },
  { "freia_aipo_sub_const", 1, 1, 0, 1, NO_PARAM, { TY_INT, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_alu, NO_POC, alu_sub_0cst, NO_MES }
  },
  { "freia_aipo_and_const", 1, 1, 0, 1, NO_PARAM, { TY_INT, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_alu, NO_POC, alu_and_0cst, NO_MES }
  },
  { "freia_aipo_or_const", 1, 1, 0, 1, NO_PARAM, { TY_INT, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_alu, NO_POC, alu_or_0cst, NO_MES }
  },
  { "freia_aipo_xor_const", 1, 1, 0, 1, NO_PARAM, { TY_INT, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_alu, NO_POC, alu_xor_0cst, NO_MES }
  },
  { "freia_aipo_addsat_const", 1, 1, 0, 1, NO_PARAM, { TY_INT, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_alu, NO_POC, alu_addsat_0cst, NO_MES }
  },
  { "freia_aipo_subsat_const", 1, 1, 0, 1, NO_PARAM, { TY_INT, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_alu, NO_POC, alu_subsat_0cst, NO_MES }
  },
  { "freia_aipo_absdiff_const", 1, 1, 0, 1, NO_PARAM, { TY_INT, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_alu, NO_POC, alu_abssub_0cst, NO_MES }
  },
  { "freia_aipo_mul_const", 1, 1, 0, 1, NO_PARAM, { TY_INT, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_alu, NO_POC, alu_mul_0cst, NO_MES }
  },
  { "freia_aipo_div_const", 1, 1, 0, 1, NO_PARAM, { TY_INT, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_alu, NO_POC, alu_div_0cst, NO_MES }
  },
  // nullary
  { "freia_aipo_set_constant", 1, 0, 0, 1, NO_PARAM, { TY_INT, NULL, NULL},
    { spoc_output_0|spoc_alu, NO_POC, alu_copy_cst, NO_MES }
  },
  // MISC
  // this one may be ignored?!
  { "freia_aipo_copy", 1, 1, 0, 0, NO_PARAM, NO_PARAM,
    { spoc_input_0|spoc_output_0, NO_POC, alu_unused, NO_MES }
  },
  { // not implemented by SPOC!
    "freia_aipo_cast", 1, 1, 0, 0, NO_PARAM, NO_PARAM,
    { spoc_nothing, NO_POC, alu_unused, NO_MES }
  },
  { "freia_aipo_set_constant", 1, 0, 0, 1, NO_PARAM, { TY_INT, NULL, NULL },
    { spoc_output_0|spoc_alu, NO_POC, alu_copy_cst, NO_MES }
  },
  { "freia_aipo_threshold", 1, 1, 0, 3, NO_PARAM, { TY_INT, TY_INT, TY_INT },
    { spoc_input_0|spoc_output_0|spoc_th_0, NO_POC, alu_unused, NO_MES }
  },
  // MORPHO
  { "freia_aipo_erode_6c", 1, 1, 0, 1, NO_PARAM, { TY_PIN, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_poc_0,
      { { spoc_poc_erode, 6 }, { spoc_poc_unused, 0 } }, alu_unused, NO_MES
    }
  },
  { "freia_aipo_dilate_6c", 1, 1, 0, 1, NO_PARAM, { TY_PIN, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_poc_0,
      { { spoc_poc_dilate, 6 }, { spoc_poc_unused, 0 } }, alu_unused, NO_MES
    }
  },
  { "freia_aipo_erode_8c", 1, 1, 0, 1, NO_PARAM, { TY_PIN, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_poc_0,
      { { spoc_poc_erode, 8 }, { spoc_poc_unused, 0 } }, alu_unused, NO_MES
    }
  },
  { "freia_aipo_dilate_8c", 1, 1, 0, 1,  NO_PARAM, { TY_PIN, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_poc_0,
      { { spoc_poc_dilate, 8 }, { spoc_poc_unused, 0 } }, alu_unused, NO_MES
    }
  },
  // MEASURES
  { "freia_aipo_global_min", 0, 1, 1, 0, { TY_PIN, NULL, NULL }, NO_PARAM,
    { spoc_input_0 | spoc_measure_0,
      NO_POC, alu_unused, { measure_min, measure_none }
    }
  },
  { "freia_aipo_global_max", 0, 1, 1, 0, { TY_PIN, NULL, NULL }, NO_PARAM,
    { spoc_input_0 | spoc_measure_0,
      NO_POC, alu_unused, { measure_max, measure_none }
    }
  },
  { "freia_aipo_global_min_coord", 0, 1, 3, 0,
    { TY_PIN, TY_PIN, TY_PIN }, NO_PARAM,
    { spoc_input_0 | spoc_measure_0,
      NO_POC, alu_unused, { measure_min_coord, measure_none }
    }
  },
  { "freia_aipo_global_max_coord", 0, 1, 3, 0,
    { TY_PIN, TY_PIN, TY_PIN }, NO_PARAM,
    { spoc_input_0 | spoc_measure_0,
      NO_POC, alu_unused, { measure_max_coord, measure_none }
    }
  },
  { "freia_aipo_global_vol", 0, 1, 1, 0,
    { TY_PIN, NULL, NULL }, NO_PARAM,
    { spoc_input_0 | spoc_measure_0,
      NO_POC, alu_unused, { measure_vol, measure_none }
    }
  },
  // ??? THIS ONE IS NOT ELEMENTARY
  /*
  { "freia_aipo_global_sad", 0, 2, 1, 0,
    { TY_PUI, NULL, NULL }, NO_PARAM,
    { spoc_input_0|spoc_input_1|spoc_measure_0,
      NO_POC, alu_unused, { measure_sad, measure_none }
    }
  },
  */
  // LINEAR
  // not implemented by SPOC!
  { "freia_aipo_convolution", 1, 1, 0, 3,
    NO_PARAM, { TY_PIN, TY_UIN, TY_UIN },
    { spoc_nothing, NO_POC, alu_unused, NO_MES }
  },
  // not implemented by SPOC!
  { "freia_aipo_fast_correlation", 1, 2, 0, 1,
    NO_PARAM, { TY_UIN, NULL, NULL },
    { spoc_nothing, NO_POC, alu_unused, NO_MES }
  },
  // last entry
  { NULL, 0, 0, 0, 0, NO_PARAM, NO_PARAM, { 0, NO_POC, alu_unused, NO_MES } }
};

/* @return the type expected for a parameter.
 * @param api structure
 * @param n parameter number (from 0)
 * @param input true for input, false for output
 */
/*
static string hwac_freia_api_param_type(freia_api_t * api, int n, bool input)
{
  if (input) {
    pips_assert("input param exists", n>=0 && n<api->arg_misc_in);
    return api->arg_in_types[n];
  }
  else {
    pips_assert("output param exists", n>=0 && n<api->arg_misc_out);
    return api->arg_out_types[n];
  }
}
*/

/* returns the index of the description of an AIPO function
 */
static int hwac_freia_api_index(string function)
{
  freia_api_t * api;
  for (api = FREIA_AIPO_API; api->function_name; api++)
    if (same_string_p(function, api->function_name))
      return api - FREIA_AIPO_API;
  return -1;
}

/* @returns the description of a FREIA AIPO API function.
 * may be moved elswhere. raise an error if not found.
 */
static freia_api_t * hwac_freia_api(string function)
{
  int index = hwac_freia_api_index(function);
  return index>=0? &FREIA_AIPO_API[index]: NULL;
}

/* @return new allocated variable name using provided prefix.
 * *params is incremented as a side effect.
 */
static string get_var(string prefix, int * params)
{
  return strdup(cat(prefix, itoa((*params)++), NULL));
}

/********************************************************* ALU CONFIGURATION */

// value to set operation, in same order as enum declaration
static const spoc_alu_op_t ALU_OP[] = {
  { alu_unused, NULL, F, F, F },
  // ADD
  { alu_add, "SPOC_ALU_ADD_IN0_IN1", F, T, T },
  { alu_add_0cst, "SPOC_ALU_ADD_IN0_CONST", T, T, F },
  { alu_add_1cst, "SPOC_ALU_ADD_IN1_CONST", T, F, T },
  // ADDSAT
  { alu_addsat, "SPOC_ALU_ADDSAT_IN0_IN1", F, T, T },
  { alu_addsat_0cst, "SPOC_ALU_ADDSAT_IN0_CONST", T, T, F },
  { alu_addsat_1cst, "SPOC_ALU_ADDSAT_IN1_CONST", T, F, T },
  // SUB
  { alu_sub_01, "SPOC_ALU_SUB_IN0_IN1", F, T, T },
  { alu_sub_10, "SPOC_ALU_SUB_IN1_IN0", F, T, T },
  { alu_sub_0cst, "SPOC_ALU_SUB_IN0_CONST", T, T, F },
  { alu_sub_1cst, "SPOC_ALU_SUB_IN1_CONST", T, F, T },
  // SUBSAT
  { alu_subsat_01, "SPOC_ALU_SUBSAT_IN0_IN1", F, T, T },
  { alu_subsat_10, "SPOC_ALU_SUBSAT_IN1_IN0", F, T, T },
  { alu_subsat_0cst, "SPOC_ALU_SUBSAT_IN0_CONST", T, T, F },
  { alu_subsat_1cst, "SPOC_ALU_SUBSAT_IN1_CONST", T, F, T },
  // ABSSUB
  { alu_abssub_01, "SPOC_ALU_ABSSUB_IN0_IN1", F, T, T },
  { alu_abssub_10, "SPOC_ALU_ABSSUB_IN1_IN0", F, T, T },
  { alu_abssub_0cst, "SPOC_ALU_ABSSUB_IN0_CONST", T, T, F },
  { alu_abssub_1cst, "SPOC_ALU_ABSSUB_IN1_CONST", T, F, T },
  // MUL
  { alu_mul, "SPOC_ALU_MUL_IN0_IN1", F, T, T },
  { alu_mul_0cst, "SPOC_ALU_MUL_IN0_CONST", T, T, F },
  { alu_mul_1cst, "SPOC_ALU_MUL_IN1_CONST", T, F, T },
  // DIV
  { alu_div_01, "SPOC_ALU_DIV_IN0_IN1", F, T, T },
  { alu_div_10, "SPOC_ALU_DIV_IN1_IN0", F, T, T },
  { alu_div_0cst, "SPOC_ALU_DIV_IN0_CONST", T, T, F },
  { alu_div_1cst, "SPOC_ALU_DIV_IN1_CONST", T, F, T },
  // INF
  { alu_inf_01, "SPOC_ALU_INF_IN0_IN1", F, T, T },
  { alu_inf_0cst, "SPOC_ALU_INF_IN0_CONST", T, T, F },
  { alu_inf_1cst, "SPOC_ALU_INF_IN1_CONST", T, F, T },
  // SUP
  { alu_sup_01, "SPOC_ALU_SUP_IN0_IN1", F, T, T },
  { alu_sup_0cst, "SPOC_ALU_SUP_IN0_CONST", T, T, F },
  { alu_sup_1cst, "SPOC_ALU_SUP_IN1_CONST", T, F, T },
  // AND
  { alu_and, "SPOC_ALU_AND_IN0_IN1", F, T, T },
  { alu_and_0cst, "SPOC_ALU_AND_IN0_CONST", T, T, F },
  { alu_and_1cst, "SPOC_ALU_AND_IN1_CONST", T, F, T },
  // OR
  { alu_or, "SPOC_ALU_OR_IN0_IN1", F, T, T },
  { alu_or_0cst, "SPOC_ALU_OR_IN0_CONST", T, T, F },
  { alu_or_1cst, "SPOC_ALU_OR_IN1_CONST", T, F, T },
  // XOR
  { alu_xor, "SPOC_ALU_XOR_IN0_IN1", F, T, T },
  { alu_xor_0cst, "SPOC_ALU_XOR_IN0_CONST", T, T, F },
  { alu_xor_1cst, "SPOC_ALU_XOR_IN1_CONST", T, F, T },
  // NOT
  { alu_not_0, "SPOC_ALU_NOT_IN0", F, T, F },
  { alu_not_1, "SPOC_ALU_NOT_IN1", T, F, T },
  { alu_copy_cst, "SPOC_ALU_COPY_CONST", T, F, F }
  // and so on
};

/* generate a configuration for the ALU hardware component.
 */
static void spoc_alu_conf
(spoc_alu_t alu,
 string_buffer head,
 __attribute__ ((__unused__)) string_buffer decl,
 string_buffer body,
 __attribute__ ((__unused__)) string_buffer tail,
 int stage,
 int * cst)
{
  if (alu!=alu_unused) // should not be called?
  {
    spoc_alu_op_t aluop = ALU_OP[alu];
    // fprintf(stderr, "alu: %d vs %d\n", alu, aluop.op);
    message_assert("alu operation found", alu==aluop.op);
    // res += "spocinstr.alu[%stage][0].op = " + aluop.setting + ";\n";
    string s_stage = strdup(itoa(stage));
    string_buffer_cat(body,
      "  si.alu[", s_stage, "][0].op = ", aluop.setting, ";\n", NULL);
    if (aluop.use_cst) {
      // string var = "c%d";
      string s_var = get_var("cst", cst);
      // (*decls) += ", int32_t " + var;
      string_buffer_cat(head, ", int32_t ", s_var, NULL);
      // res += "spocinstr.alu[%stage][0].constant = " + var + ";\n";
      string_buffer_cat(body,
	"  si.alu[", s_stage, "][0].constant = ", s_var, ";\n", NULL);
      free(s_var);
    }

    // ??? set links?
    string_buffer_cat(body,
		      "\n  // ??? what about setting links?\n"
		      "  si.mux[", s_stage, "][0].op = SPOC_MUX_IN1;\n",
		      "  si.mux[", s_stage, "][2].op = SPOC_MUX_IN0;\n",
		      NULL);

    free(s_stage);
  }
}

/***************************************** POC & TH & MEASURE CONFIGURATIONS */

/* generate a configuration for a POC (morpho) hardware component.
 */
static void spoc_poc_conf
(spoc_poc_t poc,
 string_buffer head,
 __attribute__ ((__unused__)) string_buffer decl,
 string_buffer body,
 __attribute__ ((__unused__)) string_buffer tail,
 int stage,
 int side,
 int * cst)
{
  if (poc.op!=spoc_poc_unused) // should not be called?
  {
    string
      s_stag = strdup(itoa(stage)),
      s_side = strdup(itoa(side)),
      s_var = get_var("kern", cst);

    // int32_t * var (kernel)
    string_buffer_cat(head, ", int32_t *", s_var, NULL);

    // spocinstr.poc[0][0].op = SPOC_POC_ERODE;
    string_buffer_cat(body,
      "  si.poc[", s_stag, "][", s_side, "].op = ", NULL);
    if (poc.op==spoc_poc_erode)
      string_buffer_append(body, "SPOC_POC_ERODE;\n");
    else if (poc.op==spoc_poc_dilate)
      string_buffer_append(body, "SPOC_POC_DILATE;\n");
    else
      pips_internal_error("unexpected poc operation %d", poc.op);

    // spocinstr.poc[0][0].grid = SPOC_POC_8_CONNEX;
    string_buffer_cat(body,
      "  si.poc[", s_stag, "][", s_side, "].grid = ", NULL);
    if (poc.connectivity==6)
      string_buffer_append(body, "SPOC_POC_6_CONNEX;\n");
    else if (poc.connectivity==8)
      string_buffer_append(body, "SPOC_POC_8_CONNEX;\n");
    else
      pips_internal_error("unexpected poc connectivity %d", poc.connectivity);

    // for(i=0 ; i<9 ; i++) spocparam.poc[0][0].kernel[i] = kernel[i];
    string_buffer_cat(body,
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
 __attribute__ ((__unused__)) string_buffer decl,
 string_buffer body,
 __attribute__ ((__unused__)) string_buffer tail,
 int stage,
 int side,
 int * cst)
{
  string
    s_inf = get_var("inf", cst),
    s_sup = get_var("sup", cst),
    s_bin = get_var("bin", cst),
    s_stag = strdup(itoa(stage)),
    s_side = strdup(itoa(side));

  // , int32_t cstX, int32_t cstY, bool binZ
  string_buffer_cat(head,
		    ", int32_t ", s_inf,
		    ", int32_t ", s_sup,
		    ", bool ", s_bin, NULL);

  // spocinstr.th[0][0].op =
  //   (binarize==true) ? SPOC_TH_BINARIZE : SPOC_TH_NO_BINARIZE;
  // spocparam.th[0][0].boundmin = boundinf;
  // spocparam.th[0][0].boundmax = boundsup;
  string_buffer_cat(body,
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
 __attribute__ ((__unused__)) string_buffer decl,
 __attribute__ ((__unused__)) string_buffer body,
 string_buffer tail,
 int stage,
 int side,
 int * cst)
{
  // pips_internal_error("not implemented yet");
  int n = spoc_measure_n_params(measure);
  if (n==0) return;
  // else something to do:
  pips_assert("1 to 3 parameters", n>=1 && n<=3);

  string
    s_stage = strdup(itoa(stage)),
    s_side = strdup(itoa(side)),
    reduc = strdup(cat("reduc.measure[", s_stage, "][", s_side, "].", NULL));

  string
    v_1 = get_var("red", cst), v_2 = NULL, v_3 = NULL;
  if (n>=2) v_2 = get_var("red", cst);
  if (n>=3) v_3 = get_var("red", cst);

  string_buffer_append(tail, "  // get reduction results\n");

  switch (measure) {
  case measure_max:
  case measure_min:
  case measure_max_coord:
  case measure_min_coord:
    string_buffer_cat(head, ", int32_t * ", v_1, NULL);
    string_buffer_cat(tail, "  *", v_1, " = (int32_t) ", reduc,
		      measure==measure_min||measure==measure_min_coord?
		      "minimum": "maximum", ";\n", NULL);
    if (n>1)
    {
      string_buffer_cat(head, ", uint32_t * ", v_2, ", uint32_t * ", v_3, NULL);
      string_buffer_cat(tail,
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
    string_buffer_cat(head, ", int32_t * ", v_1, NULL);
    string_buffer_cat(tail,
		      "  *", v_1, " = (int32_t) ", reduc, "volume;\n", NULL);
    break;
  default:
    pips_internal_error("unexpected measure %d", measure);
  }

  string_buffer_append(tail, "\n");

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

/* return malloc'ed function name for the helper...
 */
static string helper_function_name(string module, int n)
{
  return strdup(cat(module, HELPER, itoa(n), NULL));
}

/* return malloc'ed "foo.database/Src/%{module}_helper_functions.c"
 */
static string helper_file_name(string func_name)
{
  string src_dir = db_get_directory_name_for_module(WORKSPACE_SRC_SPACE);
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
   string_buffer decl,
   string_buffer body,
   string_buffer tail,
   int stage,
   int * cst, // numbering of arguments...
   spoc_hw_t * conf)
{
  // only one call should be used for AIPO functions ?
  if (conf->used & spoc_poc_0)
    spoc_poc_conf(conf->poc[0], head, decl, body, tail, stage, 0, cst);
  if (conf->used & spoc_poc_1)
    spoc_poc_conf(conf->poc[1], head, decl, body, tail, stage, 1, cst);
  if (conf->used & spoc_alu)
    spoc_alu_conf(conf->alu, head, decl, body, tail, stage, cst);
  if (conf->used & spoc_th_0)
    spoc_th_conf(head, decl, body, tail, stage, 0, cst);
  if (conf->used & spoc_th_1)
    spoc_th_conf(head, decl, body, tail, stage, 1, cst);
  if (conf->used & spoc_measure_0)
    spoc_measure_conf(conf->mes[0], head, decl, body, tail, stage, 0, cst);
  if (conf->used & spoc_measure_1)
    spoc_measure_conf(conf->mes[1], head, decl, body, tail, stage, 1, cst);
}

/* @return whether func is a measure.
 */
/*
static bool freia_measure_function_p(string func)
{
  freia_api_t * f = hwac_freia_api(func);
  if (f)
    return f->spoc.mes[0]!=measure_none || f->spoc.mes[1]!=measure_none ;
  // else
  return false;
}
*/

/* return the first computable vertex from dag d.
 * return NULL of none was found.
 */
/*
static dagvtx dag_get_first_computable_vertex(dag d)
{
  FOREACH(DAGVTX, v, dag_vertices(d))
  {
    if (dagvtx_preds(v)==NIL)
      return v;
  }
  return NULL;
}
*/

/* returns an allocated expression list of the parameters only
 * (i.e. do not include the input & output images).
 */
static list /* of expression*/ get_parameters(freia_api_t * api, list args)
{
  int skip = api->arg_img_in + api->arg_img_out;
  while (skip--) args = CDR(args);
  return gen_full_copy_list(args);
}

/* generate a SPoC pipeline from a single DAG for module.
 * @return the generated code
 * this function is not as clever as it should be.
 */
static string_buffer freia_spoc_pipeline
(string module,  // current module name
 string helper,  // function name to generate
 dag dpipe,      // AIPO function calls to pipeline
 list * lparams) // expressions for calling the helper as a side effect
{
  pips_debug(3, "running on '%s' for %u functions\n",
	     module, (unsigned int) gen_length(dag_vertices(dpipe)));
  pips_assert("non empty dag", gen_length(dag_vertices(dpipe))>0);

  // generate a helper function in some file
  int cst = 0;
  int stage = -1;
  string_buffer
    head = string_buffer_make(true),
    decl = string_buffer_make(true),
    body = string_buffer_make(true),
    tail = string_buffer_make(true);

  // first arguments are out & in images
  int n_im_in = gen_length(dag_inputs(dpipe));
  int n_im_out = gen_length(dag_outputs(dpipe));

  pips_assert("0, 1, 2 input images", n_im_in>=0 && n_im_in<=2);
  pips_assert("0, 1, 2 output images", n_im_out>=0 && n_im_out<=2);
  pips_assert("some input or output images", n_im_out || n_im_in);

  // generate header: freia_error helper_function(freia_data_2d ...
  string_buffer_cat(head,
		    "\n"
		    "// FREIA-SPoC helper function for module ", module, "\n"
		    "freia_status ", helper, "(", NULL);
  if (n_im_out>0) string_buffer_append(head, FREIA_IMAGE "o0");
  if (n_im_out>1) string_buffer_append(head, ", " FREIA_IMAGE "o1");
  if (n_im_out!=0) string_buffer_append(head, ", ");
  if (n_im_in>0) string_buffer_append(head, FREIA_IMAGE "i0");
  if (n_im_in>1) string_buffer_append(head, ", " FREIA_IMAGE "i1");

  string_buffer_append(decl, FREIA_SPOC_DECL);

  list vertices = gen_nreverse(gen_copy_seq(dag_vertices(dpipe)));

  FOREACH(DAGVTX, v, vertices)
  {
    statement s = pstatement_statement(vtxcontent_source(dagvtx_content(v)));
    call c = freia_statement_to_call(s);
    string func = entity_local_name(call_function(c));
    freia_api_t * api = hwac_freia_api(func);
    pips_assert("AIPO function found", api);

    *lparams = gen_nconc(*lparams, get_parameters(api, call_arguments(c)));

    // can take global measures from previous output
    // I'm not sure it should be here.
    // if (stage>0 && freia_measure_function_p(func)) stage--;
    // when is the next stage?
    stage++;

    string_buffer_cat(body, "\n  // stage ", itoa(stage), "\n", NULL);
    basic_spoc_conf(head, decl, body, tail, stage, &cst, &(api->spoc));

    // ??? what about links
  }
  gen_free_list(vertices);

  // add output & input images
  list limg = NIL;
  FOREACH(ENTITY, eo, dag_outputs(dpipe))
    limg = CONS(EXPRESSION, entity_to_expression(eo), limg);
  FOREACH(ENTITY, ei, dag_inputs(dpipe))
    limg = CONS(EXPRESSION, entity_to_expression(ei), limg);
  *lparams = gen_nconc(gen_nreverse(limg), *lparams);

  // end of headers (some arguments may have been added by stages)
  string_buffer_append(head, ")\n{\n");

  // generate actual call to the accelerator
  string_buffer_cat(body,
		    FREIA_SPOC_CALL
		    "  // actual call of spoc hardware\n"
		    "  freia_cg_template_process",
		    n_im_in==0? "": n_im_in==1? "_1i": "_2i",
		    n_im_out==0? "": n_im_out==1? "_1o": "_2o",
		    "(op, param",
		    // image parameters
		    n_im_out>=1? ", o0": "",
		    n_im_out>=2? ", o1": "",
		    n_im_in>=1? ", i0": "",
		    n_im_in>=2? ", i1": "",
		    ");\n"
		    "\n"
		    "  // get reductions\n"
		    "  freia_cg_read_reduction_results(&redres);\n"
		    "\n",
		    NULL);

  string_buffer_append(tail, "  return ret;\n}\n");

  string_buffer code = string_buffer_make(true);

  string_buffer_append_sb(code, head);
  string_buffer_free(&head);

  string_buffer_append_sb(code, decl);
  string_buffer_free(&decl);

  string_buffer_append_sb(code, body);
  string_buffer_free(&body);

  string_buffer_append_sb(code, tail);
  string_buffer_free(&tail);

  return code;
}

/************************************************************** DAG BUILDING */

static void entity_list_dump(FILE * out, string what, list l)
{
  fprintf(out, "%s: ", what);
  FOREACH(ENTITY, e, l)
    fprintf(out, " %s,", safe_entity_name(e));
  fprintf(out, "\n");
}

static _int vertex_number(const dagvtx v)
{
  return statement_number
    (pstatement_statement(vtxcontent_source(dagvtx_content(v))));
}

static void dagvtx_nb_dump(FILE * out, string what, list l)
{
  fprintf(out, "%s: ", what);
  FOREACH(DAGVTX, v, l)
    fprintf(out, " %" _intFMT ",", vertex_number(v));
  fprintf(out, "\n");
}

/* for dag debug.
 */
static void dagvtx_dump(FILE * out, dagvtx v)
{
  fprintf(out, "vertex %" _intFMT " (%p)\n", vertex_number(v), v);
  dagvtx_nb_dump(out, "  preds", dagvtx_preds(v));
  dagvtx_nb_dump(out, "  succs", dagvtx_succs(v));
  vtxcontent c = dagvtx_content(v);
  statement s = pstatement_statement(vtxcontent_source(c));
  fprintf(out,
	  "  optype: %s\n"
	  "  opid: %" _intFMT "\n"
	  "  source: %" _intFMT "/%" _intFMT "\n",
	  what_operation(vtxcontent_optype(c)),
	  vtxcontent_opid(c),
	  statement_number(s),
	  statement_ordering(s));
  entity_list_dump(out, "  inputs", vtxcontent_inputs(c));
  fprintf(out, "  output: %s\n", safe_entity_name(vtxcontent_out(c)));
  // to be continued...
}

/* for dag debug.
 */
static void dag_dump(FILE * out, string what, dag d)
{
  fprintf(out, "dag '%s' (%p):\n", what, d);

  entity_list_dump(out, "inputs", dag_inputs(d));
  entity_list_dump(out, "outputs", dag_outputs(d));

  FOREACH(DAGVTX, vx, dag_vertices(d)) {
    dagvtx_dump(out, vx);
    fprintf(out, "\n");
  }

  fprintf(out, "\n");
}

/* (re)compute the list of *GLOBAL* input & output images for this dag
 * ??? BUG the output is rather an approximation
 * should rely on used defs or out effects for the underlying sequence.
 */
static void dag_set_inputs_outputs(dag d)
{
  list ins = NIL, outs = NIL, lasts = NIL;
  // vertices are in reverse computation order...
  FOREACH(DAGVTX, v, dag_vertices(d))
  {
    vtxcontent c = dagvtx_content(v);
    entity out = vtxcontent_out(c);

    if (out!=entity_undefined)
    {
      // all out variables
      outs = gen_once(out, outs);

      // if it is not used, then it is a production...
      if (!gen_in_list_p(out, ins))
	lasts = gen_once(out, lasts);

      // remove inputs that are produced locally
      gen_remove(&ins, out);
    }

    FOREACH(ENTITY, i, vtxcontent_inputs(c))
      ins = gen_once(i, ins);
  }

  // keep argument order...
  ins = gen_nreverse(ins);

  ifdebug(9)
  {
    dag_dump(stderr, "debug dag_set_inputs_outputs", d);
    entity_list_dump(stderr, "computed ins", ins);
    entity_list_dump(stderr, "computed outs", outs);
    entity_list_dump(stderr, "computed lasts", lasts);
  }

  gen_free_list(dag_inputs(d));
  dag_inputs(d) = ins;

  gen_free_list(dag_outputs(d));
  dag_outputs(d) = lasts;

  gen_free_list(outs);
}

#define FREIA_PREFIX "freia_aipo_"

/* returns whether the entity is a freia API function.
 */
static bool entity_freia_api_p(entity f)
{
  // very partial...
  return strncmp(entity_local_name(f), FREIA_PREFIX, strlen(FREIA_PREFIX))==0;
}

static list fs_expression_list_to_entity_list(list /* of expression */ args)
{
  list /* of entity */ lent = NIL;
  FOREACH(EXPRESSION, ex, args)
  {
    syntax s = expression_syntax(ex);
    pips_assert("is a ref", syntax_reference_p(s));
    reference r = syntax_reference(s);
    pips_assert("simple ref", reference_indices(r)==NIL);
    lent = CONS(ENTITY, reference_variable(r), lent);
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

/* return (first) producer vertex or NULL of none found.
 */
static dagvtx get_producer(dag d, entity e)
{
  FOREACH(DAGVTX, v, dag_vertices(d))
  {
    vtxcontent c = dagvtx_content(v);
    // the image may kept within a pipe
    if (vtxcontent_out(c)==e)
      return v;
    // ??? BUT def/use of scalars cannot be used within a pipe...
    if (gen_in_list_p(e, vtxcontent_outparams(c)))
      return v;
  }
  return NULL; // it is an external parameter
}

/* remove vertex v from dag d.
 */
static void dag_remove_vertex(dag d, dagvtx v)
{
  pips_assert("vertex is in dag", gen_in_list_p(v, dag_vertices(d)));

  // remove from vertex list
  gen_remove(&dag_vertices(d), v);

  // unlink from predecessors...
  FOREACH(DAGVTX, p, dagvtx_preds(v))
    gen_remove(&dagvtx_succs(p), v);

  // unlink from successors
  FOREACH(DAGVTX, s, dagvtx_succs(v))
    gen_remove(&dagvtx_preds(s), v);

  // unlink vertex itself
  gen_free_list(dagvtx_preds(v)), dagvtx_preds(v) = NIL;
  gen_free_list(dagvtx_succs(v)), dagvtx_succs(v) = NIL;

  // what about updating ins & outs?
}

/* append new vertex nv to dag d.
 */
static void dag_append_vertex(dag d, dagvtx nv)
{
  pips_assert("vertex not in dag", !gen_in_list_p(nv, dag_vertices(d)));

  vtxcontent c = dagvtx_content(nv);
  list ins = vtxcontent_inputs(c);

  FOREACH(ENTITY, e, ins)
  {
    dagvtx pv = get_producer(d, e);
    if (pv) {
      dagvtx_succs(pv) = gen_once(nv, dagvtx_succs(pv));
      dagvtx_preds(nv) = gen_once(pv, dagvtx_preds(nv));
    }
    // inputs are computed later.
    // else dag_inputs(d) = gen_once(e, dag_inputs(d));
  }
  dag_vertices(d) = CONS(DAGVTX, nv, dag_vertices(d));
  // ??? what about scalar deps?
}

/* append statement s to dag d
 */
static void dag_append_freia_call(dag d, statement s)
{
  call c = freia_statement_to_call(s);
  entity called = call_function(c);
  pips_assert("FREIA API call", entity_freia_api_p(called));
  freia_api_t * api = hwac_freia_api(entity_local_name(called));
  pips_assert("some api", api!=NULL);
  list /* of entity */ args =
    fs_expression_list_to_entity_list(call_arguments(c));

  // extract arguments
  entity out = entity_undefined;
  pips_assert("one out image max for an AIPO", api->arg_img_out<=1);
  if (api->arg_img_out==1)
    out = extract_fist_item(&args);
  list ins = gen_list_head(&args, api->arg_img_in);
  list outp = gen_list_head(&args, api->arg_misc_out);
  list inp = gen_list_head(&args, api->arg_misc_in);
  pips_assert("no more arguments", gen_length(args)==0);

  vtxcontent cont =
    make_vtxcontent(0, 0, make_pstatement(s),
		    ins, out, inp, outp, -1, false, false, NIL);
  set_operation(api, &vtxcontent_optype(cont), &vtxcontent_opid(cont));

  dagvtx nv = make_dagvtx(NIL, NIL, cont);
  dag_append_vertex(d, nv);
  // ??? TODO update global outs later?
}

/************************************************************ IMPLEMENTATION */

/* ??? I'm unsure about what happens to dead code in the pipeline...
 */

/* ??? BUG the code may not work properly when image variable are
 * reused within a pipe. The underlying issues are subtle and
 * would need careful thinking: the initial code is correct, the dag
 * representation is correct, but the generated code may reorder
 * dag vertices so that reused variables are made to interact one with
 * the other. Maybe I should recreate output variables in the generated code
 * for every pipeline. this would imply a cleanup phase to removed
 * unused images at the end. I would really need an SSA form on
 * images? this function checks the assumption before proceeding further.
 */
static bool single_image_assignement_p(dag d)
{
  set outs = set_make(set_pointer);
  FOREACH(DAGVTX, v, dag_vertices(d))
  {
    vtxcontent c = dagvtx_content(v);
    entity out = vtxcontent_out(c);
    if (out!=entity_undefined)
    {
      if (set_belong_p(outs, out)) {
	set_free(outs);
	return false;
      }
      set_add_element(outs, outs, out);
    }
  }
  set_free(outs);
  return true;
}

/* return the vertices which may be computed from the list of
 * available images, excluding vertices in exclude.
 * return a list for determinism.
 */
static list /* of dagvtx */
get_computable_vertices(dag d, set exclude, set /* of entity */ avails)
{
  list computable = NIL;

  // hmmm... reverse the list to handle implicit dependencies...
  // where, there is an assert() to check that it does not happen.
  list lv = gen_nreverse(gen_copy_seq(dag_vertices(d)));

  FOREACH(DAGVTX, v, lv)
  {
    vtxcontent c = dagvtx_content(v);

    // if all needed inputs are available
    if (list_in_set_p(vtxcontent_inputs(c), avails) &&
	!set_belong_p(exclude, v))
      computable = CONS(DAGVTX, v, computable);

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
	FOREACH(DAGVTX, vsucc, succs)
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

  FOREACH(DAGVTX, v, lv)
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

/* comparison function for sorting dagvtx in qsort,
 * which rely on the underlying statement number.
 */
static int dagvtx_compare(const dagvtx * v1, const dagvtx * v2)
{
  return vertex_number(*v1) - vertex_number(*v2);
}

/* this is voodoo...
 */
static int dagvtx_priority(const dagvtx * v1, const dagvtx * v2)
{
  int
    l1 = (int) gen_length(vtxcontent_inputs(dagvtx_content(*v1))),
    l2 = (int) gen_length(vtxcontent_inputs(dagvtx_content(*v2)));
  if (l1!=l2)
    // the more images are needed, the earlier
    return l2-l1;
  else
    // otherwise use the statement numbers
    return dagvtx_compare(v1, v2);
}

// DEBUG...
static string dagvtx_to_string(const dagvtx v)
{
  return itoa(vertex_number(v));
}

/* split dag dall into a list of pipelinable dags
 * which must be processed in that order (?)
 * side effect: dall is more or less consummed...
 */
static list /* of dags */ split_dag(dag dall)
{
  pips_assert("single assignement of images in sequence",
	      single_image_assignement_p(dall));

  int nvertices = gen_length(dag_vertices(dall));
  list ld = NIL;
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
		 (string(*)(void*)) dagvtx_to_string);
      set_fprint(stderr, "current", current,
		 (string(*)(void*)) dagvtx_to_string);
      set_fprint(stderr, "avails", avails, (string(*)(void*))entity_local_name);
      set_fprint(stderr, "maybe", maybe, (string(*)(void*))entity_local_name);
      set_fprint(stderr, "sure", sure, (string(*)(void*))entity_local_name);
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
      pips_debug(5, "extracting %" _intFMT "...\n", vertex_number(vok));
      set_add_element(current, current, vok);
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

      // close current and build a deterministic dag...
      list lcurr =
	set_to_sorted_list(current,
			   (int(*)(const void*, const void*)) dagvtx_compare);
      dag nd = make_dag(NIL, NIL, NIL);
      FOREACH(DAGVTX, v, lcurr)
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

      ld = CONS(DAG, nd, ld);

      set_clear(current);
      // recompute image sets
      set_clear(sure);
      set_clear(maybe);
      // dall
      set_append_list(maybe, dag_inputs(dall));
      // and already extracted dags
      FOREACH(DAG, d, ld)
      {
	FOREACH(DAGVTX, v, dag_vertices(d))
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
  set_free(current);
  set_free(removed);
  set_free(sure);
  set_free(maybe);
  set_free(avails);

  pips_debug(5, "returning %d dags\n", (int) gen_length(ld));
  return ld;
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

static void
freia_spoc_compile_calls
(string module,
 list /* of statements */ ls,
 int * n_helper)
{
  pips_debug(3, "considering %d statements\n", (int) gen_length(ls));
  pips_assert("some statements", ls);

  // build full dag
  dag fulld = make_dag(NIL, NIL, NIL);
  FOREACH(STATEMENT, s, ls)
    dag_append_freia_call(fulld, s);
  dag_set_inputs_outputs(fulld);
  ifdebug(3) dag_dump(stderr, "fulld", fulld);

  // TODO: split dag in one-pipe dags.
  list ld = split_dag(fulld);
  pips_assert("fulld dag is empty", gen_length(dag_vertices(fulld))==0);

  // output file
  string file = helper_file_name(module);
  pips_debug(1, "generating file '%s'\n", file);
  FILE * helper = safe_fopen(file, "w");
  fprintf(helper, FREIA_SPOC_INCLUDES);

  FOREACH(DAG, d, ld)
  {
    ifdebug(4) dag_dump(stderr, "d", d);

    string fname = helper_function_name(module, *n_helper);
    (*n_helper)++;
    list /* of expression */ lparams = NIL;
    string_buffer code = freia_spoc_pipeline(module, fname, d, &lparams);

    string_buffer_to_file(code, helper);

    pips_debug(4, "expecting %d statements...\n",
	       (int) gen_length(dag_vertices(d)));

    // replace sequence by call to accelerated function...
    bool first = true;
    FOREACH(DAGVTX, dv, dag_vertices(d))
    {
      statement sc =
	pstatement_statement(vtxcontent_source(dagvtx_content(dv)));
      pips_debug(5, "in statement %" _intFMT "\n", statement_number(sc));

      instruction old = statement_instruction(sc);
      if (first)
      {
	// as they are in reverse order in the vertex list,
	// this is really the last statement.
	pips_debug(5, "sustituting...\n");
	first = false;
	// substitute by call to helper
	entity helper = make_empty_subroutine(fname); // ??? function
	call c = make_call(helper, lparams);
	if (instruction_call_p(old) &&
	  ENTITY_C_RETURN_P(call_function(instruction_call(old))))
	{
	  call ret = instruction_call(old);
	  gen_full_free_list(call_arguments(ret));
	  call_arguments(ret) = CONS(EXPRESSION, call_to_expression(c), NIL);
	}
	else
	{
	  statement_instruction(sc) = call_to_instruction(c);
	  free_instruction(old);
	}
      }
      else
      {
	statement_instruction(sc) = make_continue_instruction();
	free_instruction(old);
      }
    }
    free(fname);
  }

  safe_fclose(helper, file);

  // cleanup
  free(file);
  free_dag(fulld);
  FOREACH(DAG, dc, ld) free_dag(dc);
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
  FOREACH(STATEMENT, s, sequence_statements(sq))
  {
    if (statement_freia_call_p(s))
      // ??? and possibly some other conditions...
      ls = CONS(STATEMENT, s, ls);
    else {
      // something else, interruption of the call sequence
      if (ls!=NIL) {
	ls = gen_nreverse(ls);
	fsip->seqs = CONS(LIST, ls, fsip->seqs);
	ls = NIL;
      }
    }
  }

  // end of sequence reached
  if (ls!=NIL) {
    ls = gen_nreverse(ls);
    fsip->seqs = CONS(LIST, ls, fsip->seqs);
    ls = NIL;
  }

  return true;
}

void freia_spoc_compile(string module, statement mod_stat)
{
  freia_spoc_info fsi;
  fsi.seqs = NIL;

  // collect freia api functions...
  // is it necessarily in a sequence?
  gen_context_multi_recurse(mod_stat, &fsi,
			    sequence_domain, sequence_flt, gen_null,
			    NULL);

  int n_helpers = 0;
  FOREACH(LIST, ls, fsi.seqs)
  {
    freia_spoc_compile_calls(module, ls, &n_helpers);
    gen_free_list(ls);
  }
  gen_free_list(fsi.seqs);
}
