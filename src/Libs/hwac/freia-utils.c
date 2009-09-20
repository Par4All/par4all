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

// help reduce code size:
#define T true
#define F false
#define cat concatenate

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
static const freia_api_t FREIA_AIPO_API[] = {
  { "undefined", "?", NULL, 0, 0, 0, 0, NO_PARAM, NO_PARAM,
    { 0, NO_POC, alu_unused, NO_MES }
  },
  {
    // ARITHMETIC
    // binary
    AIPO "add", "+", AIPO "add", 1, 2, 0, 0, NO_PARAM, NO_PARAM,
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
  { AIPO "sub", "-", NULL, 1, 2, 0, 0, NO_PARAM, NO_PARAM,
    { spoc_input_0|spoc_input_1|spoc_output_0|spoc_alu,
      NO_POC, alu_sub_01, NO_MES }
  },
  { AIPO "mul", "*",  AIPO "mul", 1, 2, 0, 0, NO_PARAM, NO_PARAM,
    { spoc_input_0|spoc_input_1|spoc_output_0|spoc_alu,
      NO_POC, alu_mul, NO_MES }
  },
  { AIPO "div", "/", NULL, 1, 2, 0, 0, NO_PARAM, NO_PARAM,
    { spoc_input_0|spoc_input_1|spoc_output_0|spoc_alu,
      NO_POC, alu_div_01, NO_MES }
  },
  { AIPO "addsat", "+s", AIPO "addsat", 1, 2, 0, 0, NO_PARAM, NO_PARAM,
    { spoc_input_0|spoc_input_1|spoc_output_0|spoc_alu,
      NO_POC, alu_addsat, NO_MES }
  },
  { AIPO "subsat", "-s", NULL, 1, 2, 0, 0, NO_PARAM, NO_PARAM,
    { spoc_input_0|spoc_input_1|spoc_output_0|spoc_alu,
      NO_POC, alu_subsat_01, NO_MES }
  },
  { AIPO "absdiff", "-|", AIPO "absdiff", 1, 2, 0, 0, NO_PARAM, NO_PARAM,
    { spoc_input_0|spoc_input_1|spoc_output_0|spoc_alu,
      NO_POC, alu_abssub_01, NO_MES }
  },
  { AIPO "inf", "<", AIPO "inf", 1, 2, 0, 0, NO_PARAM, NO_PARAM,
    { spoc_input_0|spoc_input_1|spoc_output_0|spoc_alu,
      NO_POC, alu_inf_01, NO_MES }
  },
  { AIPO "sup", ">", AIPO "sup", 1, 2, 0, 0, NO_PARAM, NO_PARAM,
    { spoc_input_0|spoc_input_1|spoc_output_0|spoc_alu,
      NO_POC, alu_sup_01, NO_MES }
  },
  { AIPO "and", "&", AIPO "and", 1, 2, 0, 0, NO_PARAM, NO_PARAM,
    { spoc_input_0|spoc_input_1|spoc_output_0|spoc_alu,
      NO_POC, alu_and, NO_MES }
  },
  { AIPO "or", "|", AIPO "or", 1, 2, 0, 0, NO_PARAM, NO_PARAM,
    { spoc_input_0|spoc_input_1|spoc_output_0|spoc_alu,
      NO_POC, alu_or, NO_MES }
  },
  { AIPO "xor", "^", AIPO "xor", 1, 2, 0, 0, NO_PARAM, NO_PARAM,
    { spoc_input_0|spoc_input_1|spoc_output_0|spoc_alu,
      NO_POC, alu_xor, NO_MES }
  },
  // unary
  { AIPO "not", "!", NULL, 1, 1, 0, 0, NO_PARAM, NO_PARAM,
    { spoc_input_0|spoc_output_0|spoc_alu, NO_POC, alu_not_0, NO_MES }
  },
  { AIPO "add_const", "+.", NULL, 1, 1, 0, 1, NO_PARAM, { TY_INT, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_alu, NO_POC, alu_add_0cst, NO_MES }
  },
  { AIPO "inf_const", "<.", NULL, 1, 1, 0, 1, NO_PARAM, { TY_INT, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_alu, NO_POC, alu_inf_0cst, NO_MES }
  },
  { AIPO "sup_const", ">.", NULL, 1, 1, 0, 1, NO_PARAM, { TY_INT, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_alu, NO_POC, alu_sup_0cst, NO_MES }
  },
  { AIPO "sub_const", "-.", NULL, 1, 1, 0, 1, NO_PARAM, { TY_INT, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_alu, NO_POC, alu_sub_0cst, NO_MES }
  },
  { AIPO "and_const", "&.", NULL, 1, 1, 0, 1, NO_PARAM, { TY_INT, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_alu, NO_POC, alu_and_0cst, NO_MES }
  },
  { AIPO "or_const", "|.", NULL, 1, 1, 0, 1, NO_PARAM, { TY_INT, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_alu, NO_POC, alu_or_0cst, NO_MES }
  },
  { AIPO "xor_const", "^.", NULL, 1, 1, 0, 1, NO_PARAM, { TY_INT, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_alu, NO_POC, alu_xor_0cst, NO_MES }
  },
  { AIPO "addsat_const", "+s.", NULL, 1, 1, 0, 1, NO_PARAM,
    { TY_INT, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_alu, NO_POC, alu_addsat_0cst, NO_MES }
  },
  { AIPO "subsat_const", "-s.", NULL, 1, 1, 0, 1, NO_PARAM,
    { TY_INT, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_alu, NO_POC, alu_subsat_0cst, NO_MES }
  },
  { AIPO "absdiff_const", "-|.", NULL, 1, 1, 0, 1, NO_PARAM,
    { TY_INT, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_alu, NO_POC, alu_abssub_0cst, NO_MES }
  },
  { AIPO "mul_const", "*.", NULL, 1, 1, 0, 1, NO_PARAM, { TY_INT, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_alu, NO_POC, alu_mul_0cst, NO_MES }
  },
  { AIPO "div_const", "/.", NULL, 1, 1, 0, 1, NO_PARAM, { TY_INT, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_alu, NO_POC, alu_div_0cst, NO_MES }
  },
  // nullary
  { AIPO "set_constant", "C", NULL, 1, 0, 0, 1, NO_PARAM, { TY_INT, NULL, NULL},
    { spoc_output_0|spoc_alu, NO_POC, alu_copy_cst, NO_MES }
  },
  // MISC
  // this one may be ignored?!
  { AIPO "copy", "=", NULL, 1, 1, 0, 0, NO_PARAM, NO_PARAM,
    { spoc_input_0|spoc_output_0, NO_POC, alu_unused, NO_MES }
  },
  { // not implemented by SPOC!
    AIPO "cast", "=()", NULL, 1, 1, 0, 0, NO_PARAM, NO_PARAM,
    { spoc_nothing, NO_POC, alu_unused, NO_MES }
  },
  { AIPO "threshold", "thr", NULL, 1, 1, 0, 3, NO_PARAM,
    { TY_INT, TY_INT, TY_INT },
    { spoc_input_0|spoc_output_0|spoc_th_0, NO_POC, alu_unused, NO_MES }
  },
  // MORPHO
  { AIPO "erode_6c", "E6", NULL, 1, 1, 0, 1, NO_PARAM, { TY_PIN, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_poc_0,
      { { spoc_poc_erode, 6 }, { spoc_poc_unused, 0 } }, alu_unused, NO_MES
    }
  },
  { AIPO "dilate_6c", "D6", NULL, 1, 1, 0, 1, NO_PARAM, { TY_PIN, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_poc_0,
      { { spoc_poc_dilate, 6 }, { spoc_poc_unused, 0 } }, alu_unused, NO_MES
    }
  },
  { AIPO "erode_8c", "E8", NULL, 1, 1, 0, 1, NO_PARAM, { TY_PIN, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_poc_0,
      { { spoc_poc_erode, 8 }, { spoc_poc_unused, 0 } }, alu_unused, NO_MES
    }
  },
  { AIPO "dilate_8c", "D8", NULL, 1, 1, 0, 1,  NO_PARAM, { TY_PIN, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_poc_0,
      { { spoc_poc_dilate, 8 }, { spoc_poc_unused, 0 } }, alu_unused, NO_MES
    }
  },
  // MEASURES
  { AIPO "global_min", "min", NULL, 0, 1, 1, 0,
    { TY_PIN, NULL, NULL }, NO_PARAM,
    { spoc_input_0 | spoc_measure_0,
      NO_POC, alu_unused, { measure_min, measure_none }
    }
  },
  { AIPO "global_max", "max", NULL, 0, 1, 1, 0, { TY_PIN, NULL, NULL },
    NO_PARAM, { spoc_input_0 | spoc_measure_0,
      NO_POC, alu_unused, { measure_max, measure_none }
    }
  },
  { AIPO "global_min_coord", "min!", NULL, 0, 1, 3, 0,
    { TY_PIN, TY_PIN, TY_PIN }, NO_PARAM,
    { spoc_input_0 | spoc_measure_0,
      NO_POC, alu_unused, { measure_min_coord, measure_none }
    }
  },
  { AIPO "global_max_coord", "max!", NULL, 0, 1, 3, 0,
    { TY_PIN, TY_PIN, TY_PIN }, NO_PARAM,
    { spoc_input_0 | spoc_measure_0,
      NO_POC, alu_unused, { measure_max_coord, measure_none }
    }
  },
  { AIPO "global_vol", "vol", NULL, 0, 1, 1, 0,
    { TY_PIN, NULL, NULL }, NO_PARAM,
    { spoc_input_0 | spoc_measure_0,
      NO_POC, alu_unused, { measure_vol, measure_none }
    }
  },
  // LINEAR
  // not implemented by SPOC!
  { AIPO "convolution", "conv", NULL, 1, 1, 0, 3,
    NO_PARAM, { TY_PIN, TY_UIN, TY_UIN },
    { spoc_nothing, NO_POC, alu_unused, NO_MES }
  },
  // not implemented by SPOC!
  { AIPO "fast_correlation", "corr", NULL, 1, 2, 0, 1,
    NO_PARAM, { TY_UIN, NULL, NULL },
    { spoc_nothing, NO_POC, alu_unused, NO_MES }
  },
  // last entry
  { NULL, NULL, NULL, 0, 0, 0, 0, NO_PARAM, NO_PARAM,
    { 0, NO_POC, alu_unused, NO_MES } }
};

#define FREIA_AIPO_API_SIZE (sizeof(*FREIA_AIPO_API)/sizeof(freia_api_t))

/* returns the index of the description of an AIPO function
 */
int hwac_freia_api_index(string function)
{
  const freia_api_t * api;
  for (api = FREIA_AIPO_API; api->function_name; api++)
    if (same_string_p(function, api->function_name))
      return api - FREIA_AIPO_API;
  return -1;
}

/* @returns the description of a FREIA AIPO API function.
 * may be moved elswhere. raise an error if not found.
 */
const freia_api_t * hwac_freia_api(string function)
{
  int index = hwac_freia_api_index(function);
  return index>=0? &FREIA_AIPO_API[index]: NULL;
}

const freia_api_t * get_freia_api(int index)
{
  // pips_assert("index exists", index>=0 && index<(int) FREIA_AIPO_API_SIZE);
  return &FREIA_AIPO_API[index];
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
  // SUBSAT
  { alu_subsat_01, alu_subsat_10, "SPOC_ALU_SUBSAT_IN0_IN1", F, T, T },
  { alu_subsat_10, alu_subsat_01, "SPOC_ALU_SUBSAT_IN1_IN0", F, T, T },
  { alu_subsat_0cst, alu_subsat_1cst, "SPOC_ALU_SUBSAT_IN0_CONST", T, T, F },
  { alu_subsat_1cst, alu_subsat_0cst, "SPOC_ALU_SUBSAT_IN1_CONST", T, F, T },
  // ABSSUB
  { alu_abssub_01, alu_abssub_10, "SPOC_ALU_ABSSUB_IN0_IN1", F, T, T },
  { alu_abssub_10, alu_abssub_01, "SPOC_ALU_ABSSUB_IN1_IN0", F, T, T },
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
  { alu_not_1, alu_not_0, "SPOC_ALU_NOT_IN1", T, F, T },
  // MISC
  { alu_copy_cst, alu_copy_cst, "SPOC_ALU_COPY_CONST", T, F, F }
  // and so on?
};

const spoc_alu_op_t *get_spoc_alu_conf(spoc_alu_t alu)
{
  return &ALU_OP[alu];
}

/* @return a string which describes the type of operation
 * (i.e. the hardware used by the function).
 */
string what_operation(_int type)
{
  switch (type)
  {
  case spoc_type_oth: return "other";
  case spoc_type_nop: return "none";
  case spoc_type_inp: return "input";
  case spoc_type_poc: return "poc";
  case spoc_type_alu: return "alu";
  case spoc_type_mes: return "measure";
  case spoc_type_thr: return "threshold";
  case spoc_type_out: return "output";
  default: return "unexpected value...";
  }
}

/* SPoC: set shape depending on hardware component used by vertex
 */
string what_operation_shape(_int type)
{
  string shape;
  switch (type)
  {
  case spoc_type_oth:
    shape = "shape=none"; break;
  case spoc_type_poc:
    shape = "shape=box"; break;
  case spoc_type_alu:
    shape = "shape=trapezium,orientation=270"; break;
  case spoc_type_thr:
    shape = "shape=parallelogram"; break;
  case spoc_type_mes:
    shape = "shape=diamond"; break;
    // these should not happen
  case spoc_type_out:
  case spoc_type_inp:
  case spoc_type_nop:
  default:
    shape = "shape=circle";
  }
  return shape;
}

/* ??? beurk: I keep the operation as two ints for code regeneration.
 */
void set_operation(const freia_api_t * api, _int * type, _int * id)
{
  pips_assert("no type set", *type == spoc_type_nop);

  // set type which is enough for staging?
  if (api->spoc.used & spoc_alu)
    *type = spoc_type_alu;
  else if (api->spoc.used & (spoc_poc_0|spoc_poc_1))
    *type = spoc_type_poc;
  else if (api->spoc.used & (spoc_th_0|spoc_th_1))
    *type = spoc_type_thr;
  else if (api->spoc.used & (spoc_measure_0|spoc_measure_1))
    *type = spoc_type_mes;
  else
    *type = spoc_type_nop; // e.g. for copy? (cast? convol? ...)

  // set details
  *id = hwac_freia_api_index(api->function_name);
}

/* returns an allocated expression list of the parameters only
 * (i.e. do not include the input & output images).
 */
list /* of expression*/ freia_extract_parameters(int napi, list args)
{
  const freia_api_t * api = get_freia_api(napi);
  int skip = api->arg_img_in + api->arg_img_out;
  while (skip--) args = CDR(args);
  return gen_full_copy_list(args);
}

/* all is well
 */
static call freia_ok()
{
  // how to build the "FREIA_OK" constant?
  return make_call(local_name_to_top_level_entity("0"), NIL);
}

/* is it an assignment to ignore
 */
bool freia_assignment_p(entity e)
{
  return ENTITY_ASSIGN_P(e) || ENTITY_BITWISE_OR_UPDATE_P(e);
}

/* replace statement contents with call to c, or continue if kill
 */
void hwac_replace_statement(statement s, call newc, bool kill)
{
  instruction old = statement_instruction(s);
  pips_assert("must kill a call", instruction_call_p(old));
  call c = instruction_call(old);
  entity called = call_function(c);
  if (ENTITY_C_RETURN_P(called))
  {
    // replace argument list
    gen_full_free_list(call_arguments(c));
    call_arguments(c) = CONS(expression, call_to_expression(newc), NIL);
  }
  else if (freia_assignment_p(called))
  {
    // new second argument
    list largs = call_arguments(c);
    pips_assert("two arguments to assignment", gen_length(largs)==2);
    expression first = EXPRESSION(CAR(largs));
    free_expression(EXPRESSION(CAR(CDR(largs))));
    gen_free_list(largs), largs = NIL;
    call_arguments(c) =
      CONS(expression, first, CONS(expression, call_to_expression(newc), NIL));
  }
  else
  {
    free_instruction(old);
    if (kill)
    {
      // ??? in C, it is not a do nothing instruction!
      statement_instruction(s) = make_continue_instruction();
      free_call(newc);
    }
    else
    {
      statement_instruction(s) = make_instruction_call(newc);
    }
  }
}

/* remove contents of statement s.
 */
void hwac_kill_statement(statement s)
{
  hwac_replace_statement(s, freia_ok(), true);
}

/* rather approximative?
 */
bool freia_image_variable_p(entity var)
{
  bool is_image = false;
  if (entity_variable_p(var) && entity_scalar_p(var))
  {
    type t = ultimate_type(entity_type(var));
    basic b = variable_basic(type_variable(t));
    if (basic_pointer_p(b))
    {
      t = basic_pointer(b);
      b = variable_basic(type_variable(t));
      is_image = basic_typedef_p(b) &&
	same_string_p(entity_local_name(basic_typedef(b)),
		      "$" FREIA_IMAGE_TYPE);
    }
  }

  pips_debug(8, "%s is%s an image\n", entity_name(var), is_image? "": " not");
  return is_image;
}

/* for "ret = freia_aipo_*()": return the 'ret' variable...
 * return NULL if not an assignment
 */
static entity get_assigned_variable(statement s)
{
  entity assigned = NULL;
  pips_assert("statement is a call", statement_call_p(s));
  call c = instruction_call(statement_instruction(s));
  entity called = call_function(c);
  if (freia_assignment_p(called))
    assigned = expression_to_entity(EXPRESSION(CAR(call_arguments(c))));
  return assigned;
}

/* returns whether the entity is a freia API (AIPO) function.
 */
bool entity_freia_api_p(entity f)
{
  // very partial...
  return strncmp(entity_local_name(f), AIPO, strlen(AIPO))==0;
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
bool freia_statement_aipo_call_p(statement s)
{
  // very partial as well
  instruction i = statement_instruction(s);
  if (instruction_call_p(i)) {
    call c = instruction_call(i);
    entity called = call_function(c);
    if (entity_freia_api_p(called) &&
	// ??? should be take care later?
	freia_spoc_optimise(called))
      return true;
    else if (freia_assignment_p(called))
    {
      list la = call_arguments(c);
      pips_assert("2 arguments to assign", gen_length(la));
      syntax op2 = expression_syntax(EXPRESSION(CAR(CDR(la))));
      if (syntax_call_p(op2))
	return entity_freia_api_p(call_function(syntax_call(op2)))
	  // ??? later?
	  && freia_spoc_optimise(call_function(syntax_call(op2)));
    }
    else if (ENTITY_C_RETURN_P(called))
    {
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

#include "effects-generic.h"

/* append simple scalar entities with written/read effects to s
 * scalars assigned to are ignored (return status).
 * no attempt at managing aliasing or the like...
 */
static void set_add_scalars(set s, statement stat, bool written)
{
  effects efs = load_cumulated_rw_effects(stat);
  entity skip = NULL;

  // skip assigned variable of AIPO calls
  if (freia_statement_aipo_call_p(stat))
    skip = get_assigned_variable(stat);

  FOREACH(effect, e, effects_effects(efs))
  {
    if ((written && effect_write_p(e)) || (!written && effect_read_p(e)))
    {
      entity var = reference_variable(effect_any_reference(e));
      if (entity_variable_p(var) && entity_scalar_p(var) &&
	  !freia_image_variable_p(var) && var!=skip)
	set_add_element(s, s, var);
    }
  }
}

/* is there a simple scalar (no image) rw dependency from s to t?
 * WW deps are ignored... should be okay of computed in order?
 * @param s source statement
 * @param t target statement
 * @param vars if set, return list of scalars which hold the dependencies
 */
bool freia_scalar_rw_dep(statement s, statement t, list * vars)
{
  // pips_assert("distinct statements", s!=t);
  if (s==t || !s || !t) return false;
  // I should really use entity_conflict_p...
  set reads = set_make(set_pointer), writes = set_make(set_pointer);
  set_add_scalars(writes, s, true);
  set_add_scalars(reads, t, false);
  set inter = set_make(set_pointer);
  set_intersection(inter, reads, writes);
  bool rw_dep = !set_empty_p(inter);
  if (vars)
    *vars = set_to_sorted_list(inter, (gen_cmp_func_t) compare_entities);
  set_free(reads);
  set_free(writes);
  set_free(inter);
  pips_debug(8, "%" _intFMT " %sdependent from %" _intFMT "\n",
	     statement_number(t), rw_dep? "": "in", statement_number(s));
  return rw_dep;
}

static bool lexpression_equal_p(const list l1, const list l2)
{
  bool equal = true;
  int n1 = gen_length(l1), n2 = gen_length(l2);
  if (n1==n2) {
    list p1 = (list) l1, p2 = (list) l2;
    while (equal && p1 && p2) {
      if (!expression_equal_p(EXPRESSION(CAR(p1)), EXPRESSION(CAR(p2))))
	equal = false;
      p1 = CDR(p1), p2 = CDR(p2);
    }
  }
  else
    equal = false;
  return equal;
}

/* return the actual function call from a statement,
 * dealing with assign and returns... return NULL if not a call.
 */
call freia_statement_to_call(statement s)
{
  // sanity check, somehow redundant
  instruction i = statement_instruction(s);
  call c = instruction_call_p(i)? instruction_call(i): NULL;
  if (c && freia_assignment_p(call_function(c)))
  {
    list args = call_arguments(c);
    pips_assert("2 args", gen_length(args) == 2);
    syntax sy = expression_syntax(EXPRESSION(CAR(CDR(args))));
    c = syntax_call_p(sy)? syntax_call(sy): NULL;
  }
  else if (c && ENTITY_C_RETURN_P(call_function(c)))
  {
    list args = call_arguments(c);
    pips_assert("one arg", gen_length(args)==1);
    syntax sy = expression_syntax(EXPRESSION(CAR(args)));
    c = syntax_call_p(sy)? syntax_call(sy): NULL;
  }
  return c;
}

#include "freia_spoc_private.h"
#include "hwac.h"

bool same_constant_parameters(dagvtx v1, dagvtx v2)
{
  call
    c1 = freia_statement_to_call(dagvtx_statement(v1)),
    c2 = freia_statement_to_call(dagvtx_statement(v2));
  list
    lp1 = freia_extract_parameters(dagvtx_opid(v1), call_arguments(c1)),
    lp2 = freia_extract_parameters(dagvtx_opid(v1), call_arguments(c2));
  bool same = lexpression_equal_p(lp1, lp2);
  // should also check that there is no w effects on parameters in between
  return same;
}
