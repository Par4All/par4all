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

#define NO_POC { { spoc_poc_unused, 0 }, { spoc_poc_unused, 0 } }
#define NO_MES { measure_none, measure_none }

#define NO_SPOC { spoc_nothing, NO_POC, alu_unused, NO_MES }
#define NO_TERAPIX { 0, 0, 0, 0, 0, 0, false, NULL }

#define TRPX_OP(c, op) { 0, 0, 0, 0, 0, c, true, "TERAPIX_UCODE_" op }
#define TRPX_NG(c, op) { 1, 1, 1, 1, 0, c, false, "TERAPIX_UCODE_" op }

// preliminary stuff for volume/min/max/..., although it is not defined yet
#define TRPX_MS(m, c, op) { 0, 0, 0, 0, m, c, true, "TERAPIX_UCODE_" op }

// types used by AIPO parameters
#define TY_INT "int32_t"
#define TY_PIN "int32_t *"
#define TY_CIP "const int32_t *"
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
    { 0, NO_POC, alu_unused, NO_MES }, NO_TERAPIX
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
    }, TRPX_OP(4, "ADD")
  },
  { AIPO "sub", "-", NULL, 1, 2, 0, 0, NO_PARAM, NO_PARAM,
    { spoc_input_0|spoc_input_1|spoc_output_0|spoc_alu,
      NO_POC, alu_sub_01, NO_MES }, TRPX_OP(4, "SUB")
  },
  { AIPO "mul", "*",  AIPO "mul", 1, 2, 0, 0, NO_PARAM, NO_PARAM,
    { spoc_input_0|spoc_input_1|spoc_output_0|spoc_alu,
      NO_POC, alu_mul, NO_MES }, TRPX_OP(4, "MUL")
  },
  { AIPO "div", "/", NULL, 1, 2, 0, 0, NO_PARAM, NO_PARAM,
    { spoc_input_0|spoc_input_1|spoc_output_0|spoc_alu,
      NO_POC, alu_div_01, NO_MES }, TRPX_OP(4, "DIV")
  },
  { AIPO "addsat", "+s", AIPO "addsat", 1, 2, 0, 0, NO_PARAM, NO_PARAM,
    { spoc_input_0|spoc_input_1|spoc_output_0|spoc_alu,
      NO_POC, alu_addsat, NO_MES }, TRPX_OP(4, "ADDSAT?")
  },
  { AIPO "subsat", "-s", NULL, 1, 2, 0, 0, NO_PARAM, NO_PARAM,
    { spoc_input_0|spoc_input_1|spoc_output_0|spoc_alu,
      NO_POC, alu_subsat_01, NO_MES }, TRPX_OP(4, "SUBSAT?")
  },
  { AIPO "absdiff", "-|", AIPO "absdiff", 1, 2, 0, 0, NO_PARAM, NO_PARAM,
    { spoc_input_0|spoc_input_1|spoc_output_0|spoc_alu,
      NO_POC, alu_abssub_01, NO_MES }, TRPX_OP(4, "ABS_DIFF")
  },
  { AIPO "inf", "<", AIPO "inf", 1, 2, 0, 0, NO_PARAM, NO_PARAM,
    { spoc_input_0|spoc_input_1|spoc_output_0|spoc_alu,
      NO_POC, alu_inf_01, NO_MES }, TRPX_OP(4, "INF")
  },
  { AIPO "sup", ">", AIPO "sup", 1, 2, 0, 0, NO_PARAM, NO_PARAM,
    { spoc_input_0|spoc_input_1|spoc_output_0|spoc_alu,
      NO_POC, alu_sup_01, NO_MES }, TRPX_OP(4, "SUP")
  },
  { AIPO "and", "&", AIPO "and", 1, 2, 0, 0, NO_PARAM, NO_PARAM,
    { spoc_input_0|spoc_input_1|spoc_output_0|spoc_alu,
      NO_POC, alu_and, NO_MES }, TRPX_OP(4, "AND")
  },
  { AIPO "or", "|", AIPO "or", 1, 2, 0, 0, NO_PARAM, NO_PARAM,
    { spoc_input_0|spoc_input_1|spoc_output_0|spoc_alu,
      NO_POC, alu_or, NO_MES }, TRPX_OP(4, "OR")
  },
  { AIPO "xor", "^", AIPO "xor", 1, 2, 0, 0, NO_PARAM, NO_PARAM,
    { spoc_input_0|spoc_input_1|spoc_output_0|spoc_alu,
      NO_POC, alu_xor, NO_MES }, TRPX_OP(4, "XOR")
  },
  // unary
  { AIPO "not", "!", NULL, 1, 1, 0, 0, NO_PARAM, NO_PARAM,
    { spoc_input_0|spoc_output_0|spoc_alu, NO_POC, alu_not_0, NO_MES },
    TRPX_OP(4, "NOT") // ??? why not less
  },
  { AIPO "add_const", "+.", NULL, 1, 1, 0, 1, NO_PARAM, { TY_INT, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_alu, NO_POC, alu_add_0cst, NO_MES },
    TRPX_OP(3, "ADD_CONST")
  },
  { AIPO "inf_const", "<.", NULL, 1, 1, 0, 1, NO_PARAM, { TY_INT, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_alu, NO_POC, alu_inf_0cst, NO_MES },
    TRPX_OP(3, "INF_CONST?")
  },
  { AIPO "sup_const", ">.", NULL, 1, 1, 0, 1, NO_PARAM, { TY_INT, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_alu, NO_POC, alu_sup_0cst, NO_MES },
    TRPX_OP(3, "SUP_CONST?")
  },
  { AIPO "sub_const", "-.", NULL, 1, 1, 0, 1, NO_PARAM, { TY_INT, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_alu, NO_POC, alu_sub_0cst, NO_MES },
    TRPX_OP(3, "SUB_CONST")
  },
  { AIPO "and_const", "&.", NULL, 1, 1, 0, 1, NO_PARAM, { TY_INT, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_alu, NO_POC, alu_and_0cst, NO_MES },
    TRPX_OP(3, "AND_CONST")
  },
  { AIPO "or_const", "|.", NULL, 1, 1, 0, 1, NO_PARAM, { TY_INT, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_alu, NO_POC, alu_or_0cst, NO_MES },
    TRPX_OP(3, "OR_CONST?")
  },
  { AIPO "xor_const", "^.", NULL, 1, 1, 0, 1, NO_PARAM, { TY_INT, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_alu, NO_POC, alu_xor_0cst, NO_MES },
    TRPX_OP(3, "XOR_CONST?")
  },
  { AIPO "addsat_const", "+s.", NULL, 1, 1, 0, 1, NO_PARAM,
    { TY_INT, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_alu, NO_POC, alu_addsat_0cst, NO_MES },
    TRPX_OP(3, "ADDSAT_CONST?")
  },
  { AIPO "subsat_const", "-s.", NULL, 1, 1, 0, 1, NO_PARAM,
    { TY_INT, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_alu, NO_POC, alu_subsat_0cst, NO_MES },
    TRPX_OP(3, "SUBSAT_CONST?")
  },
  { AIPO "absdiff_const", "-|.", NULL, 1, 1, 0, 1, NO_PARAM,
    { TY_INT, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_alu, NO_POC, alu_abssub_0cst, NO_MES },
    TRPX_OP(3, "ABSDIFF_CONST?")
  },
  { AIPO "mul_const", "*.", NULL, 1, 1, 0, 1, NO_PARAM, { TY_INT, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_alu, NO_POC, alu_mul_0cst, NO_MES },
    TRPX_OP(3, "MUL_CONST")
  },
  { AIPO "div_const", "/.", NULL, 1, 1, 0, 1, NO_PARAM, { TY_INT, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_alu, NO_POC, alu_div_0cst, NO_MES },
    TRPX_OP(3, "DIV_CONST")
  },
  // nullary
  { AIPO "set_constant", "C", NULL, 1, 0, 0, 1, NO_PARAM, { TY_INT, NULL, NULL},
    { spoc_output_0|spoc_alu, NO_POC, alu_copy_cst, NO_MES },
    TRPX_OP(2, "SET_CONST?")
  },
  // MISC
  // this one may be ignored?!
  { AIPO "copy", "=", NULL, 1, 1, 0, 0, NO_PARAM, NO_PARAM,
    { spoc_input_0|spoc_output_0, NO_POC, alu_unused, NO_MES },
    TRPX_OP(3, "COPY")
  },
  { // not implemented by SPOC!
    AIPO "cast", "=()", NULL, 1, 1, 0, 0, NO_PARAM, NO_PARAM,
    NO_SPOC, NO_TERAPIX
  },
  { AIPO "threshold", "thr", NULL, 1, 1, 0, 3, NO_PARAM,
    { TY_INT, TY_INT, TY_INT },
    { spoc_input_0|spoc_output_0|spoc_th_0, NO_POC, alu_unused, NO_MES },
    TRPX_OP(5, "THRESHOLD")
  },
  // MORPHO
  { AIPO "erode_6c", "E6", NULL, 1, 1, 0, 1, NO_PARAM, { TY_CIP, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_poc_0,
      { { spoc_poc_erode, 6 }, { spoc_poc_unused, 0 } }, alu_unused, NO_MES
    },
    TRPX_NG(10, "ERODE_3_3?")
  },
  { AIPO "dilate_6c", "D6", NULL, 1, 1, 0, 1, NO_PARAM, { TY_CIP, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_poc_0,
      { { spoc_poc_dilate, 6 }, { spoc_poc_unused, 0 } }, alu_unused, NO_MES
    },
    TRPX_NG(10, "DILATE_3_3?")
  },
  { AIPO "erode_8c", "E8", NULL, 1, 1, 0, 1, NO_PARAM, { TY_CIP, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_poc_0,
      { { spoc_poc_erode, 8 }, { spoc_poc_unused, 0 } }, alu_unused, NO_MES
    },
    TRPX_NG(15, "ERODE_3_3")
  },
  { AIPO "dilate_8c", "D8", NULL, 1, 1, 0, 1,  NO_PARAM, { TY_CIP, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_poc_0,
      { { spoc_poc_dilate, 8 }, { spoc_poc_unused, 0 } }, alu_unused, NO_MES
    },
    TRPX_NG(15, "DILATE_3_3")
  },
  // MEASURES
  { AIPO "global_min", "min", NULL, 0, 1, 1, 0,
    { TY_PIN, NULL, NULL }, NO_PARAM,
    { spoc_input_0 | spoc_measure_0,
      NO_POC, alu_unused, { measure_min, measure_none }
    },
    TRPX_MS(1, 3, "MIN?")
  },
  { AIPO "global_max", "max", NULL, 0, 1, 1, 0, { TY_PIN, NULL, NULL },
    NO_PARAM, { spoc_input_0 | spoc_measure_0,
      NO_POC, alu_unused, { measure_max, measure_none }
    },
    TRPX_MS(1, 3, "MAX?")
  },
  { AIPO "global_min_coord", "min!", NULL, 0, 1, 3, 0,
    { TY_PIN, TY_PIN, TY_PIN }, NO_PARAM,
    { spoc_input_0 | spoc_measure_0,
      NO_POC, alu_unused, { measure_min_coord, measure_none }
    },
    TRPX_MS(1, 3, "MIN_COORD?")
  },
  { AIPO "global_max_coord", "max!", NULL, 0, 1, 3, 0,
    { TY_PIN, TY_PIN, TY_PIN }, NO_PARAM,
    { spoc_input_0 | spoc_measure_0,
      NO_POC, alu_unused, { measure_max_coord, measure_none }
    },
    TRPX_MS(1, 3, "MAX_COORD?")
  },
  { AIPO "global_vol", "vol", NULL, 0, 1, 1, 0,
    { TY_PIN, NULL, NULL }, NO_PARAM,
    { spoc_input_0 | spoc_measure_0,
      NO_POC, alu_unused, { measure_vol, measure_none }
    },
    TRPX_MS(2, 3, "VOL?")
  },
  // LINEAR
  // not implemented by SPOC!
  { AIPO "convolution", "conv", NULL, 1, 1, 0, 3,
    NO_PARAM, { TY_PIN, TY_UIN, TY_UIN },
    NO_SPOC, NO_TERAPIX
  },
  // not implemented by SPOC!
  { AIPO "fast_correlation", "corr", NULL, 1, 2, 0, 1,
    NO_PARAM, { TY_UIN, NULL, NULL },
    NO_SPOC, NO_TERAPIX
  },
  // last entry
  { NULL, NULL, NULL, 0, 0, 0, 0, NO_PARAM, NO_PARAM,
    { 0, NO_POC, alu_unused, NO_MES },
    NO_TERAPIX
  }
};

#define FREIA_AIPO_API_SIZE (sizeof(*FREIA_AIPO_API)/sizeof(freia_api_t))

/* returns the index of the description of an AIPO function
 */
int hwac_freia_api_index(const string function)
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
const freia_api_t * hwac_freia_api(const string function)
{
  int index = hwac_freia_api_index(function);
  return index>=0? &FREIA_AIPO_API[index]: NULL;
}

const freia_api_t * get_freia_api(int index)
{
  // pips_assert("index exists", index>=0 && index<(int) FREIA_AIPO_API_SIZE);
  return &FREIA_AIPO_API[index];
}

/* @return new allocated variable name using provided prefix.
 * *params is incremented as a side effect.
 */
static string get_var(string prefix, int * params)
{
  return strdup(cat(prefix, itoa((*params)++)));
}

/* @return a string which describes the type of operation
 * (i.e. the hardware used by the function for spoc).
 */
string what_operation(const _int type)
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
string what_operation_shape(const _int type)
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

list freia_get_params(const freia_api_t * api, list args)
{
  int skip = api->arg_img_in + api->arg_img_out;
  while (skip--) args = CDR(args);
  return args;
}

list freia_get_vertex_params(const dagvtx v)
{
  const vtxcontent vc = dagvtx_content(v);
  pips_assert("there is a statement",
              pstatement_statement_p(vtxcontent_source(vc)));
  const statement s = pstatement_statement(vtxcontent_source(vc));
  const call c = freia_statement_to_call(s);
  const freia_api_t * api = dagvtx_freia_api(v);
  return freia_get_params(api, call_arguments(c));
}

/* returns an allocated expression list of the parameters only
 * (i.e. do not include the input & output images).
 * params maps variables (if so) to already present parameters names.
 * the extraction function allocs new parameter names if necessary.
 */
list /* of expression */ freia_extract_params
  (const int napi,     // api function number
   list args,          // actual arguments to call
   string_buffer head, // function headers
   hash_table params,  // argument/variable to parameter mapping
   int * nparams)      // current number of parameters
{
  const freia_api_t * api = get_freia_api(napi);
  args = freia_get_params(api, args);
  list res = NIL;
  bool merge = get_bool_property("FREIA_MERGE_ARGUMENTS");

  pips_assert("number of arguments is okay",
              gen_length(args)==api->arg_misc_in+api->arg_misc_out);

  for (unsigned int i = 0; i<api->arg_misc_in; i++)
  {
    expression e = EXPRESSION(CAR(args));
    args = CDR(args);

    if (params)
    {
      // ??? if the expression is a constant,
      // the parameter could be skipped as well?
      entity var = expression_to_entity(e);
      if (merge && !entity_undefined_p(var) && entity_variable_p(var))
      {
        if (!hash_defined_p(params, var))
        {
          // choose new name
          string name = get_var("pi", nparams);
          if (head) sb_cat(head, ",\n  ", api->arg_in_types[i], " ", name);
          hash_put(params, e, name);
          res = CONS(expression, copy_expression(e), res);
          // keep record for the *variable* as well...
          hash_put(params, var, name);
        }
        else
        {
          // skip argument, just record its where it is found
          hash_put(params, e, hash_get(params, var));
        }
      }
      else
      {
        // append and record new parameter
        string name = get_var("pi", nparams);
        if (head) sb_cat(head, ",\n  ", api->arg_in_types[i], " ", name);
        hash_put(params, e, name);
        res = CONS(expression, copy_expression(e), res);
      }
    }
    else
      res = CONS(expression, copy_expression(e), res);
  }

  for (unsigned int i = 0; i<api->arg_misc_out; i++)
  {
    expression e = EXPRESSION(CAR(args));
    args = CDR(args);
    string name = get_var("po", nparams);
    if (head) sb_cat(head, ",\n  ", api->arg_out_types[i], " ", name);
    if (params)
      hash_put(params, e, name);
    else
      free(name);
    res = CONS(expression, copy_expression(e), res);
  }

  return gen_nreverse(res);
}

/* all is well
 */
static call freia_ok(void)
{
  // how to build the "FREIA_OK" constant?
  return make_call(local_name_to_top_level_entity("0"), NIL);
}

/* is it an assignment to ignore
 */
bool freia_assignment_p(const entity e)
{
  return ENTITY_ASSIGN_P(e) || ENTITY_BITWISE_OR_UPDATE_P(e);
}

/* @return "freia_aipo_copy(target, source);"
 */
statement freia_copy_image(const entity source, const entity target)
{
  return call_to_statement(
    make_call(local_name_to_top_level_entity(AIPO "copy"),
              CONS(expression, entity_to_expression(target),
                   CONS(expression, entity_to_expression(source), NIL))));
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
bool freia_image_variable_p(const entity var)
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
static entity get_assigned_variable(const statement s)
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
bool entity_freia_api_p(const entity f)
{
  // very partial...
  return strncmp(entity_local_name(f), AIPO, strlen(AIPO))==0;
}

/* @return whether to optimize AIPO call to function for SPoC.
*/
static bool freia_spoc_optimise(const entity called)
{
  string fname = entity_local_name(called);
  return !same_string_p(fname, "freia_aipo_convolution") &&
    !same_string_p(fname, "freia_aipo_cast") &&
    !same_string_p(fname, "freia_aipo_fast_correlation");
}

/* returns whether the statement is a FREIA call.
 */
bool freia_statement_aipo_call_p(const statement s)
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
static void set_add_scalars(set s, const statement stat, const bool written)
{
  effects efs = load_cumulated_rw_effects(stat);
  entity skip = NULL;

  // skip assigned variable of AIPO calls
  if (freia_statement_aipo_call_p(stat))
    skip = get_assigned_variable(stat);

  FOREACH(effect, e, effects_effects(efs))
  {
    if (!malloc_effect_p(e) &&
        ((written && effect_write_p(e)) || (!written && effect_read_p(e))))
    {
      entity var = reference_variable(effect_any_reference(e));
      if (entity_variable_p(var) && !freia_image_variable_p(var) &&
          var!=skip && entity_scalar_p(var))
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
bool freia_scalar_rw_dep(const statement s, const statement t, list * vars)
{
  // pips_assert("distinct statements", s!=t);
  if (s==t || !s || !t) return false;
  // I should really use entities_may_conflict_p...
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
call freia_statement_to_call(const statement s)
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

/* tell whether v1 and v2 point to statements with the same parameters.
 */
bool same_constant_parameters(const dagvtx v1, const dagvtx v2)
{
  call
    c1 = freia_statement_to_call(dagvtx_statement(v1)),
    c2 = freia_statement_to_call(dagvtx_statement(v2));
  list
    lp1 = freia_extract_params
    (dagvtx_opid(v1), call_arguments(c1), NULL, NULL, NULL),
    lp2 = freia_extract_params
    (dagvtx_opid(v1), call_arguments(c2), NULL, NULL, NULL);
  bool same = lexpression_equal_p(lp1, lp2);
  gen_free_list(lp1), gen_free_list(lp2);
  // should also check that there is no w effects on parameters in between
  return same;
}

/* freia_statement_by_helper_call
 * substitute those statement in ls that are in dag d and accelerated
 * by a call to function_name(lparams)
 * also update sets of remainings and global_remainings
 */
void freia_substitute_by_helper_call
  (dag d,
   set global_remainings,
   set remainings,
   list /* of statement */ ls,
   string function_name,
   list lparams)
{
  // statements that are not yet computed in d...
  set not_dones = set_make(set_pointer), dones = set_make(set_pointer);
  FOREACH(dagvtx, vs, dag_vertices(d))
  {
    pstatement ps = vtxcontent_source(dagvtx_content(vs));
    if (pstatement_statement_p(ps))
      set_add_element(not_dones, not_dones, pstatement_statement(ps));
  }
  set_difference(dones, remainings, not_dones);
  set_difference(remainings, remainings, dones);
  set_difference(global_remainings, global_remainings, dones);

  // replace first statement of dones in ls (so last in sequence)
  bool substitution_done = false;
  FOREACH(statement, sc, ls)
  {
    pips_debug(5, "in statement %" _intFMT "\n", statement_number(sc));

    if (set_belong_p(dones, sc))
    {
      pips_assert("statement is a call", statement_call_p(sc));
      pips_debug(5, "sustituting %" _intFMT"...\n", statement_number(sc));

      // substitute by call to helper (subroutine or function?)
      entity helper = make_empty_subroutine(function_name,
                                            make_language_unknown());
      call c = make_call(helper, lparams);

      hwac_replace_statement(sc, c, false);
      substitution_done = true;
      break;
    }
  }
  pips_assert("substitution done", substitution_done);

  set_free(not_dones);
  set_free(dones);
}

/* insert added statements to actual code sequence in "ls"
 */
void freia_insert_added_stats(list ls, list added_stats)
{
  if (added_stats)
  {
    statement slast = STATEMENT(CAR(ls));
    // fprintf(stderr, "adding stats to %p\n", slast);
    instruction ilast = statement_instruction(slast);
    statement newstat = instruction_to_statement(ilast);
    // transfer comments and some cleanup...
    statement_comments(newstat) = statement_comments(slast);
    statement_comments(slast) = string_undefined;
    statement_number(slast) = STATEMENT_NUMBER_UNDEFINED;
    // pretty ugly because return must be handled especially...
    if (instruction_call_p(ilast) &&
        ENTITY_C_RETURN_P(call_function(instruction_call(ilast))))
    {
      call c = instruction_call(ilast);
      if (!expression_constant_p(EXPRESSION(CAR(call_arguments(c)))))
      {
        // must split return...
        pips_internal_error("return splitting not implemented yet...\n");
      }
      else
      {
        added_stats = gen_nconc(added_stats, CONS(statement, newstat, NIL));
      }
    }
    else
    {
      added_stats = CONS(statement, newstat, added_stats);
    }
    statement_instruction(slast) =
      make_instruction_sequence(make_sequence(added_stats));
  }
}

/* prepend limg images in front of the argument list
 * limg is consummed by the operation.
 */
void freia_add_image_arguments
  (list limg, // of entity
   list * lparams) // of expression
{
  list largs = NIL;
  limg = gen_nreverse(limg);
  FOREACH(entity, e, limg)
    largs = CONS(expression, entity_to_expression(e), largs);
  gen_free_list(limg), limg = NIL;
  *lparams = gen_nconc(largs, *lparams);
}

/********************************************************* IMAGE OCCURRENCES */

/* hack to help replace use-def chains which did not work initially with C.
 * occurrences is: <image entity> -> { set of statements }
 * this is really a ugly hack, sorry!
 * ??? I should also consider declarations, but as they should only
 * contain image allocations so there should be no problem.
 */

static void check_ref(reference r, hash_table occs)
{
  entity v = reference_variable(r);
  if (freia_image_variable_p(v))
  {
    // ensure that target set exists
    if (!hash_defined_p(occs, v))
      hash_put(occs, v, set_make(set_pointer));
    set stats = (set) hash_get(occs, v);
    // get first containing statement
    statement up = (statement) gen_get_ancestor(statement_domain, r);
    // which MUST exist?
    pips_assert("some containing statement", up);
    // store result
    set_add_element(stats, stats, (void*) up);
  }
}

/* @return build occurrence hash table
 */
hash_table freia_build_image_occurrences(statement s)
{
  hash_table occs = hash_table_make(hash_pointer, 0);
  gen_context_recurse(s, (void*) occs, reference_domain, gen_true, check_ref);
  return occs;
}

/* cleanup occurrence data structure
 */
void freia_clean_image_occurrences(hash_table occs)
{
  HASH_FOREACH(entity, v, set, s, occs)
    set_free(s);
  hash_table_free(occs);
}
