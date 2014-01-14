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

#define NO_POC { { spoc_poc_unused, 0 }, { spoc_poc_unused, 0 } }
#define NO_MES { measure_none, measure_none }

// no operation
#define NOPE_SPOC { spoc_nothing, NO_POC, alu_unused, NO_MES }
#define NOPE_TRPX { 0, 0, 0, 0, 0, 0, false, false, NULL }
#define NOPE_OPCL { F, F, NULL, NULL }

// not implemented
#define NO_SPOC { spoc_not_implemented, NO_POC, alu_unused, NO_MES }
#define NO_TRPX { 0, 0, 0, 0, 0, -1, false, false, NULL }
#define NO_OPCL { F, F, NULL, NULL }

#define TRPX_OP(c, op) { 0, 0, 0, 0, 0, c, true, false, "TERAPIX_UCODE_" op }
#define TRPX_IO(c, op) { 0, 0, 0, 0, 0, c, true, true, "TERAPIX_UCODE_" op }
#define TRPX_NG(c, op) { 1, 1, 1, 1, 0, c, false, false, "TERAPIX_UCODE_" op }

// preliminary stuff for volume/min/max/...
#define TRPX_MS(m, c, op) { 0, 0, 0, 0, m, c, true, false, "TERAPIX_UCODE_" op }

#define OPCL(op)       { T, F, "PIXEL_" op, NULL }
#define OPCLK(op,init) { F, T, "PIXEL_" op, "PIXEL_" init }

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
    NOPE_SPOC, NOPE_TRPX, NOPE_OPCL
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
    },
    TRPX_OP(4, "ADD3"),
    OPCL("ADD")
  },
  { AIPO "sub", "-", NULL, 1, 2, 0, 0, NO_PARAM, NO_PARAM,
    { spoc_input_0|spoc_input_1|spoc_output_0|spoc_alu,
      NO_POC, alu_sub_01, NO_MES }, TRPX_OP(4, "SUB3"), OPCL("SUB")
  },
  { AIPO "mul", "*",  AIPO "mul", 1, 2, 0, 0, NO_PARAM, NO_PARAM,
    { spoc_input_0|spoc_input_1|spoc_output_0|spoc_alu,
      NO_POC, alu_mul, NO_MES }, TRPX_OP(4, "MUL3"), OPCL("MUL")
  },
  { AIPO "div", "/", NULL, 1, 2, 0, 0, NO_PARAM, NO_PARAM,
    { spoc_input_0|spoc_input_1|spoc_output_0|spoc_alu,
      NO_POC, alu_div_01, NO_MES }, TRPX_OP(4, "DIV3"), OPCL("DIV")
  },
  { AIPO "addsat", "+s", AIPO "addsat", 1, 2, 0, 0, NO_PARAM, NO_PARAM,
    { spoc_input_0|spoc_input_1|spoc_output_0|spoc_alu,
      NO_POC, alu_addsat, NO_MES }, TRPX_OP(4, "ADDSAT"), OPCL("ADDSAT")
  },
  { AIPO "subsat", "-s", NULL, 1, 2, 0, 0, NO_PARAM, NO_PARAM,
    { spoc_input_0|spoc_input_1|spoc_output_0|spoc_alu,
      NO_POC, alu_subsat_01, NO_MES }, TRPX_OP(4, "SUBSAT"), OPCL("SUBSAT")
  },
  { AIPO "absdiff", "-|", AIPO "absdiff", 1, 2, 0, 0, NO_PARAM, NO_PARAM,
    { spoc_input_0|spoc_input_1|spoc_output_0|spoc_alu,
      NO_POC, alu_abssub, NO_MES }, TRPX_OP(4, "ABS_DIFF3"), OPCL("ABSDIFF")
  },
  { AIPO "inf", "<", AIPO "inf", 1, 2, 0, 0, NO_PARAM, NO_PARAM,
    { spoc_input_0|spoc_input_1|spoc_output_0|spoc_alu,
      NO_POC, alu_inf_01, NO_MES }, TRPX_OP(4, "INF3"), OPCL("INF")
  },
  { AIPO "sup", ">", AIPO "sup", 1, 2, 0, 0, NO_PARAM, NO_PARAM,
    { spoc_input_0|spoc_input_1|spoc_output_0|spoc_alu,
      NO_POC, alu_sup_01, NO_MES }, TRPX_OP(4, "SUP3"), OPCL("SUP")
  },
  { AIPO "and", "&", AIPO "and", 1, 2, 0, 0, NO_PARAM, NO_PARAM,
    { spoc_input_0|spoc_input_1|spoc_output_0|spoc_alu,
      NO_POC, alu_and, NO_MES }, TRPX_OP(4, "AND3"), OPCL("AND")
  },
  { AIPO "or", "|", AIPO "or", 1, 2, 0, 0, NO_PARAM, NO_PARAM,
    { spoc_input_0|spoc_input_1|spoc_output_0|spoc_alu,
      NO_POC, alu_or, NO_MES }, TRPX_OP(4, "OR3"), OPCL("OR")
  },
  { AIPO "xor", "^", AIPO "xor", 1, 2, 0, 0, NO_PARAM, NO_PARAM,
    { spoc_input_0|spoc_input_1|spoc_output_0|spoc_alu,
      NO_POC, alu_xor, NO_MES }, TRPX_OP(4, "XOR3"), OPCL("XOR")
  },
  { AIPO "replace_const", ":", AIPO "replace_const",
    1, 2, 0, 1, NO_PARAM, { TY_INT, NULL, NULL},
    { spoc_input_0|spoc_input_1|spoc_output_0|spoc_alu,
      NO_POC, alu_repcst_0, NO_MES }, TRPX_IO(3, "CONV_REPLACE_EQ_CONST"),
    OPCL("REPLACE_EC")
  },
  // unary
  { AIPO "not", "!", NULL, 1, 1, 0, 0, NO_PARAM, NO_PARAM,
    { spoc_input_0|spoc_output_0|spoc_alu, NO_POC, alu_not_0, NO_MES },
    // ??? why not less?
    TRPX_OP(4, "NOT"), OPCL("NOT")
  },
  { AIPO "log2", "l2", NULL, 1, 1, 0, 0, NO_PARAM, NO_PARAM,
    { spoc_input_0|spoc_output_0|spoc_alu, NO_POC, alu_log2_0, NO_MES },
    TRPX_OP(3, "LOG2"), OPCL("LOG2")
  },
  { AIPO "add_const", "+_", NULL, 1, 1, 0, 1, NO_PARAM, { TY_INT, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_alu, NO_POC, alu_add_0cst, NO_MES },
    TRPX_OP(3, "ADD_CONST"), OPCL("ADD")
  },
  { AIPO "inf_const", "<_", NULL, 1, 1, 0, 1, NO_PARAM, { TY_INT, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_alu, NO_POC, alu_inf_0cst, NO_MES },
    TRPX_OP(3, "INF_CONST"), OPCL("INF")
  },
  { AIPO "sup_const", ">_", NULL, 1, 1, 0, 1, NO_PARAM, { TY_INT, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_alu, NO_POC, alu_sup_0cst, NO_MES },
    TRPX_OP(3, "SUP_CONST"), OPCL("SUP")
  },
  { AIPO "sub_const", "-_", NULL, 1, 1, 0, 1, NO_PARAM, { TY_INT, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_alu, NO_POC, alu_sub_0cst, NO_MES },
    TRPX_OP(3, "SUB_CONST"), OPCL("SUB")
  },
  { AIPO "const_sub", "_-", NULL, 1, 1, 0, 1, NO_PARAM, { TY_INT, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_alu, NO_POC, alu_sub_cst0, NO_MES },
    TRPX_OP(3, "CONST_SUB"), OPCL("SUBC")
  },
  { AIPO "and_const", "&_", NULL, 1, 1, 0, 1, NO_PARAM, { TY_INT, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_alu, NO_POC, alu_and_0cst, NO_MES },
    TRPX_OP(3, "AND_CONST"), OPCL("AND")
  },
  { AIPO "or_const", "|_", NULL, 1, 1, 0, 1, NO_PARAM, { TY_INT, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_alu, NO_POC, alu_or_0cst, NO_MES },
    TRPX_OP(3, "OR_CONST"), OPCL("OR")
  },
  { AIPO "xor_const", "^_", NULL, 1, 1, 0, 1, NO_PARAM, { TY_INT, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_alu, NO_POC, alu_xor_0cst, NO_MES },
    TRPX_OP(3, "XOR_CONST"), OPCL("XOR")
  },
  { AIPO "addsat_const", "+s_", NULL, 1, 1, 0, 1, NO_PARAM,
    { TY_INT, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_alu, NO_POC, alu_addsat_0cst, NO_MES },
    TRPX_OP(3, "ADDSAT_CONST"), OPCL("ADDSAT")
  },
  { AIPO "subsat_const", "-s_", NULL, 1, 1, 0, 1, NO_PARAM,
    { TY_INT, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_alu, NO_POC, alu_subsat_0cst, NO_MES },
    TRPX_OP(3, "SUBSAT_CONST"), OPCL("SUBSAT")
  },
  { AIPO "const_subsat", "_-s", NULL, 1, 1, 0, 1, NO_PARAM,
    { TY_INT, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_alu, NO_POC, alu_subsat_cst0, NO_MES },
    TRPX_OP(3, "CONST_SUBSAT"), OPCL("SUBSATC")
  },
  { AIPO "absdiff_const", "-|_", NULL, 1, 1, 0, 1, NO_PARAM,
    { TY_INT, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_alu, NO_POC, alu_abssub_0cst, NO_MES },
    TRPX_OP(3, "ABSDIFF_CONST"), OPCL("ABSDIFF")
  },
  { AIPO "mul_const", "*_", NULL, 1, 1, 0, 1, NO_PARAM, { TY_INT, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_alu, NO_POC, alu_mul_0cst, NO_MES },
    TRPX_OP(3, "MUL_CONST"), OPCL("MUL")
  },
  { AIPO "div_const", "/_", NULL, 1, 1, 0, 1, NO_PARAM, { TY_INT, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_alu, NO_POC, alu_div_0cst, NO_MES },
    TRPX_OP(3, "DIV_CONST"), OPCL("DIV")
  },
  { AIPO "const_div", "_/", NULL, 1, 1, 0, 1, NO_PARAM, { TY_INT, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_alu, NO_POC, alu_div_cst0, NO_MES },
    TRPX_OP(3, "CONST_DIV"), OPCL("DIVC")
  },
  // nullary
  { AIPO "set_constant", "C", NULL, 1, 0, 0, 1, NO_PARAM, { TY_INT, NULL, NULL},
    { spoc_output_0|spoc_alu, NO_POC, alu_copy_cst, NO_MES },
    TRPX_OP(2, "SET_CONST"), OPCL("SET")
  },
  // not a real one, this is used internally only
  // semantics of "scalar_copy(a, b);" is "*a = *b;"
  { AIPO "scalar_copy", "?=", NULL, 0, 0, 1, 1, NO_PARAM,
      { TY_INT, TY_INT, NULL},
    NOPE_SPOC, NOPE_TRPX, NOPE_OPCL
  },
  // MISC
  // this one may be ignored?!
  { AIPO "copy", "=", NULL, 1, 1, 0, 0, NO_PARAM, NO_PARAM,
    // hmmm... would NO_SPOC do?
    { spoc_input_0|spoc_output_0, NO_POC, alu_unused, NO_MES },
    TRPX_OP(3, "COPY"), OPCL("COPY")
  },
  { // not implemented by SPOC! nor TERAPIX!
    AIPO "cast", "=()", NULL, 1, 1, 0, 0, NO_PARAM, NO_PARAM,
    NO_SPOC, NO_TRPX, NO_OPCL
  },
  { AIPO "threshold", "thr", NULL, 1, 1, 0, 3, NO_PARAM,
    { TY_INT, TY_INT, TY_INT },
    { spoc_input_0|spoc_output_0|spoc_th_0, NO_POC, alu_unused, NO_MES },
    TRPX_OP(5, "THRESHOLD"), OPCL("THRESHOLD")
  },
  // MORPHO
  { AIPO "erode_6c", "E6", NULL, 1, 1, 0, 1, NO_PARAM, { TY_CIP, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_poc_0,
      { { spoc_poc_erode, 6 }, { spoc_poc_unused, 0 } }, alu_unused, NO_MES
    },
    TRPX_NG(10, "ERODE_3_3?"), NO_OPCL
  },
  { AIPO "dilate_6c", "D6", NULL, 1, 1, 0, 1, NO_PARAM, { TY_CIP, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_poc_0,
      { { spoc_poc_dilate, 6 }, { spoc_poc_unused, 0 } }, alu_unused, NO_MES
    },
    TRPX_NG(10, "DILATE_3_3?"), NO_OPCL
  },
  { AIPO "erode_8c", "E8", NULL, 1, 1, 0, 1, NO_PARAM, { TY_CIP, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_poc_0,
      { { spoc_poc_erode, 8 }, { spoc_poc_unused, 0 } }, alu_unused, NO_MES
    },
    TRPX_NG(15, "ERODE_3_3"), OPCLK("INF", "MAX")
  },
  { AIPO "dilate_8c", "D8", NULL, 1, 1, 0, 1,  NO_PARAM, { TY_CIP, NULL, NULL },
    { spoc_input_0|spoc_output_0|spoc_poc_0,
      { { spoc_poc_dilate, 8 }, { spoc_poc_unused, 0 } }, alu_unused, NO_MES
    },
    TRPX_NG(15, "DILATE_3_3"), OPCLK("SUP", "MIN")
  },
  // MEASURES
  { AIPO "global_min", "min", NULL, 0, 1, 1, 0,
    { TY_PIN, NULL, NULL }, NO_PARAM,
    { spoc_input_0 | spoc_measure_0,
      NO_POC, alu_unused, { measure_min, measure_none }
    },
    TRPX_MS(1, 3, "GLOBAL_MIN"), OPCL("MINIMUM")
  },
  { AIPO "global_max", "max", NULL, 0, 1, 1, 0, { TY_PIN, NULL, NULL },
    NO_PARAM, { spoc_input_0 | spoc_measure_0,
                NO_POC, alu_unused, { measure_max, measure_none }
    },
    TRPX_MS(1, 3, "GLOBAL_MAX"), OPCL("MAXIMUM")
  },
  { AIPO "global_min_coord", "min!", NULL, 0, 1, 3, 0,
    { TY_PIN, TY_PUI, TY_PUI }, NO_PARAM,
    { spoc_input_0 | spoc_measure_0,
      NO_POC, alu_unused, { measure_min_coord, measure_none }
    },
    TRPX_MS(5, 3, "GLOBAL_MIN_COORD"), OPCL("MIN_COORD")
  },
  { AIPO "global_max_coord", "max!", NULL, 0, 1, 3, 0,
    { TY_PIN, TY_PUI, TY_PUI }, NO_PARAM,
    { spoc_input_0 | spoc_measure_0,
      NO_POC, alu_unused, { measure_max_coord, measure_none }
    },
    TRPX_MS(5, 3, "GLOBAL_MAX_COORD"), OPCL("MAX_COORD")
  },
  { AIPO "global_vol", "vol", NULL, 0, 1, 1, 0,
    { TY_PIN, NULL, NULL }, NO_PARAM,
    { spoc_input_0 | spoc_measure_0,
      NO_POC, alu_unused, { measure_vol, measure_none }
    },
    TRPX_MS(2, 3, "GLOBAL_VOL"), OPCL("VOLUME")
  },
  // LINEAR
  // cost rather approximated for Terapix
  { AIPO "convolution", "conv", NULL, 1, 1, 0, 3,
    NO_PARAM, { TY_PIN, TY_UIN, TY_UIN }, // kernel, width, height
    // for spoc, only 3x3 convolutions
    { spoc_input_0|spoc_output_0|spoc_poc_0,
      { { spoc_poc_conv, 8 }, { spoc_poc_unused, 0 } }, alu_unused, NO_MES
    },
    // for terapix, this is a special case
    // I'm not sure about the cost model (h*35) for 3x3?
    { -1, -1, -1, -1, 0, 3, false, false, "TERAPIX_UCODE_CONV" },
    // missing at tail: / norm
    OPCLK("ADD", "ZERO")
  },
  // not implemented by SPOC! nor by TERAPIX!
  { AIPO "fast_correlation", "corr", NULL, 1, 2, 0, 1,
    NO_PARAM, { TY_UIN, NULL, NULL },
    NO_SPOC, NO_TRPX, NO_OPCL
  },
  // last entry
  { NULL, NULL, NULL, 0, 0, 0, 0, NO_PARAM, NO_PARAM,
    NO_SPOC, NO_TRPX, NO_OPCL
  }
};

#define FREIA_AIPO_API_SIZE (sizeof(*FREIA_AIPO_API)/sizeof(freia_api_t))

/* @returns the description of a FREIA AIPO API function.
 * may be moved elswhere. raise an error if not found.
 */
const freia_api_t * hwac_freia_api(const char* function)
{
  static hash_table cache = NULL;
  if (!cache)
  {
    const freia_api_t * api;
    cache = hash_table_make(hash_string, 0);
    for (api = FREIA_AIPO_API; api->function_name; api++)
      hash_put(cache, api->function_name, api);
  }
  return (const freia_api_t *)
    (hash_defined_p(cache, function)? hash_get(cache, function): NULL);
}

/* returns the index of the description of an AIPO function
 */
int hwac_freia_api_index(const string function)
{
  const freia_api_t * api = hwac_freia_api(function);
  return api? (api - FREIA_AIPO_API): -1;
}

const freia_api_t * get_freia_api(int index)
{
  // pips_assert("index exists", index>=0 && index<(int) FREIA_AIPO_API_SIZE);
  return &FREIA_AIPO_API[index];
}

const freia_api_t * get_freia_api_vtx(dagvtx v)
{
  return get_freia_api(vtxcontent_opid(dagvtx_content(v)));
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
  case spoc_type_sni: return "sni";
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
void freia_spoc_set_operation(const freia_api_t * api, _int * type, _int * id)
{
  pips_assert("no type set", *type == spoc_type_nop);

  // set type which is enough for staging?
  if (api->spoc.used==spoc_not_implemented)
    *type = spoc_type_sni;
  else if (api->spoc.used & spoc_alu)
    *type = spoc_type_alu;
  else if (api->spoc.used & (spoc_poc_0|spoc_poc_1))
    *type = spoc_type_poc;
  else if (api->spoc.used & (spoc_th_0|spoc_th_1))
    *type = spoc_type_thr;
  else if (api->spoc.used & (spoc_measure_0|spoc_measure_1))
    *type = spoc_type_mes;
  else
    *type = spoc_type_nop; // e.g. for copy? (...)

  // set details
  *id = hwac_freia_api_index(api->function_name);
}

/* @brief get freia further parameters, skipping image ones
 */
list freia_get_params(const freia_api_t * api, list args)
{
  int skip = api->arg_img_in + api->arg_img_out;
  while (skip--) args = CDR(args);
  pips_assert("number of scalar args is ok",
              gen_length(args)==api->arg_misc_in+api->arg_misc_out);
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

expression freia_get_nth_scalar_param(const dagvtx v, int n)
{
  return EXPRESSION(CAR(gen_nthcdr(n-1, freia_get_vertex_params(v))));
}

int freia_max_pixel_value(void)
{
  int bpp = FREIA_DEFAULT_BPP;
  switch (bpp)
  {
  case 8: return 0xff;
  case 16: return 0xffff;
  default:
    pips_user_error("expecting 8 or 16 for pixel size, got %d", bpp);
    return 0;
  }
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
   string_buffer head2, // function header, the retour
   hash_table params,  // argument/variable to parameter mapping
   int * nparams)      // current number of parameters
{
  const freia_api_t * api = get_freia_api(napi);
  args = freia_get_params(api, args);
  list res = NIL;
  bool merge = get_bool_property("FREIA_MERGE_ARGUMENTS");

  // important shortcut, taken when ops are switched to "copy" for
  // some reason, then the next assert would not be consistent.
  if (api->arg_misc_in==0 && api->arg_misc_out==0)
    return NIL;

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
          if (head2) sb_cat(head2, ",\n  ", api->arg_in_types[i], " ", name);
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
        if (head2) sb_cat(head2, ",\n  ", api->arg_in_types[i], " ", name);
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
    if (head2) sb_cat(head2, ",\n  ", api->arg_out_types[i], " ", name);
    if (params)
      hash_put(params, e, name);
    else
      free(name);
    res = CONS(expression, copy_expression(e), res);
  }

  return gen_nreverse(res);
}

/* @brief build all is well freia constant
 */
call freia_ok(void)
{
  // how to build the "FREIA_OK" enum value constant?
  return make_call(local_name_to_top_level_entity("0"), NIL);
}

/* @brief tell whether it is an assignment to ignore?
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
  if (var && var!=entity_undefined &&
      entity_variable_p(var) && entity_pointer_p(var))
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

/* returns whether the statement is a FREIA call.
 */
bool freia_statement_aipo_call_p(const statement s)
{
  // very partial as well
  instruction i = statement_instruction(s);
  if (instruction_call_p(i)) {
    call c = instruction_call(i);
    entity called = call_function(c);
    if (entity_freia_api_p(called))
      return true;
    else if (freia_assignment_p(called))
    {
      list la = call_arguments(c);
      pips_assert("2 arguments to assign", gen_length(la));
      syntax op2 = expression_syntax(EXPRESSION(CAR(CDR(la))));
      if (syntax_call_p(op2))
        return entity_freia_api_p(call_function(syntax_call(op2)));
    }
    else if (ENTITY_C_RETURN_P(called))
    {
      list la = call_arguments(c);
      if (gen_length(la)==1) {
        syntax op = expression_syntax(EXPRESSION(CAR(la)));
        if (syntax_call_p(op))
          return entity_freia_api_p(call_function(syntax_call(op)));
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
      if (entity_variable_p(var) && var!=skip &&
          !freia_image_variable_p(var) && (entity_scalar_p(var) || entity_pointer_p(var)))
        set_add_element(s, s, var);
    }
  }
}

/************************************************ SCALAR DEPENDENCIES (HACK) */

/* is there a simple scalar (no image) rw dependency from s to t?
 * WW deps are ignored... should be okay of computed in order?
 * @param s source statement
 * @param t target statement
 * @param vars if set, return list of scalars which hold the dependencies
 */
static bool
real_freia_scalar_rw_dep(const statement s, const statement t, list * vars)
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

// Hmmm...
// this is called very often and the performance is abysmal
// even on not so big codes, so here is a cached version...

// { s -> { t -> (vars) } }
static hash_table dep_cache = NULL;

void freia_init_dep_cache(void)
{
  pips_assert("dep_cache is NULL", !dep_cache);
  dep_cache = hash_table_make(hash_pointer, 0);
}

void freia_close_dep_cache(void)
{
  pips_assert("dep_cache is not NULL", dep_cache);

  HASH_FOREACH(statement, s1, hash_table, h, dep_cache) {
    HASH_FOREACH(statement, s2, list, l, h)
      if (l) gen_free_list(l);
    hash_table_free(h);
  }

  hash_table_free(dep_cache), dep_cache = NULL;
}

// cached version
bool freia_scalar_rw_dep(const statement s, const statement t, list * vars)
{
  pips_assert("dep_cache is not NULL", dep_cache);

  // short circuit
  if (s==t || !s || !t) return false;

  // first level
  if (!hash_defined_p(dep_cache, s))
    hash_put(dep_cache, s, hash_table_make(hash_pointer, 0));
  hash_table h = (hash_table) hash_get(dep_cache, s);

  // second level
  if (!hash_defined_p(h, t)) {
    list l = NIL;
    real_freia_scalar_rw_dep(s, t, &l);
    hash_put(h, t, l);
  }

  list l = (list) hash_get(h, t);
  if (vars) *vars = gen_copy_seq(l);
  return l!=NIL;
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

static bool is_freia_this_call(const statement s, const string fname)
{
  call c = freia_statement_to_call(s);
  const char* called = c? entity_user_name(call_function(c)): "";
  return same_string_p(called, fname);
}

bool is_freia_alloc(const statement s)
{
  return is_freia_this_call(s, FREIA_ALLOC);
}

bool is_freia_dealloc(const statement s)
{
  return is_freia_this_call(s, FREIA_FREE);
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
    (dagvtx_opid(v1), call_arguments(c1), NULL, NULL, NULL, NULL),
    lp2 = freia_extract_params
    (dagvtx_opid(v1), call_arguments(c2), NULL, NULL, NULL, NULL);
  bool same = lexpression_equal_p(lp1, lp2);
  gen_free_list(lp1), gen_free_list(lp2);
  // should also check that there is no w effects on parameters in between
  return same;
}

entity freia_create_helper_function(const string function_name, list lparams)
{
  // build helper entity
  entity example = local_name_to_top_level_entity("freia_aipo_add");
  pips_assert("example is a function", entity_function_p(example));
  entity helper = make_empty_function(function_name,
        copy_type(functional_result(type_functional(entity_type(example)))),
                                      make_language_c());

  // update type of parameters
  list larg_params = NIL;
  FOREACH(expression, e, lparams)
  {
    debug_on("RI_UTILS_DEBUG_LEVEL");
    type t = expression_to_user_type(e);
    debug_off();
    larg_params = CONS(parameter,
                       make_parameter(t,
                                      make_mode_value(),
                                      make_dummy_unknown()),
                       larg_params);
  }
  larg_params = gen_nreverse(larg_params);
  module_functional_parameters(helper) = larg_params;

  return helper;
}

/* substitute those statement in ls that are in dag d and accelerated
 * by a call to function_name(lparams)
 * also update sets of remainings and global_remainings
 *
 * the function must chose one of the statements?
 * the current implementation is a buggy mess.
 * it should be fully re-thought from scratch.
 *
 * what should be done, is to chose the FIRST statement after the LAST
 * dependency of the compiled statements? at least it must exist beacause
 * one of the dependent statement must appear after that, but is it enough
 * for all possible graph? so maybe we should do a topological sort of
 * all the statements and reshuffle everything instead of trying to preserver
 * the initial code as much as possible?
 */
int freia_substitute_by_helper_call(
  // dag reminder, may be null
  dag d,
  set global_remainings,
  set remainings,
  list /* of statement */ ls,
  const string function_name,
  list lparams,
  set helpers,    // for signatures
  int preceeding) // statement which stored the previous insert, -1 for none
{
  pips_debug(7, "%d statements, must come after statement %d\n",
             (int) gen_length(ls), preceeding);

  // buid the set of statements that are not yet computed
  set not_dones = set_make(set_pointer), dones = set_make(set_pointer);
  if (d)
  {
    FOREACH(dagvtx, vs, dag_vertices(d))
    {
      pstatement ps = vtxcontent_source(dagvtx_content(vs));
      if (pstatement_statement_p(ps))
        set_add_element(not_dones, not_dones, pstatement_statement(ps));
    }
  }
  // dones: those statements which are handled by this helper
  set_difference(dones, remainings, not_dones);
  // now update remainings & global_remainings
  set_difference(remainings, remainings, dones);
  set_difference(global_remainings, global_remainings, dones);

  // replace first statement of dones in ls,
  // which comes after the preceeding ones.
  statement found = NULL, sos = NULL;
  FOREACH(statement, sc, ls)
  {
    pips_debug(5, "in statement %" _intFMT "\n", statement_number(sc));
    if (set_belong_p(dones, sc))
    {
      sos = sos? (statement_number(sc)<statement_number(sos)? sc: sos): sc;
      if (statement_number(sc)>preceeding)
      {
        pips_assert("statement is a call", statement_call_p(sc));
        found = found?
          (statement_number(sc)<statement_number(found)? sc: found): sc;
      }
    }
  }

  if (!found)
  {
    pips_user_warning("no statement found after preceeding insertion, "
                      "using first statement as backup...\n");
    // use backup
    found = sos;
  }

  pips_assert("some statement found", found);

  // create and record helper function
  entity helper = freia_create_helper_function(function_name, lparams);
  set_add_element(helpers, helpers, helper);

  // substitute by call to helper
  call c = make_call(helper, lparams);

  hwac_replace_statement(found, c, false);

  set_free(not_dones);
  set_free(dones);

  return statement_number(found);
}

static statement find_aipo_statement(list ls, bool before)
{
  pips_assert("some statements", ls);
  pips_debug(8, "choosing among %d statements\n", (int) gen_length(ls));

  statement found = NULL;

  FOREACH(statement, s, ls)
  {
    pips_debug(8, "stmt %" _intFMT "\n", statement_number(s));
    if (before)
    {
      if (!found || statement_number(found)>statement_number(s))
        found = s;
    }
    else // after
    {
      if (!found || statement_number(found)<statement_number(s))
        found = s;
    }
  }

  // ??? au pif
  if (!found && ls)
  {
    pips_debug(8, "no aipo call found... backup plan?\n");
    if (before)
      found = STATEMENT(CAR(gen_last(ls))); // list is reversed...
    else
      found = STATEMENT(CAR(ls));
  }

  return found;
}

/* insert statements to actual code sequence in "ls"
 * *BEWARE* that ls is assumed to be in reverse order...
 */
void freia_insert_added_stats(list ls, list stats, bool before)
{
  if (stats)
  {
    statement sref = find_aipo_statement(ls, before);
    pips_debug(8, "sref for %s is %" _intFMT "\n",
               before? "before": "after", statement_number(sref));
    instruction iref = statement_instruction(sref);
    statement newstat = instruction_to_statement(iref);
    // transfer comments and some cleanup...
    statement_comments(newstat) = statement_comments(sref);
    statement_comments(sref) = string_undefined;
    statement_number(sref) = STATEMENT_NUMBER_UNDEFINED;
    // pretty ugly because return must be handled especially...
    // ??? not sure that it is ok if !before?
    if (instruction_call_p(iref) &&
        ENTITY_C_RETURN_P(call_function(instruction_call(iref))))
    {
      call c = instruction_call(iref);
      if (!expression_constant_p(EXPRESSION(CAR(call_arguments(c)))))
      {
        // must split return...
        pips_internal_error("return splitting not implemented yet...");
      }
      else
      {
        if (before)
          stats = gen_nconc(stats, CONS(statement, newstat, NIL));
        else
          // ???
          stats = CONS(statement, newstat, stats);
      }
    }
    else
    {
      if (before)
        stats = CONS(statement, newstat, stats);
      else
        stats = gen_nconc(stats, CONS(statement, newstat, NIL));
    }
    statement_instruction(sref) =
      make_instruction_sequence(make_sequence(stats));
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

#define E_WRITE(v) (v)
#define E_READ(v) ((void*)(((_int)v)+1))

/* return the argument number, starting from 1, of this reference
 * or 0 if not found.
 */
static int reference_argument_number(const reference r, const list le)
{
  int n = 0, i=0;
  FOREACH(expression, e, le)
  {
    i++;
    if (expression_reference_p(e) && expression_reference(e)==r)
      n = i;
  }
  return n;
}

/* tell about the image effect. if in doubt, return true.
 */
static bool reference_written_p(
  const reference r,
  const hash_table signatures)
{
  const char* why = "default", *func = "?";
  bool written = false;
  call c = (call) gen_get_ancestor(call_domain, r);
  if (c)
  {
    entity called = call_function(c);
    func = entity_local_name(called);
    int n = reference_argument_number(r, call_arguments(c));
    if (n==0)
      written=true, why = "not found";
    else
    {
      const freia_api_t * api = hwac_freia_api(entity_local_name(called));
      if (api)
        written = n <= (int) api->arg_img_out, why = "api";
      else
        if (signatures && hash_defined_p(signatures, called))
          written = n <= (_int) hash_get(signatures, called), why = "sig";
        else
          written = true, why = "no sig";
    }
  }
  else
    // in doubt, assume a write effect?
    pips_internal_error("reference to %s outside of a call?\n",
                        entity_local_name(reference_variable(r)));

  pips_debug(8, "reference to %s is %s (%s in %s)\n",
             entity_local_name(reference_variable(r)),
             written? "written": "read", why, func);

  return written;
}

typedef struct {
  // built image occurences:
  // image -> set of statements with written effects
  // image+1 -> set of statements with read effects
  hash_table occs;
  // enclosing statement for inner recursion
  statement enclosing;
  // set of statements with image operations
  set image_occs_stats;
  // statement -> set of W images
  // statement+1 -> set of R images
  hash_table image_stats;
  // helper entity -> number of written image args
  const hash_table signatures;
} occs_ctx;

/* hack to help replace use-def chains which did not work initially with C.
 * occurrences is: <image entity> -> { set of statements }
 * this is really a ugly hack, sorry!
 * ??? I should also consider declarations, but as they should only
 * contain image allocations so there should be no problem.
 */

static bool check_ref(reference r, occs_ctx * ctx)
{
  entity v = reference_variable(r);
  if (freia_image_variable_p(v))
  {
    // ensure that target set exists
    if (!hash_defined_p(ctx->occs, v))
    {
      hash_put(ctx->occs, E_WRITE(v), set_make(set_pointer));
      hash_put(ctx->occs, E_READ(v), set_make(set_pointer));
    }
    bool written = reference_written_p(r, ctx->signatures);
    set stats = (set) hash_get(ctx->occs, written? E_WRITE(v): E_READ(v));
    // get containing statement
    statement up = ctx->enclosing? ctx->enclosing:
      (statement) gen_get_ancestor(statement_domain, r);
    // which MUST exist?
    pips_assert("some containing statement", up);
    if (ctx->image_stats) {
      set sop = (set) hash_get(ctx->image_stats,
                               written? E_WRITE(up): E_READ(up));
      set_add_element(sop, sop, (void*) v);
    }
    // store result
    set_add_element(stats, stats, (void*) up);
    if (ctx->image_occs_stats)
      set_add_element(ctx->image_occs_stats, ctx->image_occs_stats, up);

    pips_debug(9, "entity %s in statement %"_intFMT"\n",
               entity_name(v), statement_number(up));
  }
  return true;
}

static bool check_stmt(statement s, occs_ctx * ctx)
{
  // ensure existing set
  if (ctx->image_stats) {
    pips_debug(8, "creating for statement %"_intFMT"\n",
               statement_number(s));
    hash_put(ctx->image_stats, E_WRITE(s), set_make(set_pointer));
    hash_put(ctx->image_stats, E_READ(s), set_make(set_pointer));
  }
  ctx->enclosing = s;
  FOREACH(entity, var, statement_declarations(s))
    gen_context_recurse(entity_initial(var), ctx,
                        reference_domain, gen_true, check_ref);
  ctx->enclosing = NULL;
  return true;
}

/* @param image_occs_stats set of statements with image occurences (may be NULL)
 * @param signatures helper entity -> (_int) # out args
 * @return build occurrence hash table: { entity -> set of statements }
 */
hash_table freia_build_image_occurrences(
  statement s,
  set image_occs_stats,
  hash_table image_stats,
  const hash_table signatures)
{
  pips_debug(7, "entering\n");
  occs_ctx ctx = { hash_table_make(hash_pointer, 0), NULL,
                   image_occs_stats, image_stats, signatures };
  gen_context_multi_recurse(s, &ctx,
                            statement_domain, check_stmt, gen_null,
                            reference_domain, check_ref, gen_null,
                            NULL);
  pips_debug(7, "done\n");
  return ctx.occs;
}

/* cleanup occurrence data structure
 */
void freia_clean_image_occurrences(hash_table occs)
{
  HASH_FOREACH(entity, v, set, s, occs)
    set_free(s);
  hash_table_free(occs);
}

/************************************************************* VARIOUS TESTS */

/* @brief whether api available with SPoC
 */
bool freia_aipo_spoc_implemented(const freia_api_t * api)
{
  pips_assert("some api", api!=NULL);
  return api->spoc.used!=spoc_not_implemented;
}

/* @brief whether api available with Ter@pix
 */
bool freia_aipo_terapix_implemented(const freia_api_t * api)
{
  pips_assert("some api", api!=NULL);
  // some special cases are handled manually
  if (same_string_p(api->function_name, "undefined") ||
      // this one is inserted to deal with replicated measures
      same_string_p(api->function_name, AIPO "scalar_copy"))
    return true;
  return api->terapix.cost!=-1;
}

/******************************************************* CONVOLUTION HELPERS */

/* @brief is it the convolution special case?
 */
bool freia_convolution_p(dagvtx v)
{
  return same_string_p(dagvtx_operation(v), "convolution");
}

/* @brief get width & height of convolution
 * @return whether they could be extracted
 */
bool freia_convolution_width_height(dagvtx v, _int * pw, _int * ph, bool check)
{
  // pips_assert("vertex is convolution", ...)
  list largs = freia_get_vertex_params(v);
  if (check) pips_assert("3 args to convolution", gen_length(largs)==3);
  bool bw = expression_integer_value(EXPRESSION(CAR(CDR(largs))), pw);
  if (check) pips_assert("constant convolution width", bw);
  if (check) pips_assert("odd convolution width", ((*pw)%2)==1);
  bool bh = expression_integer_value(EXPRESSION(CAR(CDR(CDR(largs)))), ph);
  if (check) pips_assert("constant convolution height", bh);
  if (check) pips_assert("odd convolution height", ((*ph)%2)==1);
  return bw && bh;
}

static void clean_stats_to_image(hash_table s2i)
{
  HASH_FOREACH(statement, st, set, imgs, s2i)
    set_free(imgs);
  hash_table_free(s2i);
}

/****************************************************** NEW IMAGE ALLOCATION */

/*
static statement image_alloc(entity v)
{
  return make_assign_statement
    (entity_to_expression(v),
     call_to_expression(make_call(local_name_to_top_level_entity(FREIA_ALLOC),
                                  NIL)));
}
*/

/* generate statement: "freia_free(v);"
 */
static statement image_free(entity v)
{
  return call_to_statement(
    make_call(local_name_to_top_level_entity(FREIA_FREE),
              CONS(expression, entity_to_expression(v), NIL)));
}

struct related_ctx {
  const entity img;
  const hash_table new_images;
  const hash_table image_stats;
  bool write_only;
  bool some_effect;
};

/* is there an effect to this image or related images in the statement?
 */

/* are img1 and img2 related?
 */
static bool related_images_p(const entity img1, const entity img2,
                             const hash_table new_images)
{
  entity oimg1 = NULL, oimg2 = NULL;
  if (img1==img2) return true;
  if (hash_defined_p(new_images, img1))
    oimg1 = (entity) hash_get(new_images, img1);
  if (hash_defined_p(new_images, img2))
    oimg2 = (entity) hash_get(new_images, img2);

  pips_debug(9, "images: %s -> %s / %s -> %s\n",
             entity_local_name(img1), oimg1? entity_local_name(oimg1): "NONE",
             entity_local_name(img2), oimg2? entity_local_name(oimg2): "NONE");

  if (oimg1 && oimg1==img2) return true;
  if (oimg2 && img1==oimg2) return true;
  // this is really the one which should be triggered?
  return oimg1 && oimg2 && oimg1==oimg2;
}

static bool related_effect(statement s, struct related_ctx * ctx)
{
  pips_debug(8, "on statement %"_intFMT"\n", statement_number(s));
  entity written = NULL;
  SET_FOREACH(entity, w, (set) hash_get(ctx->image_stats, E_WRITE(s)))
  {
    if (related_images_p(ctx->img, w, ctx->new_images))
    {
      pips_debug(8, "W relation for %s & %s\n",
                 entity_local_name(ctx->img), entity_local_name(w));
      ctx->some_effect = true;
      gen_recurse_stop(NULL);
      return false;
    }
    written = w;
  }
  if (!ctx->write_only)
  {
    pips_assert("some written stuff", written);
    SET_FOREACH(entity, r, (set) hash_get(ctx->image_stats, E_READ(s)))
    {
      if (related_images_p(ctx->img, r, ctx->new_images))
      {
        pips_debug(8, "R relation for %s & %s\n",
                   entity_local_name(ctx->img), entity_local_name(written));
        ctx->some_effect = true;
        gen_recurse_stop(NULL);
        return false;
      }
    }
  }
  return true;
}

static bool some_related_image_effect(
  statement s,
  entity img,
  hash_table new_images,
  hash_table image_stats,
  bool write_only)
{
  struct related_ctx ctx = { img, new_images, image_stats, write_only, false };
  gen_context_recurse(s, &ctx, statement_domain, related_effect, gen_null);
  return ctx.some_effect;
}

/* tell whether there is no image processing statements between s1 and l2
 */
static bool only_minor_statements_in_between(
  // the new image we are interrested in
  entity image,
  // new to old image mapping
  hash_table new_images,
  // [RW](stats) -> sets
  hash_table image_stats,
  // list of statements being considered (within a sequence)
  list ls,
  // image production statement
  statement s1,
  // list of "use" statements
  list l2,
  // set of statements with image occurences
  set image_occurences)
{
  bool s1_seen = false, in_sequence = false;
  pips_assert("consistent statement & list", !gen_in_list_p(s1, l2));
  int n2 = gen_length(l2);

  ifdebug(8) {
    pips_debug(8, "img=%s s=%"_intFMT"\nls = (\n",
               entity_local_name(image), statement_number(s1));
    FOREACH(statement, sls, ls)
      pips_debug(8, " - %"_intFMT" %s\n", statement_number(sls),
                 sls==s1? "!": gen_in_list_p(sls, l2)? "*": "");
    pips_debug(8, ")\n");
  }

  // scan the sequence list, looking for s1 & l2 statements
  FOREACH(statement, s, ls)
  {
    if (!in_sequence && s==s1)
      in_sequence = true, s1_seen = true;
    else if (in_sequence && gen_in_list_p(s, l2))
    {
      n2--;
      if (n2) {
        // we are still going on, BUT we must check that this statement
        // does not rewrite the image we are interested in and may change back.
        if (some_related_image_effect(s, image, new_images, image_stats, true))
          return false;
      }
      // we stop when we have seen all intermediate statements
      else
        return true;
    }
    else if (in_sequence && set_belong_p(image_occurences, s))
    {
      // let us try to do something intelligent here...
      // if images are unrelated to "image", then there will be no interaction
      if (some_related_image_effect(s, image, new_images, image_stats, false))
        return false;
    }
  }

  // ??? should really be an error...
  pips_user_warning("should not get there: s1 seen=%s, seq=%s, n2=%d\n",
                    bool_to_string(s1_seen), bool_to_string(in_sequence), n2);

  // let us be optimistic, this is a prototype
  return true;
}

/* insert image allocation if needed, for intermediate image inserted before
 * if an image is used only twice, then it is switched back to the initial one
 *
 * This could/should be improved:
 * - temporary images kept could be reused if possible, instead of new ones
 * - not sure about the condition to move back to the initial image
 *
 * @param ls list of statements to consider
 * @param images list of entities to check and maybe allocate
 * @param init new image -> initial image
 * @param signatures helper -> _int # written args ahead (may be NULL)
 *
 * @return list of actually allocated images
 */
list freia_allocate_new_images_if_needed
(list ls,
 list images,
 // R(entity) and W(entity) -> set of statements
 const hash_table occs,
 // entity -> entity
 const hash_table init,
 // entity -> # out image
 const hash_table signatures)
{
  // check for used images
  set img_stats = set_make(set_pointer);
  hash_table image_stats_detailed = hash_table_make(hash_pointer, 0);
  sequence sq = make_sequence(ls);
  hash_table newoccs =
    freia_build_image_occurrences((statement) sq, img_stats,
                                  image_stats_detailed, signatures);
  sequence_statements(sq) = NIL;
  free_sequence(sq);

  list allocated = NIL;
  FOREACH(entity, v, images)
  {
    if (!hash_defined_p(newoccs, v))
      // no written occurences, the image is not kept
      continue;

    if (get_bool_property("FREIA_REUSE_INITIAL_IMAGES"))
    {
      set where_write = (set) hash_get(newoccs, E_WRITE(v));
      set where_read = (set) hash_get(newoccs, E_READ(v));
      int nw = set_size(where_write), nr = set_size(where_read);

      pips_debug(6, "image %s used %d+%d statements\n", entity_name(v), nw, nr);

      // ??? should be used once only in the statement if written!
      // how to I know about W/R for helper functions?
      // its siblings should also be take into account

      // n>1 ??
      // if used only twice, substitude back to initial variable???
      // well, it may depends whether the the initial variable is used?
      // there is possibly something intelligent to do here, but it should
      // be checked carefully...
      if (nw==1 && nr>=1 && hash_defined_p(init, v))
      {
        entity old = (entity) hash_get(init, v);

        // get statements
        list l1 = set_to_list(where_write), l2 = set_to_list(where_read);
        statement s1 = STATEMENT(CAR(l1));
        gen_free_list(l1), l1 = NIL;

        // does not interact with possibly used old
        // if we could differentiate read & write, we could do better,
        // but the information is not currently available.
        bool skip = false;
        if (hash_defined_p(newoccs, old))
        {
          set old_write = (set) hash_get(newoccs, E_WRITE(old));
          set old_read = (set) hash_get(newoccs, E_READ(old));
          if (set_belong_p(old_write, s1))
            skip = true;
          else
          {
            FOREACH(statement, s2, l2)
              if (set_belong_p(old_read, s2))
                skip = true;
          }
          // note that we can handle a read in s1 and a write in s2
        }

        pips_debug(7, "testing for %s -> %s: %s\n",
                   entity_local_name(v), entity_local_name(old),
                   skip?"skip":"ok");

        // do we want to switch back?
        if (!skip &&
            only_minor_statements_in_between(v, init, image_stats_detailed,
                                             ls, s1, l2, img_stats))
        {
          // yes, they are successive, just remove?? Am I that sure???
          // ??? hmmm, maybe we could have :
          //   X   = stuff(...)
          //   X_1 = stuff(...)
          //   ... = stuff(X_1, ...)
          //   ... = stuff(X...)
          // where X_1 -> X is just a bad idea because it overwrites X?
          // I'm not sure this can happen with AIPO, as X would be X_2
          // and will not be changed?
          pips_debug(7, "substituting back %s by %s\n",
                     entity_local_name(v), entity_local_name(old));

          // we can perform any substitution here
          // note that it may be generated helper functions
          freia_switch_image_in_statement(s1, v, old, true);
          FOREACH(statement, s2, l2)
            freia_switch_image_in_statement(s2, v, old, false);

          // create/update uses of "old" to avoid interactions
          if (!hash_defined_p(newoccs, old))
          {
            hash_put(newoccs, E_WRITE(old), set_make(set_pointer));
            hash_put(newoccs, E_READ(old), set_make(set_pointer));
          }
          set old_write = (set) hash_get(newoccs, E_WRITE(old));
          set_add_element(old_write, old_write, s1);
          set old_read = (set) hash_get(newoccs, E_READ(old));
          FOREACH(statement, s2, l2)
            set_add_element(old_read, old_read, s2);
        }
        else
          allocated = CONS(entity, v, allocated);
      }
      else
        allocated = CONS(entity, v, allocated);
    }
    else
      allocated = CONS(entity, v, allocated);
  }

  clean_stats_to_image(image_stats_detailed);
  freia_clean_image_occurrences(newoccs);
  set_free(img_stats);

  // allocate those which are finally used
  if (allocated)
  {
    pips_assert("some statements", ls!=NIL);
    statement first = STATEMENT(CAR(ls));
    statement last = STATEMENT(CAR(gen_last(ls)));
    pips_assert("at least two statements", first!=last);

    FOREACH(entity, v, images)
    {
      pips_debug(7, "allocating image %s\n", entity_name(v));
      // add_declaration_statement(first, v); // NO, returned
      // insert_statement(first, image_alloc(v), true); // NO, init
      // ??? deallocation should be at end of allocation block...
      entity model = (entity) hash_get(init, v);
      bool before = false;
      statement s = freia_memory_management_statement(model, occs, false);
      if (!s)
      {
        pips_user_warning("not quite sure where to put %s deallocation\n",
                          entity_name(v));
        // hmmm... try at the program level
        statement ms = get_current_module_statement();
        if (statement_sequence_p(ms))
        {
          s = STATEMENT(CAR(
             gen_last(sequence_statements(statement_sequence(ms)))));
          // hmmm... does not seem to work properly
          //if (statement_call_p(s) &&
          // ENTITY_C_RETURN_P(call_function(statement_call(s))))
          before = true;
        }
      }
      insert_statement(s? s: last, image_free(v), before);
    }
  }

  return allocated;
}

/************************************************************* AIPO COUNTERS */

/* @return the number for FREIA AIPO ops in dag
 */
int freia_aipo_count(dag d, int * pa, int * pc)
{
  int aipos = 0, copies = 0;
  FOREACH(dagvtx, v, dag_vertices(d))
  {
    string op = dagvtx_function_name(v);
    if (strncmp(op, AIPO, strlen(AIPO))==0) aipos++;
    // handle exceptions afterwards
    if (same_string_p(op, AIPO "copy")) copies++, aipos--;
    else if (same_string_p(op, AIPO "cast")) copies++, aipos--;
    else if (same_string_p(op, AIPO "scalar_copy")) aipos--;
  }
  *pa = aipos, *pc = copies;
  return aipos+copies;
}

/******************************************************* BUILD OUTPUT IMAGES */

static void oi_call_rwt(call c, set images)
{
  entity called = call_function(c);
  list args = call_arguments(c);
  if (!args) return;
  if (ENTITY_RETURN_P(called) ||
      same_string_p(entity_local_name(called), FREIA_OUTPUT))
  {
    entity var = expression_to_entity(EXPRESSION(CAR(args)));
    if (freia_image_variable_p(var))
      set_add_element(images, images, var);
  }
}

static void oi_stmt_rwt(statement s, set images)
{
  FOREACH(entity, var, statement_declarations(s))
    gen_context_recurse(entity_initial(var), images,
                        call_domain, gen_true, oi_call_rwt);
}

/* @return the set of images which are output somehow
 * this is a little bit simplistic...
 * read output effects of statement?
 */
set freia_compute_output_images(entity module, statement s)
{
  set images = set_make(set_pointer);

  // image formal parameters
  FOREACH(entity, var, module_functional_parameters(module))
  {
    if (freia_image_variable_p(var))
      set_add_element(images, images, var);
  }

  // some image uses in the code
  gen_context_multi_recurse(s, images,
                            statement_domain, gen_true, oi_stmt_rwt,
                            call_domain, gen_true, oi_call_rwt,
                            NULL);
  return images;
}

set freia_compute_current_output_images(void)
{
  return freia_compute_output_images(get_current_module_entity(),
                                     get_current_module_statement());
}

/******************************************************** MIGRATE STATEMENTS */

static string stat_nb(statement s)
{
  return itoa((int) statement_number(s));
}

/*
  migrate a subset of statements in a sequence so that they are close together.
  the relative order of statements is kept. the sequence is updated.
  @param sq sequence to update, may be NULL
  @param stats statements to move together
  @param before statements that must appear before, may be NULL
*/
void freia_migrate_statements(sequence sq, const set stats, const set before)
{
  ifdebug(4) {
    pips_debug(4, "migrating %d statements in %p\n", set_size(stats), sq);
    set_fprint(stderr, "stats", stats, (gen_string_func_t) stat_nb);
    set_fprint(stderr, "before", before, (gen_string_func_t) stat_nb);
  }

  // nothing to do
  if (sq==NULL)
  {
    pips_assert("nothing to migrate", set_size(stats)<=1);
    return;
  }

  // nothing to do either
  if (set_size(stats)==0)
    return;

  // build before/in/end statement lists in reverse order
  list lbefore = NIL, lin = NIL, lend = NIL;
  FOREACH(statement, s, sequence_statements(sq))
  {
    if (set_belong_p(stats, s))
      lin = CONS(statement, s, lin);
    else
    {
      if (before && set_belong_p(before, s))
        lbefore = CONS(statement, s, lbefore);
      else
      {
        if (lin)
          lend = CONS(statement, s, lend);
        else
          lbefore = CONS(statement, s, lbefore);
      }
    }
  }

  // check consistency
  pips_assert("all statements seen", set_size(stats)==(int) gen_length(lin));

  // update sequence
  gen_free_list(sequence_statements(sq));
  lin = gen_nconc(gen_nreverse(lin), gen_nreverse(lend));
  sequence_statements(sq) = gen_nconc(gen_nreverse(lbefore), lin);
}

/* extract values from a kernel definition
 * return 9 values, expected to be 0/1 elsewhere...
 * @return whether it succeeded
 */
static bool freia_extract_kernel(
  expression e,
  bool strict, // whether all values must be known, if not 1 is assumed
  intptr_t * k00, intptr_t * k10, intptr_t * k20,
  intptr_t * k01, intptr_t * k11, intptr_t * k21,
  intptr_t * k02, intptr_t * k12, intptr_t * k22)
{
  // set default value anyway
  *k00 = 1, *k10 = 1, *k20 = 1,
  *k01 = 1, *k11 = 1, *k21 = 1,
  *k02 = 1, *k12 = 1, *k22 = 1;

  // analyse kernel
  if (!expression_reference_p(e)) return !strict;
  entity var = expression_variable(e);
  // ??? should check const...
  value val = entity_initial(var);
  if (!value_expression_p(val)) return !strict;
  expression ival = value_expression(val);
  if (!brace_expression_p(ival)) return !strict;
  list iargs = call_arguments(syntax_call(expression_syntax(ival)));
  pips_assert("must be a 3x3 kernel...", gen_length(iargs)==9);
  if (!expression_integer_value(EXPRESSION(CAR(iargs)), k00) && strict)
    return false;
  iargs = CDR(iargs);
  if (!expression_integer_value(EXPRESSION(CAR(iargs)), k10) && strict)
    return false;
  iargs = CDR(iargs);
  if (!expression_integer_value(EXPRESSION(CAR(iargs)), k20) && strict)
    return false;
  iargs = CDR(iargs);
  if (!expression_integer_value(EXPRESSION(CAR(iargs)), k01) && strict)
    return false;
  iargs = CDR(iargs);
  if (!expression_integer_value(EXPRESSION(CAR(iargs)), k11) && strict)
    return false;
  iargs = CDR(iargs);
  if (!expression_integer_value(EXPRESSION(CAR(iargs)), k21) && strict)
    return false;
  iargs = CDR(iargs);
  if (!expression_integer_value(EXPRESSION(CAR(iargs)), k02) && strict)
    return false;
  iargs = CDR(iargs);
  if (!expression_integer_value(EXPRESSION(CAR(iargs)), k12) && strict)
    return false;
  iargs = CDR(iargs);
  if (!expression_integer_value(EXPRESSION(CAR(iargs)), k22) && strict)
    return false;
  iargs = CDR(iargs);
  pips_assert("end of list reached", iargs==NIL);
  return true;
}

/* vertex-based version
 */
bool freia_extract_kernel_vtx(
  dagvtx v, bool strict,
  intptr_t * k00, intptr_t * k10, intptr_t *k20,
  intptr_t * k01, intptr_t * k11, intptr_t *k21,
  intptr_t * k02, intptr_t * k12, intptr_t *k22)
{
  list largs = freia_get_vertex_params(v);
  // for convolution there is one kernel & two args
  // pips_assert("one kernel", gen_length(largs)==1);
  expression e = EXPRESSION(CAR(largs));
  return freia_extract_kernel(e, strict, k00, k10, k20,
                              k01, k11, k21, k02, k12, k22);
}
