/*

  $Id$

  Copyright 1989-2010 MINES ParisTech
  Copyright 2010 HPC Project

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

#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "effects.h"
#include "effects-util.h"
#include "text-util.h"
#include "effects-simple.h"
#include "effects-generic.h"
#include "misc.h"

#include "pointer_values.h"

/**************** SPECIFIC INTRINSIC FUNCTIONS */

static void assignment_intrinsic_to_post_pv(entity func, list func_args, list l_in, pv_results * pv_res, pv_context *ctxt);

static void binary_arithmetic_operator_to_post_pv(entity func, list func_args, list l_in, pv_results * pv_res, pv_context *ctxt);
static void unary_arithmetic_operator_to_post_pv(entity func, list func_args, list l_in, pv_results * pv_res, pv_context *ctxt);
static void update_operator_to_post_pv(entity func, list func_args, list l_in, pv_results * pv_res, pv_context *ctxt);
static void logical_operator_to_post_pv(entity func, list func_args, list l_in, pv_results * pv_res, pv_context *ctxt);
static void conditional_operator_to_post_pv(entity func, list func_args, list l_in, pv_results * pv_res, pv_context *ctxt);


static void dereferencing_to_post_pv(entity func, list func_args, list l_in, pv_results * pv_res, pv_context *ctxt);
static void field_to_post_pv(entity func, list func_args, list l_in, pv_results * pv_res, pv_context *ctxt);
static void point_to_to_post_pv(entity func, list func_args, list l_in, pv_results * pv_res, pv_context *ctxt);
static void address_of_to_post_pv(entity func, list func_args, list l_in, pv_results * pv_res, pv_context *ctxt);

static void c_io_function_to_post_pv(entity func, list func_args, list l_in, pv_results * pv_res, pv_context *ctxt);
static void unix_io_function_to_post_pv(entity func, list func_args, list l_in, pv_results * pv_res, pv_context *ctxt);
static void string_function_to_post_pv(entity func, list func_args, list l_in, pv_results * pv_res, pv_context *ctxt);
static void va_list_function_to_post_pv(entity func, list func_args, list l_in, pv_results * pv_res, pv_context *ctxt);
static void heap_intrinsic_to_post_pv(entity func, list func_args, list l_in, pv_results * pv_res, pv_context *ctxt);
//static void stop_to_post_pv(entity func, list func_args, list l_in, pv_results * pv_res, pv_context *ctxt);
static void c_return_to_post_pv(entity func, list func_args, list l_in, pv_results * pv_res, pv_context *ctxt);

static void safe_intrinsic_to_post_pv(entity func, list func_args, list l_in, pv_results * pv_res, pv_context *ctxt);

static void intrinsic_to_identical_post_pv(entity func, list func_args, list l_in, pv_results * pv_res, pv_context *ctxt);

static void unknown_intrinsic_to_post_pv(entity func, list func_args, list l_in, pv_results * pv_res, pv_context *ctxt);

static void default_intrinsic_to_post_pv(entity func, list func_args, list l_in, pv_results * pv_res, pv_context *ctxt);

/**************** YET ANOTHER INTRINSICS TABLE */

/* the following data structure describes an intrinsic function: its
name and the function to apply on a call to this intrinsic to get the
post pointer values of the call */

/* These intrinsics are arranged in the order of the standard ISO/IEC 9899:TC2. MB */

typedef struct IntrinsicDescriptor
{
    string name;
    void (*to_post_pv_function)();
} IntrinsicToPostPVDescriptor;

static IntrinsicToPostPVDescriptor IntrinsicToPostPVDescriptorTable[] = {
  {PLUS_OPERATOR_NAME,                     binary_arithmetic_operator_to_post_pv},
  {MINUS_OPERATOR_NAME,                    binary_arithmetic_operator_to_post_pv},
  {DIVIDE_OPERATOR_NAME,                   binary_arithmetic_operator_to_post_pv},
  {MULTIPLY_OPERATOR_NAME,                 binary_arithmetic_operator_to_post_pv},

  {INVERSE_OPERATOR_NAME,                  unary_arithmetic_operator_to_post_pv},
  {UNARY_MINUS_OPERATOR_NAME,              unary_arithmetic_operator_to_post_pv},
  {POWER_OPERATOR_NAME,                    unary_arithmetic_operator_to_post_pv},

  {EQUIV_OPERATOR_NAME,                    logical_operator_to_post_pv},
  {NON_EQUIV_OPERATOR_NAME,                logical_operator_to_post_pv},
  {OR_OPERATOR_NAME,                       logical_operator_to_post_pv},
  {AND_OPERATOR_NAME,                      logical_operator_to_post_pv},
  {LESS_THAN_OPERATOR_NAME,                logical_operator_to_post_pv},
  {GREATER_THAN_OPERATOR_NAME,             logical_operator_to_post_pv},
  {LESS_OR_EQUAL_OPERATOR_NAME,            logical_operator_to_post_pv},
  {GREATER_OR_EQUAL_OPERATOR_NAME,         logical_operator_to_post_pv},
  {EQUAL_OPERATOR_NAME,                    logical_operator_to_post_pv},
  {NON_EQUAL_OPERATOR_NAME,                logical_operator_to_post_pv},
  {CONCATENATION_FUNCTION_NAME,            logical_operator_to_post_pv},
  {NOT_OPERATOR_NAME,                      logical_operator_to_post_pv},

  {CONTINUE_FUNCTION_NAME,                 intrinsic_to_identical_post_pv},
  {"ENDDO",                                unknown_intrinsic_to_post_pv},
  {PAUSE_FUNCTION_NAME,                    unknown_intrinsic_to_post_pv},
  {RETURN_FUNCTION_NAME,                   unknown_intrinsic_to_post_pv},
  {STOP_FUNCTION_NAME,                     unknown_intrinsic_to_post_pv},
  {END_FUNCTION_NAME,                      unknown_intrinsic_to_post_pv},
  {FORMAT_FUNCTION_NAME,                   unknown_intrinsic_to_post_pv},

  { IMPLIED_COMPLEX_NAME,                  unknown_intrinsic_to_post_pv},
  { IMPLIED_DCOMPLEX_NAME,                 unknown_intrinsic_to_post_pv},

  {INT_GENERIC_CONVERSION_NAME,            unknown_intrinsic_to_post_pv},
  {IFIX_GENERIC_CONVERSION_NAME,           unknown_intrinsic_to_post_pv},
  {IDINT_GENERIC_CONVERSION_NAME,          unknown_intrinsic_to_post_pv},
  {REAL_GENERIC_CONVERSION_NAME,           unknown_intrinsic_to_post_pv},
  {FLOAT_GENERIC_CONVERSION_NAME,          unknown_intrinsic_to_post_pv},
  {DFLOAT_GENERIC_CONVERSION_NAME,         unknown_intrinsic_to_post_pv},
  {SNGL_GENERIC_CONVERSION_NAME,           unknown_intrinsic_to_post_pv},
  {DBLE_GENERIC_CONVERSION_NAME,           unknown_intrinsic_to_post_pv},
  {DREAL_GENERIC_CONVERSION_NAME,          unknown_intrinsic_to_post_pv}, /* Added for Arnauld Leservot */
  {CMPLX_GENERIC_CONVERSION_NAME,          unknown_intrinsic_to_post_pv},
  {DCMPLX_GENERIC_CONVERSION_NAME,         unknown_intrinsic_to_post_pv},
  {INT_TO_CHAR_CONVERSION_NAME,            unknown_intrinsic_to_post_pv},
  {CHAR_TO_INT_CONVERSION_NAME,            unknown_intrinsic_to_post_pv},
  {AINT_CONVERSION_NAME,                   unknown_intrinsic_to_post_pv},
  {DINT_CONVERSION_NAME,                   unknown_intrinsic_to_post_pv},
  {ANINT_CONVERSION_NAME,                  unknown_intrinsic_to_post_pv},
  {DNINT_CONVERSION_NAME,                  unknown_intrinsic_to_post_pv},
  {NINT_CONVERSION_NAME,                   unknown_intrinsic_to_post_pv},
  {IDNINT_CONVERSION_NAME,                 unknown_intrinsic_to_post_pv},
  {IABS_OPERATOR_NAME,                     unknown_intrinsic_to_post_pv},
  {ABS_OPERATOR_NAME,                      unknown_intrinsic_to_post_pv},
  {DABS_OPERATOR_NAME,                     unknown_intrinsic_to_post_pv},
  {CABS_OPERATOR_NAME,                     unknown_intrinsic_to_post_pv},
  {CDABS_OPERATOR_NAME,                    unknown_intrinsic_to_post_pv},

  {MODULO_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {REAL_MODULO_OPERATOR_NAME,              intrinsic_to_identical_post_pv},
  {DOUBLE_MODULO_OPERATOR_NAME,            intrinsic_to_identical_post_pv},
  {ISIGN_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {SIGN_OPERATOR_NAME,                     intrinsic_to_identical_post_pv},
  {DSIGN_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {IDIM_OPERATOR_NAME,                     intrinsic_to_identical_post_pv},
  {DIM_OPERATOR_NAME,                      intrinsic_to_identical_post_pv},
  {DDIM_OPERATOR_NAME,                     intrinsic_to_identical_post_pv},
  {DPROD_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {MAX_OPERATOR_NAME,                      intrinsic_to_identical_post_pv},
  {MAX0_OPERATOR_NAME,                     intrinsic_to_identical_post_pv},
  {AMAX1_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {DMAX1_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {AMAX0_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {MAX1_OPERATOR_NAME,                     intrinsic_to_identical_post_pv},
  {MIN_OPERATOR_NAME,                      intrinsic_to_identical_post_pv},
  {MIN0_OPERATOR_NAME,                     intrinsic_to_identical_post_pv},
  {AMIN1_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {DMIN1_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {AMIN0_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {MIN1_OPERATOR_NAME,                     intrinsic_to_identical_post_pv},
  {LENGTH_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {INDEX_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {AIMAG_CONVERSION_NAME,                  intrinsic_to_identical_post_pv},
  {DIMAG_CONVERSION_NAME,                  intrinsic_to_identical_post_pv},
  {CONJG_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {DCONJG_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {SQRT_OPERATOR_NAME,                     intrinsic_to_identical_post_pv},
  {DSQRT_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {CSQRT_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},

  {EXP_OPERATOR_NAME,                      intrinsic_to_identical_post_pv},
  {DEXP_OPERATOR_NAME,                     intrinsic_to_identical_post_pv},
  {CEXP_OPERATOR_NAME,                     intrinsic_to_identical_post_pv},
  {LOG_OPERATOR_NAME,                      intrinsic_to_identical_post_pv},
  {ALOG_OPERATOR_NAME,                     intrinsic_to_identical_post_pv},
  {DLOG_OPERATOR_NAME,                     intrinsic_to_identical_post_pv},
  {CLOG_OPERATOR_NAME,                     intrinsic_to_identical_post_pv},
  {LOG10_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {ALOG10_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {DLOG10_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {SIN_OPERATOR_NAME,                      intrinsic_to_identical_post_pv},
  {DSIN_OPERATOR_NAME,                     intrinsic_to_identical_post_pv},
  {CSIN_OPERATOR_NAME,                     intrinsic_to_identical_post_pv},
  {COS_OPERATOR_NAME,                      intrinsic_to_identical_post_pv},
  {DCOS_OPERATOR_NAME,                     intrinsic_to_identical_post_pv},
  {CCOS_OPERATOR_NAME,                     intrinsic_to_identical_post_pv},
  {TAN_OPERATOR_NAME,                      intrinsic_to_identical_post_pv},
  {DTAN_OPERATOR_NAME,                     intrinsic_to_identical_post_pv},
  {ASIN_OPERATOR_NAME,                     intrinsic_to_identical_post_pv},
  {DASIN_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {ACOS_OPERATOR_NAME,                     intrinsic_to_identical_post_pv},
  {DACOS_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {ATAN_OPERATOR_NAME,                     intrinsic_to_identical_post_pv},
  {DATAN_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {ATAN2_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {DATAN2_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {SINH_OPERATOR_NAME,                     intrinsic_to_identical_post_pv},
  {DSINH_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {COSH_OPERATOR_NAME,                     intrinsic_to_identical_post_pv},
  {DCOSH_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {TANH_OPERATOR_NAME,                     intrinsic_to_identical_post_pv},
  {DTANH_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},

  {LGE_OPERATOR_NAME,                      intrinsic_to_identical_post_pv},
  {LGT_OPERATOR_NAME,                      intrinsic_to_identical_post_pv},
  {LLE_OPERATOR_NAME,                      intrinsic_to_identical_post_pv},
  {LLT_OPERATOR_NAME,                      intrinsic_to_identical_post_pv},

  {LIST_DIRECTED_FORMAT_NAME,              intrinsic_to_identical_post_pv},
  {UNBOUNDED_DIMENSION_NAME,               intrinsic_to_identical_post_pv},

  {ASSIGN_OPERATOR_NAME,                   assignment_intrinsic_to_post_pv},

  /* Fortran IO related intrinsic */
  {WRITE_FUNCTION_NAME,                    unknown_intrinsic_to_post_pv},
  {REWIND_FUNCTION_NAME,                   unknown_intrinsic_to_post_pv},
  {BACKSPACE_FUNCTION_NAME,                unknown_intrinsic_to_post_pv},
  {OPEN_FUNCTION_NAME,                     unknown_intrinsic_to_post_pv},
  {CLOSE_FUNCTION_NAME,                    unknown_intrinsic_to_post_pv},
  {INQUIRE_FUNCTION_NAME,                  unknown_intrinsic_to_post_pv},
  {READ_FUNCTION_NAME,                     unknown_intrinsic_to_post_pv},
  {BUFFERIN_FUNCTION_NAME,                 unknown_intrinsic_to_post_pv},
  {BUFFEROUT_FUNCTION_NAME,                unknown_intrinsic_to_post_pv},
  {ENDFILE_FUNCTION_NAME,                  unknown_intrinsic_to_post_pv},
  {IMPLIED_DO_NAME,                        unknown_intrinsic_to_post_pv},

  {SUBSTRING_FUNCTION_NAME,                unknown_intrinsic_to_post_pv},
  {ASSIGN_SUBSTRING_FUNCTION_NAME,         unknown_intrinsic_to_post_pv},

  /* These operators are used within the OPTIMIZE transformation in
     order to manipulate operators such as n-ary add and multiply or
     multiply-add operators ( JZ - sept 98) */
  {EOLE_SUM_OPERATOR_NAME,                 intrinsic_to_identical_post_pv },
  {EOLE_PROD_OPERATOR_NAME,                intrinsic_to_identical_post_pv },
  {EOLE_FMA_OPERATOR_NAME,                 intrinsic_to_identical_post_pv },

  {IMA_OPERATOR_NAME,                      intrinsic_to_identical_post_pv },
  {IMS_OPERATOR_NAME,                      intrinsic_to_identical_post_pv },


  /* Bit manipulation F90 functions. ISO/IEC 1539 : 1991  Amira Mensi */

  {ISHFT_OPERATOR_NAME,                    intrinsic_to_identical_post_pv },
  {ISHFTC_OPERATOR_NAME,                   intrinsic_to_identical_post_pv },
  {IBITS_OPERATOR_NAME,                    intrinsic_to_identical_post_pv },
  {MVBITS_OPERATOR_NAME,                   intrinsic_to_identical_post_pv },
  {BTEST_OPERATOR_NAME,                    intrinsic_to_identical_post_pv },
  {IBCLR_OPERATOR_NAME,                    intrinsic_to_identical_post_pv },
  {BIT_SIZE_OPERATOR_NAME,                 intrinsic_to_identical_post_pv },
  {IBSET_OPERATOR_NAME,                    intrinsic_to_identical_post_pv },
  {IAND_OPERATOR_NAME,                     intrinsic_to_identical_post_pv },
  {IEOR_OPERATOR_NAME,                     intrinsic_to_identical_post_pv },
  {IOR_OPERATOR_NAME,                      intrinsic_to_identical_post_pv },

  /* Here are C intrinsics.*/

  /* ISO 6.5.2.3 structure and union members */
  {FIELD_OPERATOR_NAME,                    field_to_post_pv},
  {POINT_TO_OPERATOR_NAME,                 point_to_to_post_pv},

  /* ISO 6.5.2.4 postfix increment and decrement operators, real or pointer type operand */
  {POST_INCREMENT_OPERATOR_NAME,           update_operator_to_post_pv},
  {POST_DECREMENT_OPERATOR_NAME,           update_operator_to_post_pv},

  /* ISO 6.5.3.1 prefix increment and decrement operators, real or pointer type operand */
  {PRE_INCREMENT_OPERATOR_NAME,            update_operator_to_post_pv},
  {PRE_DECREMENT_OPERATOR_NAME,            update_operator_to_post_pv},

  /* ISO 6.5.3.2 address and indirection operators, add pointer type */
  {ADDRESS_OF_OPERATOR_NAME,               address_of_to_post_pv},
  {DEREFERENCING_OPERATOR_NAME,            dereferencing_to_post_pv},

  /* ISO 6.5.3.3 unary arithmetic operators */
  {UNARY_PLUS_OPERATOR_NAME,               unary_arithmetic_operator_to_post_pv},
  // {"-unary",                            intrinsic_to_identical_post_pv},UNARY_MINUS_OPERATOR already exist (FORTRAN)
  {BITWISE_NOT_OPERATOR_NAME,              unary_arithmetic_operator_to_post_pv},
  {C_NOT_OPERATOR_NAME,                    logical_operator_to_post_pv},

  {C_MODULO_OPERATOR_NAME,                 intrinsic_to_identical_post_pv},

  /* ISO 6.5.6 additive operators, arithmetic types or pointer + integer type*/
  {PLUS_C_OPERATOR_NAME,                   binary_arithmetic_operator_to_post_pv},
  {MINUS_C_OPERATOR_NAME,                  binary_arithmetic_operator_to_post_pv},

  /* ISO 6.5.7 bitwise shift operators*/
  {LEFT_SHIFT_OPERATOR_NAME,               intrinsic_to_identical_post_pv},
  {RIGHT_SHIFT_OPERATOR_NAME,              intrinsic_to_identical_post_pv},

  /* ISO 6.5.8 relational operators,arithmetic or pointer types */
  {C_LESS_THAN_OPERATOR_NAME,              logical_operator_to_post_pv},
  {C_GREATER_THAN_OPERATOR_NAME,           logical_operator_to_post_pv},
  {C_LESS_OR_EQUAL_OPERATOR_NAME,          logical_operator_to_post_pv},
  {C_GREATER_OR_EQUAL_OPERATOR_NAME,       logical_operator_to_post_pv},

  /* ISO 6.5.9 equality operators, return 0 or 1*/
  {C_EQUAL_OPERATOR_NAME,                  logical_operator_to_post_pv},
  {C_NON_EQUAL_OPERATOR_NAME,              logical_operator_to_post_pv},

 /* ISO 6.5.10 bitwise AND operator */
  {BITWISE_AND_OPERATOR_NAME,              intrinsic_to_identical_post_pv},

 /* ISO 6.5.11 bitwise exclusive OR operator */
  {BITWISE_XOR_OPERATOR_NAME,              intrinsic_to_identical_post_pv},

  /* ISO 6.5.12 bitwise inclusive OR operator */
  {BITWISE_OR_OPERATOR_NAME,               intrinsic_to_identical_post_pv},

  /* ISO 6.5.13 logical AND operator */
  {C_AND_OPERATOR_NAME,                    logical_operator_to_post_pv},

  /* ISO 6.5.14 logical OR operator */
  {C_OR_OPERATOR_NAME,                     logical_operator_to_post_pv},

  /* ISO 6.5.15 conditional operator */
  {CONDITIONAL_OPERATOR_NAME,              conditional_operator_to_post_pv},

  /* ISO 6.5.16.2 compound assignments*/
  {MULTIPLY_UPDATE_OPERATOR_NAME,          update_operator_to_post_pv},
  {DIVIDE_UPDATE_OPERATOR_NAME,            update_operator_to_post_pv},
  {MODULO_UPDATE_OPERATOR_NAME,            update_operator_to_post_pv},
  {PLUS_UPDATE_OPERATOR_NAME,              update_operator_to_post_pv},
  {MINUS_UPDATE_OPERATOR_NAME,             update_operator_to_post_pv},
  {LEFT_SHIFT_UPDATE_OPERATOR_NAME,        update_operator_to_post_pv},
  {RIGHT_SHIFT_UPDATE_OPERATOR_NAME,       update_operator_to_post_pv},
  {BITWISE_AND_UPDATE_OPERATOR_NAME,       update_operator_to_post_pv},
  {BITWISE_XOR_UPDATE_OPERATOR_NAME,       update_operator_to_post_pv},
  {BITWISE_OR_UPDATE_OPERATOR_NAME,        update_operator_to_post_pv},

  /* ISO 6.5.17 comma operator */
  {COMMA_OPERATOR_NAME,                    default_intrinsic_to_post_pv},
  
  {BREAK_FUNCTION_NAME,                    intrinsic_to_identical_post_pv},
  {CASE_FUNCTION_NAME,                     intrinsic_to_identical_post_pv},
  {DEFAULT_FUNCTION_NAME,                  intrinsic_to_identical_post_pv},
  {C_RETURN_FUNCTION_NAME,                 c_return_to_post_pv},
 
  /* intrinsic to handle C initialization */

  {BRACE_INTRINSIC,                        intrinsic_to_identical_post_pv},


  /* assert.h */
  /* These intrinsics are added with intrinsic_to_identical_post_pv to work with C.
     The real effects on aliasing must be studied !!! I do not have time for the moment */

  {ASSERT_FUNCTION_NAME,                   intrinsic_to_identical_post_pv},
  {ASSERT_FAIL_FUNCTION_NAME,              intrinsic_to_identical_post_pv}, /* in fact, IO effect, does not return */
  
  /* #include <complex.h> */
  {CACOS_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {CACOSF_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {CACOSL_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {CASIN_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {CASINF_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {CASINL_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {CATAN_OPERATOR_NAME,                    intrinsic_to_identical_post_pv}, 
  {CATANF_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {CATANL_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {C_CCOS_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {CCOSF_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {CCOSL_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {C_CSIN_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {CSINF_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {CSINL_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {CTAN_OPERATOR_NAME,                     intrinsic_to_identical_post_pv},
  {CTANF_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {CTANL_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {CACOSH_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {CACOSHF_OPERATOR_NAME,                  intrinsic_to_identical_post_pv},
  {CACOSHL_OPERATOR_NAME,                  intrinsic_to_identical_post_pv},
  {CASINH_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {CASINHF_OPERATOR_NAME,                  intrinsic_to_identical_post_pv},
  {CASINHL_OPERATOR_NAME,                  intrinsic_to_identical_post_pv},
  {CATANH_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {CATANHF_OPERATOR_NAME,                  intrinsic_to_identical_post_pv},
  {CATANHL_OPERATOR_NAME,                  intrinsic_to_identical_post_pv},
  {CCOSH_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {CCOSHF_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {CCOSHL_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {CSINH_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {CSINHF_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {CSINHL_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {CTANH_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {CTANHF_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {CTANHL_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {C_CEXP_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {CEXPF_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {CEXPL_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {C_CLOG_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {CLOGF_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {CLOGL_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {C_CABS_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {CABSF_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {CABSL_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {CPOW_OPERATOR_NAME,                     intrinsic_to_identical_post_pv},
  {CPOWF_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {CPOWL_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {C_CSQRT_OPERATOR_NAME,                  intrinsic_to_identical_post_pv},
  {CSQRTF_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {CSQRTL_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {CARG_OPERATOR_NAME,                     intrinsic_to_identical_post_pv},
  {CARGF_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {CARGL_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {CIMAG_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {CIMAGF_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {CIMAGL_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {CONJ_OPERATOR_NAME,                     intrinsic_to_identical_post_pv},
  {CONJF_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {CONJL_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {CPROJ_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {CPROJF_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {CPROJL_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {CREAL_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {CREALF_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {CREALL_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},

  /* #include <ctype.h>*/

  {ISALNUM_OPERATOR_NAME,                  intrinsic_to_identical_post_pv},
  {ISALPHA_OPERATOR_NAME,                  intrinsic_to_identical_post_pv},
  {ISBLANK_OPERATOR_NAME,                  intrinsic_to_identical_post_pv},
  {ISCNTRL_OPERATOR_NAME,                  intrinsic_to_identical_post_pv},
  {ISDIGIT_OPERATOR_NAME,                  intrinsic_to_identical_post_pv},
  {ISGRAPH_OPERATOR_NAME,                  intrinsic_to_identical_post_pv},
  {ISLOWER_OPERATOR_NAME,                  intrinsic_to_identical_post_pv},
  {ISPRINT_OPERATOR_NAME,                  intrinsic_to_identical_post_pv},
  {ISPUNCT_OPERATOR_NAME,                  intrinsic_to_identical_post_pv},
  {ISSPACE_OPERATOR_NAME,                  intrinsic_to_identical_post_pv},
  {ISUPPER_OPERATOR_NAME,                  intrinsic_to_identical_post_pv},
  {ISXDIGIT_OPERATOR_NAME,                 intrinsic_to_identical_post_pv},
  {TOLOWER_OPERATOR_NAME,                  intrinsic_to_identical_post_pv},
  {TOUPPER_OPERATOR_NAME,                  intrinsic_to_identical_post_pv},
 
  /* errno.h */
  // MB: errno is usually an extern int variable, but *errno() is allowed (ISO section 7.5 in C99)  
  {"errno",                                intrinsic_to_identical_post_pv},

  /* fenv.h */

  {FECLEAREXCEPT_FUNCTION_NAME,            intrinsic_to_identical_post_pv},
  {FERAISEEXCEPT_FUNCTION_NAME,            intrinsic_to_identical_post_pv},
  {FESETEXCEPTFLAG_FUNCTION_NAME,          intrinsic_to_identical_post_pv},
  {FETESTEXCEPT_FUNCTION_NAME,             intrinsic_to_identical_post_pv},
  {FEGETROUND_FUNCTION_NAME,               intrinsic_to_identical_post_pv},
  {FESETROUND_FUNCTION_NAME,               intrinsic_to_identical_post_pv},
  // fenv_t *
  // {FESETENV_FUNCTION_NAME,                 intrinsic_to_identical_post_pv},
  //{FEUPDATEENV_FUNCTION_NAME,              intrinsic_to_identical_post_pv},

  
  /* inttypes.h */
  {IMAXABS_FUNCTION_NAME,                  intrinsic_to_identical_post_pv},
  {IMAXDIV_FUNCTION_NAME,                  intrinsic_to_identical_post_pv},

  /* locale.h */
  {SETLOCALE_FUNCTION_NAME,                intrinsic_to_identical_post_pv},
  {"localeconv",                           intrinsic_to_identical_post_pv},

  /* #include <math.h>*/

  {FPCLASSIFY_OPERATOR_NAME,               intrinsic_to_identical_post_pv},
  {ISFINITE_OPERATOR_NAME,                 intrinsic_to_identical_post_pv},
  {ISINF_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {ISNAN_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {ISNANL_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {ISNANF_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {ISNORMAL_OPERATOR_NAME,                 intrinsic_to_identical_post_pv},
  {SIGNBIT_OPERATOR_NAME,                  intrinsic_to_identical_post_pv},
  {C_ACOS_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {ACOSF_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {ACOSL_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {C_ASIN_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {ASINF_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {ASINL_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {C_ATAN_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {ATANF_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {ATANL_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {C_ATAN2_OPERATOR_NAME,                  intrinsic_to_identical_post_pv},
  {ATAN2F_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {ATAN2L_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {C_COS_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {COSF_OPERATOR_NAME,                     intrinsic_to_identical_post_pv},
  {COSL_OPERATOR_NAME,                     intrinsic_to_identical_post_pv},
  {C_SIN_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {SINF_OPERATOR_NAME,                     intrinsic_to_identical_post_pv},
  {SINL_OPERATOR_NAME,                     intrinsic_to_identical_post_pv},
  {C_TAN_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {TANF_OPERATOR_NAME,                     intrinsic_to_identical_post_pv},
  {TANL_OPERATOR_NAME,                     intrinsic_to_identical_post_pv},
  {C_ACOSH_OPERATOR_NAME ,                 intrinsic_to_identical_post_pv},
  {ACOSHF_OPERATOR_NAME ,                  intrinsic_to_identical_post_pv},
  {ACOSHL_OPERATOR_NAME ,                  intrinsic_to_identical_post_pv},
  {C_ASINH_OPERATOR_NAME,                  intrinsic_to_identical_post_pv},
  {ASINHF_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {ASINHL_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {C_ATANH_OPERATOR_NAME,                  intrinsic_to_identical_post_pv},
  {ATANHF_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {ATANHL_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {C_COSH_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {COSHF_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {COSHL_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {C_SINH_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {SINHF_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {SINHL_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {C_TANH_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {TANHF_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {TANHL_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {C_EXP_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {EXPF_OPERATOR_NAME,                     intrinsic_to_identical_post_pv},
  {EXPL_OPERATOR_NAME,                     intrinsic_to_identical_post_pv},
  {EXP2_OPERATOR_NAME,                     intrinsic_to_identical_post_pv},
  {EXP2F_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {EXP2L_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {EXPM1_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {EXPM1F_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {EXPM1L_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  //frexp has a write effect not defined correctly. MB
  {FREXP_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {ILOGB_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {ILOGBF_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {ILOGBL_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {LDEXP_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {LDEXPF_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {LDEXPL_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {C_LOG_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {LOGF_OPERATOR_NAME,                     intrinsic_to_identical_post_pv},
  {LOGL_OPERATOR_NAME,                     intrinsic_to_identical_post_pv},
  {C_LOG10_OPERATOR_NAME,                  intrinsic_to_identical_post_pv},
  {LOG10F_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {LOG10L_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {LOG1P_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {LOG1PF_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {LOG1PL_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {LOG2_OPERATOR_NAME,                     intrinsic_to_identical_post_pv},
  {LOG2F_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {LOG2L_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {LOGB_OPERATOR_NAME,                     intrinsic_to_identical_post_pv},
  {LOGBF_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {LOGBL_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  //modf & modff have write effects not defined correctly. MB
  {MODF_OPERATOR_NAME,                     intrinsic_to_identical_post_pv},
  {MODFF_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {SCALBN_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {SCALBNF_OPERATOR_NAME,                  intrinsic_to_identical_post_pv},
  {SCALBNL_OPERATOR_NAME,                  intrinsic_to_identical_post_pv},
  {SCALB_OPERATOR_NAME,                    intrinsic_to_identical_post_pv}, /* POSIX.1-2001, The scalb function is the BSD name for ldexp */
  {SCALBLN_OPERATOR_NAME,                  intrinsic_to_identical_post_pv},
  {SCALBLNF_OPERATOR_NAME,                 intrinsic_to_identical_post_pv},
  {SCALBLNL_OPERATOR_NAME,                 intrinsic_to_identical_post_pv},
  {CBRT_OPERATOR_NAME,                     intrinsic_to_identical_post_pv},
  {CBRTF_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {CBRTL_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {FABS_OPERATOR_NAME,                     intrinsic_to_identical_post_pv},
  {FABSF_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {FABSL_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {HYPOT_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {HYPOTF_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {HYPOTL_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {POW_OPERATOR_NAME,                      intrinsic_to_identical_post_pv},
  {POWF_OPERATOR_NAME,                     intrinsic_to_identical_post_pv},
  {POWL_OPERATOR_NAME,                     intrinsic_to_identical_post_pv},
  {C_SQRT_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {SQRTF_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {SQRTL_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {ERF_OPERATOR_NAME,                      intrinsic_to_identical_post_pv},
  {ERFF_OPERATOR_NAME,                     intrinsic_to_identical_post_pv},
  {ERFL_OPERATOR_NAME,                     intrinsic_to_identical_post_pv},
  {ERFC_OPERATOR_NAME,                     intrinsic_to_identical_post_pv},
  {ERFCF_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {ERFCL_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {GAMMA_OPERATOR_NAME,                    intrinsic_to_identical_post_pv}, /* GNU C Library */
  {LGAMMA_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {LGAMMAF_OPERATOR_NAME,                  intrinsic_to_identical_post_pv},
  {LGAMMAL_OPERATOR_NAME,                  intrinsic_to_identical_post_pv},
  {TGAMMA_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {TGAMMAF_OPERATOR_NAME,                  intrinsic_to_identical_post_pv},
  {TGAMMAL_OPERATOR_NAME,                  intrinsic_to_identical_post_pv},
  {CEIL_OPERATOR_NAME,                     intrinsic_to_identical_post_pv},
  {CEILF_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {CEILL_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {FLOOR_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {FLOORF_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {FLOORL_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {NEARBYINT_OPERATOR_NAME,                intrinsic_to_identical_post_pv},
  {NEARBYINTF_OPERATOR_NAME,               intrinsic_to_identical_post_pv},
  {NEARBYINTL_OPERATOR_NAME,               intrinsic_to_identical_post_pv},
  {RINT_OPERATOR_NAME,                     intrinsic_to_identical_post_pv},
  {RINTF_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {RINTL_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {LRINT_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {LRINTF_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {LRINTL_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {LLRINT_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {LLRINTF_OPERATOR_NAME,                  intrinsic_to_identical_post_pv},
  {LLRINTL_OPERATOR_NAME,                  intrinsic_to_identical_post_pv},
  {ROUND_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {ROUNDF_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {ROUNDL_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {LROUND_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {LROUNDF_OPERATOR_NAME,                  intrinsic_to_identical_post_pv},
  {LROUNDL_OPERATOR_NAME,                  intrinsic_to_identical_post_pv},
  {LLROUND_OPERATOR_NAME,                  intrinsic_to_identical_post_pv},
  {LLROUNDF_OPERATOR_NAME,                 intrinsic_to_identical_post_pv},
  {LLROUNDL_OPERATOR_NAME,                 intrinsic_to_identical_post_pv},
  {TRUNC_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {TRUNCF_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {TRUNCL_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {FMOD_OPERATOR_NAME,                     intrinsic_to_identical_post_pv},
  {FMODF_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {FMODL_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {REMAINDER_OPERATOR_NAME,                intrinsic_to_identical_post_pv},
  {REMAINDERF_OPERATOR_NAME,               intrinsic_to_identical_post_pv},
  {REMAINDERL_OPERATOR_NAME,               intrinsic_to_identical_post_pv},
  {COPYSIGN_OPERATOR_NAME,                 intrinsic_to_identical_post_pv},
  {COPYSIGNF_OPERATOR_NAME,                intrinsic_to_identical_post_pv},
  {COPYSIGNL_OPERATOR_NAME,                intrinsic_to_identical_post_pv},
  {NAN_OPERATOR_NAME,                      intrinsic_to_identical_post_pv},
  {NANF_OPERATOR_NAME,                     intrinsic_to_identical_post_pv},
  {NANL_OPERATOR_NAME,                     intrinsic_to_identical_post_pv},
  {NEXTAFTER_OPERATOR_NAME,                intrinsic_to_identical_post_pv},
  {NEXTAFTERF_OPERATOR_NAME,               intrinsic_to_identical_post_pv},
  {NEXTAFTERL_OPERATOR_NAME,               intrinsic_to_identical_post_pv},
  {NEXTTOWARD_OPERATOR_NAME,               intrinsic_to_identical_post_pv},
  {NEXTTOWARDF_OPERATOR_NAME,              intrinsic_to_identical_post_pv},
  {NEXTTOWARDL_OPERATOR_NAME,              intrinsic_to_identical_post_pv},
  {FDIM_OPERATOR_NAME,                     intrinsic_to_identical_post_pv},
  {FDIMF_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {FDIML_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {FMAX_OPERATOR_NAME,                     intrinsic_to_identical_post_pv},
  {FMAXF_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {FMAXL_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {FMIN_OPERATOR_NAME,                     intrinsic_to_identical_post_pv},
  {FMINF_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {FMINL_OPERATOR_NAME,                    intrinsic_to_identical_post_pv},
  {FMA_OPERATOR_NAME,                      intrinsic_to_identical_post_pv},
  {FMAF_OPERATOR_NAME,                     intrinsic_to_identical_post_pv},
  {FMAL_OPERATOR_NAME,                     intrinsic_to_identical_post_pv},
  {ISGREATER_OPERATOR_NAME,                intrinsic_to_identical_post_pv},
  {ISGREATEREQUAL_OPERATOR_NAME,           intrinsic_to_identical_post_pv},
  {ISLESS_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {ISLESSEQUAL_OPERATOR_NAME,              intrinsic_to_identical_post_pv},
  {ISLESSGREATER_OPERATOR_NAME,            intrinsic_to_identical_post_pv},
  {ISUNORDERED_OPERATOR_NAME,              intrinsic_to_identical_post_pv},


  /*#include <setjmp.h>*/

  {"setjmp",                               intrinsic_to_identical_post_pv},
  {"__setjmp",                             intrinsic_to_identical_post_pv},
  {"longjmp",                              intrinsic_to_identical_post_pv}, // control effect 7.13 in C99
  {"__longjmp",                            intrinsic_to_identical_post_pv},
  {"sigsetjmp",                            intrinsic_to_identical_post_pv}, //POSIX.1-2001
  {"siglongjmp",                           intrinsic_to_identical_post_pv}, //POSIX.1-2001


  /* signal.h 7.14 */
  {SIGFPE_OPERATOR_NAME,                   intrinsic_to_identical_post_pv}, //macro
  {SIGNAL_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {RAISE_FUNCTION_NAME,                    intrinsic_to_identical_post_pv},


  /* stdarg.h */

  {BUILTIN_VA_START,                       va_list_function_to_post_pv},
  {BUILTIN_VA_END,                         va_list_function_to_post_pv},
  {BUILTIN_VA_COPY,                        va_list_function_to_post_pv},
  /* va_arg is not a standard call; it is directly represented in PIPS
     internal representation. */

  /*#include <stdio.h>*/
  // IO functions
  {REMOVE_FUNCTION_NAME,                   intrinsic_to_identical_post_pv},
  {RENAME_FUNCTION_NAME,                   intrinsic_to_identical_post_pv},
  {TMPFILE_FUNCTION_NAME,                  intrinsic_to_identical_post_pv},
  {TMPNAM_FUNCTION_NAME,                   intrinsic_to_identical_post_pv},
  {FCLOSE_FUNCTION_NAME,                   c_io_function_to_post_pv},
  {FFLUSH_FUNCTION_NAME,                   intrinsic_to_identical_post_pv},
  {FOPEN_FUNCTION_NAME,                    c_io_function_to_post_pv},
  {FREOPEN_FUNCTION_NAME,                  intrinsic_to_identical_post_pv},
  {SETBUF_FUNCTION_NAME,                   intrinsic_to_identical_post_pv},
  {SETVBUF_FUNCTION_NAME ,                 intrinsic_to_identical_post_pv},
  {FPRINTF_FUNCTION_NAME,                  c_io_function_to_post_pv},
  {FSCANF_FUNCTION_NAME,                   c_io_function_to_post_pv},
  {ISOC99_FSCANF_FUNCTION_NAME,            c_io_function_to_post_pv},
  {PRINTF_FUNCTION_NAME,                   c_io_function_to_post_pv},
  {SCANF_FUNCTION_NAME,                    c_io_function_to_post_pv},
  {ISOC99_SCANF_FUNCTION_NAME,             c_io_function_to_post_pv},
  {SNPRINTF_FUNCTION_NAME,                 c_io_function_to_post_pv},
  {SPRINTF_FUNCTION_NAME,                  c_io_function_to_post_pv},
  {SSCANF_FUNCTION_NAME,                   c_io_function_to_post_pv},
  {ISOC99_SSCANF_FUNCTION_NAME,            c_io_function_to_post_pv},
  {VFPRINTF_FUNCTION_NAME,                 c_io_function_to_post_pv},
  {VFSCANF_FUNCTION_NAME,                  c_io_function_to_post_pv},
  {ISOC99_VFSCANF_FUNCTION_NAME,           c_io_function_to_post_pv},
  {VPRINTF_FUNCTION_NAME,                  c_io_function_to_post_pv},
  {VSCANF_FUNCTION_NAME,                   c_io_function_to_post_pv},
  {ISOC99_VSCANF_FUNCTION_NAME,            c_io_function_to_post_pv},
  {VSNPRINTF_FUNCTION_NAME,                c_io_function_to_post_pv},
  {VSPRINTF_FUNCTION_NAME,                 c_io_function_to_post_pv},
  {VSSCANF_FUNCTION_NAME,                  c_io_function_to_post_pv},
  {ISOC99_VSSCANF_FUNCTION_NAME,           c_io_function_to_post_pv},
  {FGETC_FUNCTION_NAME,                    c_io_function_to_post_pv},
  {FGETS_FUNCTION_NAME,                    c_io_function_to_post_pv},
  {FPUTC_FUNCTION_NAME,                    c_io_function_to_post_pv},
  {FPUTS_FUNCTION_NAME,                    c_io_function_to_post_pv},
  {GETC_FUNCTION_NAME,                     c_io_function_to_post_pv},
  {_IO_GETC_FUNCTION_NAME,                 c_io_function_to_post_pv},
  {GETCHAR_FUNCTION_NAME,                  c_io_function_to_post_pv},
  {GETS_FUNCTION_NAME,                     c_io_function_to_post_pv},
  {PUTC_FUNCTION_NAME,                     c_io_function_to_post_pv},
  {_IO_PUTC_FUNCTION_NAME,                 c_io_function_to_post_pv},
  {PUTCHAR_FUNCTION_NAME,                  c_io_function_to_post_pv},
  {PUTS_FUNCTION_NAME,                     c_io_function_to_post_pv},
  {UNGETC_FUNCTION_NAME,                   c_io_function_to_post_pv},
  {FREAD_FUNCTION_NAME,                    c_io_function_to_post_pv},
  {FWRITE_FUNCTION_NAME,                   c_io_function_to_post_pv},
  {FGETPOS_FUNCTION_NAME,                  c_io_function_to_post_pv},
  {FSEEK_FUNCTION_NAME,                    c_io_function_to_post_pv},
  {FSETPOS_FUNCTION_NAME,                  c_io_function_to_post_pv},
  {FTELL_FUNCTION_NAME,                    c_io_function_to_post_pv},
  {C_REWIND_FUNCTION_NAME,                 c_io_function_to_post_pv},
  {CLEARERR_FUNCTION_NAME,                 c_io_function_to_post_pv},
  {FEOF_FUNCTION_NAME,                     c_io_function_to_post_pv},
  {FERROR_FUNCTION_NAME,                   c_io_function_to_post_pv},
  {PERROR_FUNCTION_NAME,                   c_io_function_to_post_pv},


  /* #include <stdlib.h> */
  {ATOF_FUNCTION_NAME,                     intrinsic_to_identical_post_pv},
  {ATOI_FUNCTION_NAME,                     intrinsic_to_identical_post_pv},
  {ATOL_FUNCTION_NAME,                     intrinsic_to_identical_post_pv},
  {ATOLL_FUNCTION_NAME,                    intrinsic_to_identical_post_pv},
  {STRTOD_FUNCTION_NAME,                   intrinsic_to_identical_post_pv},
  {STRTOF_FUNCTION_NAME,                   intrinsic_to_identical_post_pv},
  {STRTOL_FUNCTION_NAME,                   safe_intrinsic_to_post_pv},
  {STRTOLL_FUNCTION_NAME,                  safe_intrinsic_to_post_pv},
  {STRTOUL_FUNCTION_NAME,                  safe_intrinsic_to_post_pv},
  {STRTOULL_FUNCTION_NAME,                 safe_intrinsic_to_post_pv},
  {RAND_FUNCTION_NAME,                     intrinsic_to_identical_post_pv},
  {SRAND_FUNCTION_NAME,                    intrinsic_to_identical_post_pv},
  {CALLOC_FUNCTION_NAME,                   heap_intrinsic_to_post_pv},
  {FREE_FUNCTION_NAME,                     heap_intrinsic_to_post_pv},
  {MALLOC_FUNCTION_NAME,                   heap_intrinsic_to_post_pv},
  {REALLOC_FUNCTION_NAME,                  heap_intrinsic_to_post_pv},
  /* SG: I am setting an any_heap_effects for alloca, which is over pessimistic ... */
  {ALLOCA_FUNCTION_NAME,                   heap_intrinsic_to_post_pv},
  {ABORT_FUNCTION_NAME,                    intrinsic_to_identical_post_pv},
  {ATEXIT_FUNCTION_NAME,                   intrinsic_to_identical_post_pv},
  {EXIT_FUNCTION_NAME,                     intrinsic_to_identical_post_pv},
  {_EXIT_FUNCTION_NAME,                    intrinsic_to_identical_post_pv},
  {GETENV_FUNCTION_NAME,                   intrinsic_to_identical_post_pv},
  {SYSTEM_FUNCTION_NAME,                   intrinsic_to_identical_post_pv},
  {BSEARCH_FUNCTION_NAME,                  intrinsic_to_identical_post_pv},
  {QSORT_FUNCTION_NAME,                    intrinsic_to_identical_post_pv},
  {C_ABS_FUNCTION_NAME,                    intrinsic_to_identical_post_pv},
  {LABS_FUNCTION_NAME,                     intrinsic_to_identical_post_pv},
  {LLABS_FUNCTION_NAME,                    intrinsic_to_identical_post_pv},
  {DIV_FUNCTION_NAME,                      intrinsic_to_identical_post_pv},
  {LDIV_FUNCTION_NAME,                     intrinsic_to_identical_post_pv},
  {LLDIV_FUNCTION_NAME,                    intrinsic_to_identical_post_pv},
  {MBLEN_FUNCTION_NAME,                    intrinsic_to_identical_post_pv},
  {MBTOWC_FUNCTION_NAME,                   intrinsic_to_identical_post_pv},
  {WCTOMB_FUNCTION_NAME,                   intrinsic_to_identical_post_pv},
  {MBSTOWCS_FUNCTION_NAME,                 intrinsic_to_identical_post_pv},
  {WCSTOMBS_FUNCTION_NAME,                 intrinsic_to_identical_post_pv},

   /*#include <string.h>*/

  {MEMCPY_FUNCTION_NAME,                   string_function_to_post_pv},
  {MEMMOVE_FUNCTION_NAME,                  string_function_to_post_pv},
  {STRCPY_FUNCTION_NAME,                   string_function_to_post_pv},
  {STRNCPY_FUNCTION_NAME,                  string_function_to_post_pv},
  {STRCAT_FUNCTION_NAME,                   string_function_to_post_pv},
  {STRNCAT_FUNCTION_NAME,                  string_function_to_post_pv},
  {MEMCMP_FUNCTION_NAME,                   intrinsic_to_identical_post_pv},
  {STRCMP_FUNCTION_NAME,                   intrinsic_to_identical_post_pv},
  {STRCOLL_FUNCTION_NAME,                  intrinsic_to_identical_post_pv},
  {STRDUP_FUNCTION_NAME,                   heap_intrinsic_to_post_pv},
  {STRNCMP_FUNCTION_NAME,                  intrinsic_to_identical_post_pv},
  {STRXFRM_FUNCTION_NAME,                  string_function_to_post_pv},
  {MEMCHR_FUNCTION_NAME,                   intrinsic_to_identical_post_pv},
  {STRCHR_FUNCTION_NAME,                   intrinsic_to_identical_post_pv},
  {STRCSPN_FUNCTION_NAME,                  intrinsic_to_identical_post_pv},
  {STRPBRK_FUNCTION_NAME,                  intrinsic_to_identical_post_pv},
  {STRRCHR_FUNCTION_NAME,                  intrinsic_to_identical_post_pv},
  {STRSPN_FUNCTION_NAME,                   intrinsic_to_identical_post_pv},
  {STRSTR_FUNCTION_NAME,                   intrinsic_to_identical_post_pv},
  {STRTOK_FUNCTION_NAME,                   intrinsic_to_identical_post_pv},
  {MEMSET_FUNCTION_NAME,                   string_function_to_post_pv},
  {STRERROR_FUNCTION_NAME,                 intrinsic_to_identical_post_pv},
  {STRERROR_R_FUNCTION_NAME,               intrinsic_to_identical_post_pv},
  {STRLEN_FUNCTION_NAME,                   intrinsic_to_identical_post_pv},

  /*#include <time.h>*/
  {TIME_FUNCTION_NAME,                     intrinsic_to_identical_post_pv},
  {LOCALTIME_FUNCTION_NAME,                intrinsic_to_identical_post_pv},
  {DIFFTIME_FUNCTION_NAME,                 intrinsic_to_identical_post_pv},
  {GETTIMEOFDAY_FUNCTION_NAME,             intrinsic_to_identical_post_pv},
  {CLOCK_GETTIME_FUNCTION_NAME,            intrinsic_to_identical_post_pv},
  {CLOCK_FUNCTION_NAME,                    intrinsic_to_identical_post_pv},
  {SECOND_FUNCTION_NAME,                   intrinsic_to_identical_post_pv}, // gfortran intrinsic

  /*#include <wchar.h>*/
  {FWSCANF_FUNCTION_NAME,                  c_io_function_to_post_pv},
  {SWSCANF_FUNCTION_NAME,                  c_io_function_to_post_pv},
  {WSCANF_FUNCTION_NAME,                   c_io_function_to_post_pv},

  /* #include <wctype.h> */
  {ISWALNUM_OPERATOR_NAME,                 intrinsic_to_identical_post_pv},
  {ISWALPHA_OPERATOR_NAME,                 intrinsic_to_identical_post_pv},
  {ISWBLANK_OPERATOR_NAME,                 intrinsic_to_identical_post_pv},
  {ISWCNTRL_OPERATOR_NAME,                 intrinsic_to_identical_post_pv},
  {ISWDIGIT_OPERATOR_NAME,                 intrinsic_to_identical_post_pv},
  {ISWGRAPH_OPERATOR_NAME,                 intrinsic_to_identical_post_pv},
  {ISWLOWER_OPERATOR_NAME,                 intrinsic_to_identical_post_pv},
  {ISWPRINT_OPERATOR_NAME,                 intrinsic_to_identical_post_pv},
  {ISWPUNCT_OPERATOR_NAME,                 intrinsic_to_identical_post_pv},
  {ISWSPACE_OPERATOR_NAME,                 intrinsic_to_identical_post_pv},
  {ISWUPPER_OPERATOR_NAME,                 intrinsic_to_identical_post_pv},
  {ISWXDIGIT_OPERATOR_NAME,                intrinsic_to_identical_post_pv},
  {ISWCTYPE_OPERATOR_NAME,                 intrinsic_to_identical_post_pv},
  {WCTYPE_OPERATOR_NAME,                   intrinsic_to_identical_post_pv},
  {TOWLOWER_OPERATOR_NAME,                 intrinsic_to_identical_post_pv},
  {TOWUPPER_OPERATOR_NAME,                 intrinsic_to_identical_post_pv},
  {TOWCTRANS_OPERATOR_NAME,                intrinsic_to_identical_post_pv},
  {WCTRANS_OPERATOR_NAME,                  intrinsic_to_identical_post_pv},



  //not found in standard C99 (in GNU C Library)
  {ISASCII_OPERATOR_NAME,                  intrinsic_to_identical_post_pv}, //This function is a BSD extension and is also an SVID extension.
  {TOASCII_OPERATOR_NAME,                  intrinsic_to_identical_post_pv}, //This function is a BSD extension and is also an SVID extension.
  {_TOLOWER_OPERATOR_NAME,                 intrinsic_to_identical_post_pv}, //This function is provided for compatibility with the SVID
  {_TOUPPER_OPERATOR_NAME,                 intrinsic_to_identical_post_pv}, //This function is provided for compatibility with the SVID

  /* Part of the binary standard */
  {CTYPE_B_LOC_OPERATOR_NAME,              intrinsic_to_identical_post_pv},



  {"__flt_rounds",                         intrinsic_to_identical_post_pv},

  {"_sysconf",                             intrinsic_to_identical_post_pv},
  {"wdinit",                               intrinsic_to_identical_post_pv},
  {"wdchkind",                             intrinsic_to_identical_post_pv},
  {"wdbindf",                              intrinsic_to_identical_post_pv},
  {"wddelim",                              intrinsic_to_identical_post_pv},
  {"mcfiller",                             intrinsic_to_identical_post_pv},
  {"mcwrap",                               intrinsic_to_identical_post_pv},
 
  //GNU C Library
  {"dcgettext",                            intrinsic_to_identical_post_pv},
  {"dgettext",                             intrinsic_to_identical_post_pv},
  {"gettext",                              intrinsic_to_identical_post_pv},
  {"textdomain",                           intrinsic_to_identical_post_pv},
  {"bindtextdomain",                       intrinsic_to_identical_post_pv},


  /* not found in C99 standard (in GNU C Library) */

  {J0_OPERATOR_NAME,                       intrinsic_to_identical_post_pv},
  {J1_OPERATOR_NAME,                       intrinsic_to_identical_post_pv},
  {JN_OPERATOR_NAME,                       intrinsic_to_identical_post_pv},
  {Y0_OPERATOR_NAME,                       intrinsic_to_identical_post_pv},
  {Y1_OPERATOR_NAME,                       intrinsic_to_identical_post_pv},
  {YN_OPERATOR_NAME,                       intrinsic_to_identical_post_pv},

  //In the System V math library
  {MATHERR_OPERATOR_NAME,                  intrinsic_to_identical_post_pv},

  //This function exists mainly for use in certain standardized tests of IEEE 754 conformance.
  {SIGNIFICAND_OPERATOR_NAME,              intrinsic_to_identical_post_pv},

  /* netdb.h not in C99 standard (in GNU library) */
  {__H_ERRNO_LOCATION_OPERATOR_NAME,       intrinsic_to_identical_post_pv},

  /* bits/errno.h */
  {__ERRNO_LOCATION_OPERATOR_NAME,         intrinsic_to_identical_post_pv},


  //Posix LEGACY Std 1003.1
  {ECVT_FUNCTION_NAME,                     intrinsic_to_identical_post_pv},
  {FCVT_FUNCTION_NAME,                     intrinsic_to_identical_post_pv},
  {GCVT_FUNCTION_NAME,                     intrinsic_to_identical_post_pv},


  /* Random number generators in stdlib.h Conforming to SVr4, POSIX.1-2001 but not in C99 */

  {RANDOM_FUNCTION_NAME,                   intrinsic_to_identical_post_pv},
  {SRANDOM_FUNCTION_NAME,                  intrinsic_to_identical_post_pv},
  {DRAND48_FUNCTION_NAME,                  intrinsic_to_identical_post_pv},
  {ERAND48_FUNCTION_NAME,                  intrinsic_to_identical_post_pv},
  {JRAND48_FUNCTION_NAME,                  intrinsic_to_identical_post_pv},
  {LRAND48_FUNCTION_NAME,                  intrinsic_to_identical_post_pv},
  {MRAND48_FUNCTION_NAME,                  intrinsic_to_identical_post_pv},
  {NRAND48_FUNCTION_NAME,                  intrinsic_to_identical_post_pv},
  {SRAND48_FUNCTION_NAME,                  intrinsic_to_identical_post_pv},
  {SEED48_FUNCTION_NAME,                   intrinsic_to_identical_post_pv},
  {LCONG48_FUNCTION_NAME,                  intrinsic_to_identical_post_pv},


  //Posix
  {POSIX_MEMALIGN_FUNCTION_NAME,           intrinsic_to_identical_post_pv},
  {ATOQ_FUNCTION_NAME,                     intrinsic_to_identical_post_pv},
  {LLTOSTR_FUNCTION_NAME,                  safe_intrinsic_to_post_pv},
  {ULLTOSTR_FUNCTION_NAME,                 intrinsic_to_identical_post_pv},

  //MB: not found in C99 standard. POSIX.2
  {__FILBUF_FUNCTION_NAME,                 intrinsic_to_identical_post_pv},
  {__FILSBUF_FUNCTION_NAME,                intrinsic_to_identical_post_pv},
  {SETBUFFER_FUNCTION_NAME,                intrinsic_to_identical_post_pv},
  {SETLINEBUF_FUNCTION_NAME,               intrinsic_to_identical_post_pv},
  {FDOPEN_FUNCTION_NAME,                   intrinsic_to_identical_post_pv},
  {CTERMID_FUNCTION_NAME,                  intrinsic_to_identical_post_pv},
  {FILENO_FUNCTION_NAME,                   intrinsic_to_identical_post_pv},
  {POPEN_FUNCTION_NAME,                    intrinsic_to_identical_post_pv},
  {CUSERID_FUNCTION_NAME,                  intrinsic_to_identical_post_pv},
  {TEMPNAM_FUNCTION_NAME,                  intrinsic_to_identical_post_pv},
  {GETOPT_FUNCTION_NAME,                   intrinsic_to_identical_post_pv},
  {GETSUBOPT_FUNCTION_NAME,                intrinsic_to_identical_post_pv},
  {GETW_FUNCTION_NAME,                     intrinsic_to_identical_post_pv},
  {PUTW_FUNCTION_NAME,                     intrinsic_to_identical_post_pv},
  {PCLOSE_FUNCTION_NAME,                   intrinsic_to_identical_post_pv},
  {FSEEKO_FUNCTION_NAME,                   intrinsic_to_identical_post_pv},
  {FTELLO_FUNCTION_NAME,                   intrinsic_to_identical_post_pv},
  {FOPEN64_FUNCTION_NAME,                  intrinsic_to_identical_post_pv},
  {FREOPEN64_FUNCTION_NAME,                intrinsic_to_identical_post_pv},
  {TMPFILE64_FUNCTION_NAME,                intrinsic_to_identical_post_pv},
  {FGETPOS64_FUNCTION_NAME,                intrinsic_to_identical_post_pv},
  {FSETPOS64_FUNCTION_NAME,                intrinsic_to_identical_post_pv},
  {FSEEKO64_FUNCTION_NAME,                 intrinsic_to_identical_post_pv},
  {FTELLO64_FUNCTION_NAME,                 intrinsic_to_identical_post_pv},

  //MB: not found in C99
  {SINGLE_TO_DECIMAL_OPERATOR_NAME,        intrinsic_to_identical_post_pv},
  {DOUBLE_TO_DECIMAL_OPERATOR_NAME,        intrinsic_to_identical_post_pv},
  {EXTENDED_TO_DECIMAL_OPERATOR_NAME,      intrinsic_to_identical_post_pv},
  {QUADRUPLE_TO_DECIMAL_OPERATOR_NAME,     intrinsic_to_identical_post_pv},
  {DECIMAL_TO_SINGLE_OPERATOR_NAME,        intrinsic_to_identical_post_pv},
  {DECIMAL_TO_DOUBLE_OPERATOR_NAME,        intrinsic_to_identical_post_pv},
  {DECIMAL_TO_EXTENDED_OPERATOR_NAME,      intrinsic_to_identical_post_pv},
  {DECIMAL_TO_QUADRUPLE_OPERATOR_NAME,     intrinsic_to_identical_post_pv},
  {STRING_TO_DECIMAL_OPERATOR_NAME,        intrinsic_to_identical_post_pv},
  {FUNC_TO_DECIMAL_OPERATOR_NAME,          intrinsic_to_identical_post_pv},
  {FILE_TO_DECIMAL_OPERATOR_NAME,          intrinsic_to_identical_post_pv},
  //3bsd
  {SECONVERT_OPERATOR_NAME,                intrinsic_to_identical_post_pv},
  {SFCONVERT_OPERATOR_NAME,                intrinsic_to_identical_post_pv},
  {SGCONVERT_OPERATOR_NAME,                intrinsic_to_identical_post_pv},
  {ECONVERT_OPERATOR_NAME,                 intrinsic_to_identical_post_pv},
  {FCONVERT_OPERATOR_NAME,                 intrinsic_to_identical_post_pv},
  {GCONVERT_OPERATOR_NAME,                 intrinsic_to_identical_post_pv},
  {QECONVERT_OPERATOR_NAME,                intrinsic_to_identical_post_pv},
  {QFCONVERT_OPERATOR_NAME,                intrinsic_to_identical_post_pv},
  {QGCONVERT_OPERATOR_NAME,                intrinsic_to_identical_post_pv},

  /* C IO system functions in man -S 2 unistd.h */

  {C_OPEN_FUNCTION_NAME,                   unix_io_function_to_post_pv},
  {CREAT_FUNCTION_NAME,                    unix_io_function_to_post_pv},
  {C_CLOSE_FUNCTION_NAME,                  unix_io_function_to_post_pv},
  {C_WRITE_FUNCTION_NAME,                  unix_io_function_to_post_pv},
  {C_READ_FUNCTION_NAME,                   unix_io_function_to_post_pv},
  {LINK_FUNCTION_NAME,                     intrinsic_to_identical_post_pv},
  {SYMLINK_FUNCTION_NAME,                  intrinsic_to_identical_post_pv},
  {UNLINK_FUNCTION_NAME,                   intrinsic_to_identical_post_pv},

  {FCNTL_FUNCTION_NAME,                    unix_io_function_to_post_pv},
  {FSYNC_FUNCTION_NAME,                    unix_io_function_to_post_pv},
  {FDATASYNC_FUNCTION_NAME,                unix_io_function_to_post_pv},
  {IOCTL_FUNCTION_NAME,                    unix_io_function_to_post_pv},
  {SELECT_FUNCTION_NAME,                   unix_io_function_to_post_pv},
  {PSELECT_FUNCTION_NAME,                  unix_io_function_to_post_pv},
  {STAT_FUNCTION_NAME,                     intrinsic_to_identical_post_pv}, /* sys/stat.h */
  {FSTAT_FUNCTION_NAME,                    unix_io_function_to_post_pv},
  {LSTAT_FUNCTION_NAME,                    intrinsic_to_identical_post_pv},



  /*  {char *getenv(const char *, 0, 0},
      {long int labs(long, 0, 0},
      {ldiv_t ldiv(long, long, 0, 0},*/

  /* F95 */
  {ALLOCATE_FUNCTION_NAME,                 unknown_intrinsic_to_post_pv},
  {DEALLOCATE_FUNCTION_NAME,               unknown_intrinsic_to_post_pv},
  {ETIME_FUNCTION_NAME,                    intrinsic_to_identical_post_pv},
  {DTIME_FUNCTION_NAME,                    intrinsic_to_identical_post_pv},

  /* F2003 */
  {C_LOC_FUNCTION_NAME,                    intrinsic_to_identical_post_pv},

  /* BSD <err.h> */
  /* SG: concerning the err* family of functions, they also exit() from the program
   * This is not represented in the EXIT_FUNCTION_NAME description, so neither it is here
   * but it seems an error to me */
  {ERR_FUNCTION_NAME,                      c_io_function_to_post_pv},
  {ERRX_FUNCTION_NAME,                     c_io_function_to_post_pv},
  {WARN_FUNCTION_NAME,                     c_io_function_to_post_pv},
  {WARNX_FUNCTION_NAME,                    c_io_function_to_post_pv},
  {VERR_FUNCTION_NAME,                     c_io_function_to_post_pv},
  {VERRX_FUNCTION_NAME,                    c_io_function_to_post_pv},
  {VWARN_FUNCTION_NAME,                    c_io_function_to_post_pv},
  {VWARNX_FUNCTION_NAME,                   c_io_function_to_post_pv},

  /*Conforming to 4.3BSD, POSIX.1-2001.*/
  /* POSIX.1-2001 declares this function obsolete; use nanosleep(2) instead.*/
  /*POSIX.1-2008 removes the specification of usleep()*/
  {USLEEP_FUNCTION_NAME,                   intrinsic_to_identical_post_pv},

  /* _POSIX_C_SOURCE >= 199309L */
  {NANOSLEEP_FUNCTION_NAME,                intrinsic_to_identical_post_pv},

  /* {int mblen(const char *, size_t, 0, 0},
     {size_t mbstowcs(wchar_t *, const char *, size_t, 0, 0},
     {int mbtowc(wchar_t *, const char *, size_t, 0, 0},
     {void qsort(void *, size_t, size_t,
     int (*)(const void *, const void *), 0, 0},
     {int rand(void, 0, 0},
     {void *realloc(void *, size_t, 0, 0},
     {void srand(unsigned int, 0, 0},
     {double strtod(const char *, char **, 0, 0},
     {long int strtol(const char *, char **, int, 0, 0},
     {unsigned long int strtoul(const char *, char **, int, 0, 0},
     {int system(const char *, 0, 0},
     {int wctomb(char *, wchar_t, 0, 0},
     {size_t wcstombs(char *, const wchar_t *, size_t, 0, 0},
     {void _exithandle(void, 0, 0},
     {double drand48(void, 0, 0},
     {double erand48(unsigned short *, 0, 0},
     {long jrand48(unsigned short *, 0, 0},
     {void lcong48(unsigned short *, 0, 0},
     {long lrand48(void, 0, 0},
     {long mrand48(void, 0, 0},
     {long nrand48(unsigned short *, 0, 0},
     {unsigned short *seed48(unsigned short *, 0, 0},
     {void srand48(long, 0, 0},
     {int putenv(char *, 0, 0},
     {void setkey(const char *, 0, 0},
     {void swab(const char *, char *, ssize_t, 0, 0},
     {int       mkstemp(char *, 0, 0},
     {int       mkstemp64(char *, 0, 0},
     {long a64l(const char *, 0, 0},
     {char *ecvt(double, int, int *, int *, 0, 0},
     {char *fcvt(double, int, int *, int *, 0, 0},
     {char *gcvt(double, int, char *, 0, 0},
     {int getsubopt(char **, char *const *, char **, 0, 0},
     {int  grantpt(int, 0, 0},
     {char *initstate(unsigned, char *, size_t, 0, 0},
     {char *l64a(long, 0, 0},
     {char *mktemp(char *, 0, 0},
     {char *ptsname(int, 0, 0},
     {long random(void, 0, 0},
     {char *realpath(const char *, char *, 0, 0},
     {char *setstate(const char *, 0, 0},
     {void srandom(unsigned, 0, 0},
     {int ttyslot(void, 0, 0},
     {int  unlockpt(int, 0, 0},
     {void *valloc(size_t, 0, 0},
     {int dup2(int, int, 0, 0},
     {char *qecvt(long double, int, int *, int *, 0, 0},
     {char *qfcvt(long double, int, int *, int *, 0, 0},
     {char *qgcvt(long double, int, char *, 0, 0},
     {char *getcwd(char *, size_t, 0, 0},
     {const char *getexecname(void, 0, 0},
     {char *getlogin(void, 0, 0},
     {int getopt(int, char *const *, const char *, 0, 0},
     {char *optarg;
     {int optind, opterr, optopt;
     {char *getpass(const char *, 0, 0},
     {char *getpassphrase(const char *, 0, 0},
     {int getpw(uid_t, char *, 0, 0},
     {int isatty(int, 0, 0},
     {void *memalign(size_t, size_t, 0, 0},
     {char *ttyname(int, 0, 0},
     {long long atoll(const char *, 0, 0},
     {long long llabs(long long, 0, 0},
     {lldiv_t lldiv(long long, long long, 0, 0},
     {char *lltostr(long long, char *, 0, 0},
     {long long strtoll(const char *, char **, int, 0, 0},
     {unsigned long long strtoull(const char *, char **, int, 0, 0},
     {char *ulltostr(unsigned long long, char *, 0, 0},*/
  {NULL, 0}
};

/******************************************/

static void assignment_intrinsic_to_post_pv(entity  __attribute__ ((unused))func,
					    list func_args,
					    list l_in, pv_results * pv_res,
					    pv_context *ctxt)
{
  expression lhs = EXPRESSION(CAR(func_args));
  expression rhs = EXPRESSION(CAR(CDR(func_args)));
  assignment_to_post_pv(lhs, false, rhs, false, l_in, pv_res, ctxt);
}

static void binary_arithmetic_operator_to_post_pv(entity func, list func_args,
						  list l_in, pv_results * pv_res,
						  pv_context *ctxt)
{
  list l_in_cur;

  pips_debug(1, "begin\n");

  pv_results pv_res1 = make_pv_results();
  expression arg1 = EXPRESSION(CAR(func_args));
  expression_to_post_pv(arg1, l_in, &pv_res1, ctxt);

  l_in_cur = pv_res1.l_out;
  if (l_in != l_in_cur) gen_full_free_list(l_in);

  pv_results pv_res2 = make_pv_results();
  expression arg2 = EXPRESSION(CAR(CDR(func_args)));
  expression_to_post_pv(arg2, l_in_cur, &pv_res2, ctxt);
  if (pv_res2.l_out != l_in_cur) gen_full_free_list(l_in_cur);


  type t1 = expression_to_type(arg1);
  bool pointer_t1 = pointer_type_p(t1);
  free_type(t1);
  type t2 = expression_to_type(arg2);
  bool pointer_t2 = pointer_type_p(t2);
  free_type(t2);

  const char* func_name = entity_local_name(func);

  /* From ISO/IEC 9899:TC3 :

     - For addition, either both operands shall have arithmetic type,
     or one operand shall be a pointer to an object type and the other
     shall have integer type.

     - For substraction, one of the following shall hold :

       - both operands have arithmetic types
       - both operands are pointers to qualified or unqualified versions of
         compatible object types;
         (in this case, both pointers shall point to elements of the same array object,
	 or one past the last element of the array object; the result is the difference
         of the subscripts of the two array elements)
	 (as a consequence the result is not of a pointer type)
       - the left operand is a pointer to an object type and the right operand has
         integer type
  */
  list l_eff1 = pv_res1.result_paths;
  list l_eff2 = pv_res2.result_paths;

  bool pointer_arithmetic = false;
  list l_eff_pointer = NIL;
  expression other_arg = expression_undefined;

  if (pointer_t1 && !pointer_t2 )
    /* pointer arithmetic, the pointer is in the first expression */
    {
      pointer_arithmetic = true;
      l_eff_pointer = l_eff1;
      other_arg = arg2;
    }
  else if (pointer_t2 && !pointer_t1)
    {
      pointer_arithmetic = true;
      l_eff_pointer = l_eff2;
      other_arg = arg1;
    }

  if (pointer_arithmetic)
    {
      free_pv_results_paths(pv_res);
      if (!anywhere_effect_p(EFFECT(CAR(l_eff_pointer))))
	{
	  /* build new effects */
	  list l_new_eff = NIL;
	  list l_new_eff_kind = NIL;
	  expression new_arg = copy_expression(other_arg);

	  if (same_string_p(func_name, MINUS_C_OPERATOR_NAME))
	    {
	      entity unary_minus_ent =
		gen_find_tabulated(make_entity_fullname(TOP_LEVEL_MODULE_NAME,
							UNARY_MINUS_OPERATOR_NAME),
				   entity_domain);
	      new_arg = MakeUnaryCall(unary_minus_ent, new_arg);
	    }

	  FOREACH(EFFECT, eff, l_eff_pointer)
	    {
	      effect new_eff = copy_effect(eff);
	      (*effect_add_expression_dimension_func)(new_eff, new_arg);
	      l_new_eff = CONS(EFFECT, new_eff, l_new_eff);
	      l_new_eff_kind = CONS(CELL_INTERPRETATION,
				    make_cell_interpretation_address_of(), NIL);
	    }
	  gen_nreverse(l_new_eff);
	  gen_nreverse(l_new_eff_kind);
	  pv_res->result_paths = l_new_eff;
	  pv_res->result_paths_interpretations = l_new_eff_kind;
	  free_expression(new_arg);
	}
    }
  pv_res->l_out = pv_res2.l_out;
  free_pv_results_paths(&pv_res1);
  free_pv_results_paths(&pv_res2);
  pips_debug_pv_results(1, "end with pv_res =\n", *pv_res);
}

static void unary_arithmetic_operator_to_post_pv(entity __attribute__ ((unused))func,
						 list func_args,
						 list l_in, pv_results * pv_res,
						 pv_context *ctxt)
{
  expression arg = EXPRESSION(CAR(func_args));
  expression_to_post_pv(arg, l_in, pv_res, ctxt);
  ifdebug(1)
    {
      type t = expression_to_type(arg);
      pips_assert("unary arithmetic operators should not have pointer arguments",
		  !pointer_type_p(t));
      free_type(t);
    }
}

static void update_operator_to_post_pv(entity func, list func_args, list l_in,
				       pv_results * pv_res, pv_context *ctxt)
{
  pips_debug(1, "begin for update operator\n");

  expression arg = EXPRESSION(CAR(func_args));
  expression_to_post_pv(arg, l_in, pv_res, ctxt);
  list l_in_cur = pv_res->l_out;
  if (l_in != l_in_cur) gen_full_free_list(l_in);

  type t = expression_to_type(arg);
  if (pointer_type_p(t))
    {
      list l_lhs_eff = pv_res->result_paths;
      //list l_lhs_kind = pv_res->result_paths_interpretations;
      const char* func_name = entity_local_name(func);

      pips_assert("update operators admit a single path\n",
		  gen_length(l_lhs_eff) == (size_t) 1);

      effect lhs_eff = EFFECT(CAR(l_lhs_eff));
      effect rhs_eff = copy_effect(lhs_eff);
      list l_rhs_kind = CONS(CELL_INTERPRETATION, make_cell_interpretation_address_of(),
			     NIL);

      if (!anywhere_effect_p(lhs_eff))
	{
	  /* build rhs */
	  expression new_dim = expression_undefined;

	  if (same_string_p(func_name, POST_INCREMENT_OPERATOR_NAME)
	      || same_string_p(func_name, PRE_INCREMENT_OPERATOR_NAME))
	    {
	      new_dim = int_to_expression(1);
	    }
	  else if (same_string_p(func_name, POST_DECREMENT_OPERATOR_NAME)
		   || same_string_p(func_name, PRE_DECREMENT_OPERATOR_NAME))
	    {
	      new_dim = int_to_expression(-1);
	    }
	  else
	    //pips_internal_error("unexpected update operator on pointers");
	    new_dim = make_unbounded_expression();

	  (*effect_add_expression_dimension_func)(rhs_eff, new_dim);
	  /*l_lhs_kind = CONS(CELL_INTERPRETATION, make_cell_interpretation_address_of(),
			    NIL);*/
	}
      list l_rhs_eff = CONS(EFFECT, rhs_eff, NIL);
      single_pointer_assignment_to_post_pv(lhs_eff, l_rhs_eff, l_rhs_kind, false,
					   l_in_cur, pv_res, ctxt);
    }
  pips_debug_pv_results(1, "end with pv_res:\n", *pv_res);
}

static void logical_operator_to_post_pv(entity __attribute__ ((unused))func,
					list func_args, list l_in,
					pv_results *pv_res, pv_context *ctxt)
{
  expression e1 = EXPRESSION(CAR(func_args));
  expression e2 = EXPRESSION(CAR(CDR(func_args)));
  pv_results pv_res1 = make_pv_results();

  /* If it is a pure logical operator (&& or ||), there is a
     sequence point just after the evaluation of the first operand
     so we must evaluate the second operand in the memory store
     resulting from the evaluation of the first operand.
     If it's a relational operator, the evaluation order is not important,
     so let's do the same !
  */
  /* first operand */
  expression_to_post_pv(e1, l_in, &pv_res1, ctxt);
  list l_in_cur = pv_res1.l_out;
  if (l_in != l_in_cur) gen_full_free_list(l_in);
  free_pv_results_paths(&pv_res1);

  /* second operand */
  expression_to_post_pv(e2, l_in_cur, pv_res, ctxt);
  free_pv_results_paths(pv_res);
  /* the resulting path of a relational expression is not a pointer value */
  pv_res->result_paths = NIL;
  pv_res->result_paths_interpretations = NIL;
}

static void conditional_operator_to_post_pv(entity __attribute__ ((unused)) func,
					    list func_args, list l_in,
					    pv_results * pv_res, pv_context *ctxt)
{
  pips_debug(1, "begin for conditional operator\n");

  expression t_cond = EXPRESSION(CAR(func_args));
  POP(func_args);
  expression t_true = EXPRESSION(CAR(func_args));
  POP(func_args);
  expression t_false = EXPRESSION(CAR(func_args));

  pv_results pv_res_cond = make_pv_results();
  expression_to_post_pv(t_cond, l_in, &pv_res_cond, ctxt);

  list l_in_branches = pv_res_cond.l_out;

  pv_results pv_res_true = make_pv_results();
  list l_in_true = gen_full_copy_list(l_in_branches);
  expression_to_post_pv(t_true, l_in_true, &pv_res_true, ctxt);
  if (pv_res_true.l_out != l_in_true) gen_full_free_list(l_in_true);

  pv_results pv_res_false = make_pv_results();
  list l_in_false = gen_full_copy_list(l_in_branches);
  expression_to_post_pv(t_false, l_in_false, &pv_res_false, ctxt);
  if (pv_res_false.l_out != l_in_false) gen_full_free_list(l_in_false);


  pv_res->l_out = (*ctxt->pvs_may_union_func)(pv_res_true.l_out,
					      gen_full_copy_list(pv_res_false.l_out));

  /* well, it should be a union, but there may not be such stupid things as (..)? a:a;
   I cannot use the effects test union operator because I must also merge
   interpretations */
  pv_res->result_paths = gen_nconc(pv_res_true.result_paths, pv_res_false.result_paths);
  effects_to_may_effects(pv_res->result_paths);
  pv_res->result_paths_interpretations =
    gen_nconc(pv_res_true.result_paths_interpretations,
	      pv_res_false.result_paths_interpretations);

  pips_debug_pv_results(1, "end with pv_results =\n", *pv_res);
  pips_debug(1, "end\n");

}



static void dereferencing_to_post_pv(entity __attribute__ ((unused))func, list func_args,
				     list l_in, pv_results * pv_res, pv_context *ctxt)
{
  expression_to_post_pv(EXPRESSION(CAR(func_args)), l_in, pv_res, ctxt);
  list l_eff_ci = pv_res->result_paths_interpretations;
  FOREACH(EFFECT, eff, pv_res->result_paths)
    {
      cell_interpretation ci = CELL_INTERPRETATION(CAR(l_eff_ci));
      if (cell_interpretation_value_of_p(ci))
	effect_add_dereferencing_dimension(eff);
      else
	cell_interpretation_tag(ci) = is_cell_interpretation_value_of;
      POP(l_eff_ci);
    }
}

static void field_to_post_pv(entity __attribute__ ((unused))func, list func_args,
			     list l_in, pv_results * pv_res, pv_context *ctxt)
{
  expression e2 = EXPRESSION(CAR(CDR(func_args)));
  syntax s2 = expression_syntax(e2);
  reference r2 = syntax_reference(s2);
  entity f = reference_variable(r2);

  pips_assert("e2 is a reference", syntax_reference_p(s2));
  pips_debug(4, "It's a field operator\n");

  expression_to_post_pv(EXPRESSION(CAR(func_args)), l_in, pv_res, ctxt);
  FOREACH(EFFECT, eff, pv_res->result_paths)
    {
      effect_add_field_dimension(eff,f);
    }
}

static void point_to_to_post_pv(entity __attribute__ ((unused))func, list func_args,
				list l_in, pv_results * pv_res, pv_context *ctxt)
{
  expression e2 = EXPRESSION(CAR(CDR(func_args)));
  syntax s2 = expression_syntax(e2);
  entity f;

  pips_assert("e2 is a reference", syntax_reference_p(s2));
  f = reference_variable(syntax_reference(s2));

  pips_debug(4, "It's a point to operator\n");
  expression_to_post_pv(EXPRESSION(CAR(func_args)), l_in, pv_res, ctxt);

  FOREACH(EFFECT, eff, pv_res->result_paths)
    {
      /* We add a dereferencing */
      effect_add_dereferencing_dimension(eff);
      /* we add the field dimension */
      effect_add_field_dimension(eff,f);
    }
}

static void address_of_to_post_pv(entity __attribute__ ((unused))func, list func_args,
				  list l_in, pv_results * pv_res, pv_context *ctxt)
{
  expression_to_post_pv(EXPRESSION(CAR(func_args)), l_in, pv_res, ctxt);
  FOREACH(CELL_INTERPRETATION, ci, pv_res->result_paths_interpretations)
    {
      cell_interpretation_tag(ci) = is_cell_interpretation_address_of;
    }
}

static void c_io_function_to_post_pv(entity func, list func_args, list l_in,
				     pv_results * pv_res, pv_context *ctxt)
{
  const char* func_name = entity_local_name(func);
  list general_args = NIL;
  bool free_general_args = false;
  list l_in_cur = l_in;

  /* first argument is a FILE*  */
  if (same_string_p(func_name,FCLOSE_FUNCTION_NAME)
      || same_string_p(func_name,FPRINTF_FUNCTION_NAME)
      || same_string_p(func_name,FSCANF_FUNCTION_NAME)
      || same_string_p(func_name,VFPRINTF_FUNCTION_NAME)
      || same_string_p(func_name,VFSCANF_FUNCTION_NAME)
      || same_string_p(func_name,ISOC99_VFSCANF_FUNCTION_NAME)
      || same_string_p(func_name,FGETC_FUNCTION_NAME)
      || same_string_p(func_name,GETC_FUNCTION_NAME)
      || same_string_p(func_name,_IO_GETC_FUNCTION_NAME)
      || same_string_p(func_name,FGETPOS_FUNCTION_NAME)
      || same_string_p(func_name,FSEEK_FUNCTION_NAME)
      || same_string_p(func_name,FSETPOS_FUNCTION_NAME)
      || same_string_p(func_name,FTELL_FUNCTION_NAME)
      || same_string_p(func_name,C_REWIND_FUNCTION_NAME)
      || same_string_p(func_name,CLEARERR_FUNCTION_NAME)
      || same_string_p(func_name,FEOF_FUNCTION_NAME)
      || same_string_p(func_name,FERROR_FUNCTION_NAME))
    {
      general_args = CDR(func_args);
    }
  /* last argument is a FILE*  */
  else if (same_string_p(func_name,FGETS_FUNCTION_NAME)
	   || same_string_p(func_name,FPUTC_FUNCTION_NAME)
	   || same_string_p(func_name,FPUTS_FUNCTION_NAME)
	   || same_string_p(func_name,PUTC_FUNCTION_NAME)
	   || same_string_p(func_name,_IO_PUTC_FUNCTION_NAME)
	   || same_string_p(func_name,UNGETC_FUNCTION_NAME)
	   || same_string_p(func_name,FREAD_FUNCTION_NAME)
	   || same_string_p(func_name,FWRITE_FUNCTION_NAME))
    {
      for(; !ENDP(CDR(func_args)); POP(func_args))
	{
	  general_args = CONS(EXPRESSION, EXPRESSION(CAR(func_args)), general_args);
	}
      free_general_args = true;
    }

  /* we assume that there is no effects on aliasing due to FILE* argument if any */

  safe_intrinsic_to_post_pv(func, general_args, l_in_cur, pv_res, ctxt);

  if (free_general_args)
    gen_free_list(general_args);
}

static void unix_io_function_to_post_pv(entity func, list func_args, list l_in,
					pv_results * pv_res, pv_context *ctxt)
{
  safe_intrinsic_to_post_pv(func, func_args, l_in, pv_res, ctxt);
}

static void string_function_to_post_pv(entity func, list func_args, list l_in,
				       pv_results * pv_res, pv_context *ctxt)
{
  safe_intrinsic_to_post_pv(func, func_args, l_in, pv_res, ctxt);
}

static void va_list_function_to_post_pv(entity func, list func_args, list l_in,
					pv_results * pv_res, pv_context *ctxt)
{
  safe_intrinsic_to_post_pv(func, func_args, l_in, pv_res, ctxt);
}


static void free_to_post_pv(list l_free_eff, list l_in,
			    pv_results * pv_res, pv_context *ctxt)
{
  pips_debug_effects(5, "begin with input effects :\n", l_free_eff);

  FOREACH(EFFECT, eff, l_free_eff)
    {
      /* for each freed pointer, find it's targets */

      list l_remnants = NIL;
      cell_relation exact_eff_pv = cell_relation_undefined;
      list l_values = NIL;

      pips_debug_effect(4, "begin, looking for an exact target for eff:\n",
			eff);

      l_values = effect_find_equivalent_pointer_values(eff, l_in,
						       &exact_eff_pv,
						       &l_remnants);
      pips_debug_pvs(3, "l_values:\n", l_values);
      pips_debug_pvs(3, "l_remnants:\n", l_remnants);
      pips_debug_pv(3, "exact_eff_pv:\n", exact_eff_pv);

      list l_heap_eff = NIL;

      if (exact_eff_pv != cell_relation_undefined)
	{
	  cell heap_c = cell_undefined;
	  /* try to find the heap location */
	  cell c1 = cell_relation_first_cell(exact_eff_pv);
	  entity e1 = reference_variable(cell_reference(c1));

	  cell c2 = cell_relation_second_cell(exact_eff_pv);
	  entity e2 = reference_variable(cell_reference(c2));

	  if (entity_flow_or_context_sentitive_heap_location_p(e1))
	    heap_c = c1;
	  else if (entity_flow_or_context_sentitive_heap_location_p(e2))
	    heap_c = c2;
	  if (!cell_undefined_p(heap_c))
	    l_heap_eff =
	      CONS(EFFECT,
		   make_effect(copy_cell(heap_c),
			       make_action_write_memory(),
			       copy_approximation(effect_approximation(eff)),
			       make_descriptor_none()),
		   NIL);
	  else
	    {
	      /* try to find the heap target in remants */
	      cell other_c = cell_undefined;
	      if (same_entity_p(effect_entity(eff), e1))
		other_c = c2;
	      else other_c = c1;
	      list l_tmp =
		CONS(EFFECT,
		     make_effect(copy_cell(other_c),
				 make_action_write_memory(),
				 copy_approximation(effect_approximation(eff)),
				 make_descriptor_none()),
		     NIL);
	      free_to_post_pv(l_tmp, l_remnants, pv_res, ctxt);
	      pv_res->l_out = CONS(CELL_RELATION,
				   copy_cell_relation(exact_eff_pv),
				   pv_res->l_out);
	      gen_free_list(l_tmp);
	      return;
	    }
	}
      else
	{
	  /* try first to find another target */
	  FOREACH(CELL_RELATION, pv_tmp, l_values)
	    {
	      cell heap_c = cell_undefined;
	      /* try to find the heap location */
	      cell c1 = cell_relation_first_cell(pv_tmp);
	      entity e1 = reference_variable(cell_reference(c1));

	      cell c2 = cell_relation_second_cell(pv_tmp);
	      entity e2 = reference_variable(cell_reference(c2));

	      if (entity_flow_or_context_sentitive_heap_location_p(e1))
		heap_c = c1;
	      else if (entity_flow_or_context_sentitive_heap_location_p(e2))
		heap_c = c2;
	      if (!cell_undefined_p(heap_c))
		l_heap_eff =
		  CONS(EFFECT,
		       make_effect(copy_cell(heap_c),
				   make_action_write_memory(),
				   cell_relation_exact_p(pv_tmp)
				   ? copy_approximation
				   (effect_approximation(eff))
				   :make_approximation_may(),
				   make_descriptor_none()),
		       NIL);

	    }
	}

      if (!ENDP(l_heap_eff))
	{
	  /* assign an undefined_value to freed pointer */
	  list l_rhs =
	    CONS(EFFECT,
		 make_effect( make_undefined_pointer_value_cell(),
			      make_action_write_memory(),
			      make_approximation_exact(),
			      make_descriptor_none()),
		 NIL);
	  list l_kind = CONS(CELL_INTERPRETATION,
			     make_cell_interpretation_value_of(),
			     NIL);
	  single_pointer_assignment_to_post_pv(eff, l_rhs, l_kind,
					       false, l_in, pv_res, ctxt);
	  gen_full_free_list(l_rhs);
	  gen_full_free_list(l_kind);
	  free_pv_results_paths(pv_res);
	  if (l_in != pv_res->l_out)
	    {
	      gen_full_free_list(l_in);
	      l_in = pv_res->l_out;
	    }
	  pips_debug_pvs(5, "l_in after assigning "
			 "undefined value to freed pointer:\n",
			 l_in);

	  FOREACH(EFFECT, heap_eff, l_heap_eff)
	    {
	      entity heap_e =
		reference_variable(effect_any_reference(heap_eff));
	      pips_debug(5, "heap entity found (%s)\n", entity_name(heap_e));

	      pointer_values_remove_var(heap_e,
					effect_may_p(eff)
					|| effect_may_p(heap_eff),
					l_in,
					pv_res, ctxt);
	      l_in= pv_res->l_out;
	    }
	}
      else /* no flow or context sensitive variable found */
	{
	  list l_rhs =
	    CONS(EFFECT,
		 make_effect( make_undefined_pointer_value_cell(),
			      make_action_write_memory(),
			      make_approximation_exact(),
			      make_descriptor_none()),
		 NIL);
	  list l_kind = CONS(CELL_INTERPRETATION,
			     make_cell_interpretation_value_of(),
			     NIL);
	  single_pointer_assignment_to_post_pv(eff, l_rhs, l_kind,
					       false, l_in, pv_res, ctxt);
	  gen_full_free_list(l_rhs);
	  gen_full_free_list(l_kind);
	  free_pv_results_paths(pv_res);
	  if (l_in != pv_res->l_out)
	    {
	      gen_full_free_list(l_in);
	      l_in = pv_res->l_out;
	    }
	  pips_debug_pvs(5, "l_in after assigning "
			 "undefined value to freed pointer:\n",
			 l_in);
	}

    } /* FOREACH */
  pips_debug_pvs(5, "end with pv_res->l_out:", pv_res->l_out);
}

static void heap_intrinsic_to_post_pv(entity func, list func_args, list l_in,
				      pv_results * pv_res, pv_context *ctxt)
{
  const char* func_name = entity_local_name(func);
  expression malloc_arg = expression_undefined;
  bool free_malloc_arg = false;
  list l_in_cur = l_in;


  /* free the previously allocated path if need be */
  if (same_string_p(func_name, FREE_FUNCTION_NAME)
      || same_string_p(func_name, REALLOC_FUNCTION_NAME))
    {
      expression free_ptr_exp = EXPRESSION(CAR(func_args));

      pv_results pv_res_arg = make_pv_results();
      expression_to_post_pv(free_ptr_exp, l_in_cur, &pv_res_arg, ctxt);
      if (pv_res_arg.l_out != l_in_cur)
	{
	  gen_full_free_list(l_in_cur);
	  l_in_cur = pv_res_arg.l_out;
	}
      free_to_post_pv(pv_res_arg.result_paths, l_in_cur, pv_res, ctxt);
      l_in_cur = pv_res->l_out;
    }

  /* Then retrieve the allocated path if any */
  if(same_string_p(func_name, MALLOC_FUNCTION_NAME))
    {
      malloc_arg = EXPRESSION(CAR(func_args));
    }
  else if(same_string_p(func_name, CALLOC_FUNCTION_NAME))
    {
      malloc_arg =
	make_op_exp("*",
		   copy_expression(EXPRESSION(CAR(func_args))),
		   copy_expression(EXPRESSION(CAR(CDR(func_args)))));
      free_malloc_arg = true;
    }
  else if (same_string_p(func_name, REALLOC_FUNCTION_NAME))
    {
      malloc_arg = EXPRESSION(CAR(CDR(func_args)));
    }
  else if (same_string_p(func_name, STRDUP_FUNCTION_NAME))
  {
      malloc_arg = MakeBinaryCall(
              entity_intrinsic(PLUS_OPERATOR_NAME),
              int_to_expression(1),
              MakeUnaryCall(
                  entity_intrinsic(STRLEN_FUNCTION_NAME),
                  copy_expression(EXPRESSION(CAR(func_args)))
                  )
              );
      free_malloc_arg = true;
  }

  if (!expression_undefined_p(malloc_arg))
    {

      /* first, impact of argument evaluation on pointer values */
      pv_results pv_res_arg = make_pv_results();
      expression_to_post_pv(malloc_arg, l_in_cur, &pv_res_arg, ctxt);
      free_pv_results_paths(&pv_res_arg);
      if (pv_res_arg.l_out != l_in_cur)
	{
	  gen_full_free_list(l_in_cur);
	  l_in_cur = pv_res_arg.l_out;
	}

      sensitivity_information si =
	make_sensitivity_information(pv_context_statement_head(ctxt),
				     get_current_module_entity(),
				     NIL);
      entity e = malloc_to_abstract_location(malloc_arg, &si);

      effect eff = make_effect(make_cell_reference(make_reference(e, NIL)),
			       make_action_write_memory(),
			       make_approximation_exact(),
			       make_descriptor_none());

      if (!entity_all_heap_locations_p(e) &&
	  !entity_all_module_heap_locations_p(e))
	  effect_add_dereferencing_dimension(eff);
      else
	effect_to_may_effect(eff);
      pv_res->result_paths = CONS(EFFECT, eff, NIL);
      pv_res->result_paths_interpretations =
	CONS(CELL_INTERPRETATION,
	     make_cell_interpretation_address_of(),
	     NIL);
      if (free_malloc_arg)
	free_expression(malloc_arg);
    }

  pv_res->l_out = l_in_cur;
}

#if 0
static void stop_to_post_pv(entity __attribute__ ((unused))func, list func_args,
			    list l_in, pv_results * pv_res, pv_context *ctxt)
{
  /* The call is never returned from. No information is available
     for the dead code that follows.
  */
  pv_res->l_out = NIL;
  pv_res->result_paths = NIL;
  pv_res->result_paths_interpretations = NIL;
}
#endif

static void c_return_to_post_pv(entity __attribute__ ((unused)) func, list func_args,
				list l_in, pv_results * pv_res, pv_context *ctxt)
{
  /* but we have to evaluate the impact
     of the argument evaluation on pointer values
     eliminate local variables, retrieve the value of the returned pointer if any...
  */
  expression_to_post_pv(EXPRESSION(CAR(func_args)), l_in, pv_res, ctxt);
}

static void safe_intrinsic_to_post_pv(entity __attribute__ ((unused)) func,
				      list func_args, list l_in,
				      pv_results * pv_res, pv_context *ctxt)
{
  /* first assume that all pointers reachable from arguments are written and set to
     anywhere */
  /* this should be quickly refined */
  list l_anywhere_eff = CONS(EFFECT, make_anywhere_effect(make_action_write_memory()),
			     NIL);
  list l_rhs_kind = CONS(CELL_INTERPRETATION,
			 make_cell_interpretation_address_of(), NIL);

  pips_debug(1, "begin\n");

  if (!ENDP(func_args))
    {
      list lw = NIL;
      list l_in_cur = l_in;
      FOREACH(EXPRESSION, arg, func_args)
	{
	  pv_results pv_res_arg = make_pv_results();
	  expression_to_post_pv(arg, l_in_cur, &pv_res_arg, ctxt);
	  free_pv_results_paths(&pv_res_arg);
	  if (pv_res_arg.l_out != l_in_cur)
	    {
	      //gen_full_free_list(l_in_cur);
	      l_in_cur = pv_res_arg.l_out;
	    }
	  /* well this is not safe in case of arguments with external function calls
	     I should use the result paths of pv_res_arg, but there is a lot of work
	     done in c_actual_argument_to_may_ summary_effect about arrays, and I'm not
	     sure expression_to_post_pv does the same job
	  */
	  lw = gen_nconc(lw, c_actual_argument_to_may_summary_effects(arg, 'w'));
	}

      pips_debug_effects(3, "effects to be killed: \n", lw);

      /* assume all pointers now point to anywhere */

      FOREACH(EFFECT, eff, lw)
	{
	  bool to_be_freed = false;
	  type t = cell_to_type(effect_cell(eff), &to_be_freed);
	  if (pointer_type_p(t))
	    {
	      single_pointer_assignment_to_post_pv(eff,
						   l_anywhere_eff, l_rhs_kind,
						   false, l_in_cur,
						   pv_res, ctxt);
	      if (pv_res->l_out != l_in_cur)
		{
		  gen_full_free_list(l_in_cur);
		  l_in_cur = pv_res->l_out;
		}
	      free_pv_results_paths(pv_res);
	    }
	  if (to_be_freed) free_type(t);
	}
      pv_res->l_out = l_in_cur;
    }
  else
    pv_res->l_out = l_in;
  /* then retrieve the return value type to set pv_res->result_paths */
  /* if it is a pointer type, set it to anywhere for the moment
     specific work should be done for each intrinsic
  */
  pv_res->result_paths = l_anywhere_eff;
  pv_res->result_paths_interpretations = l_rhs_kind;

  pips_debug_pv_results(1, "ending with pv_res:", *pv_res);

}

static void intrinsic_to_identical_post_pv(entity __attribute__ ((unused)) func,
					   list __attribute__ ((unused)) func_args,
					   list l_in, pv_results * pv_res, pv_context __attribute__ ((unused)) *ctxt)
{
  pv_res->l_out = l_in;
}

static void unknown_intrinsic_to_post_pv(entity __attribute__ ((unused)) func,
					 list __attribute__ ((unused)) func_args,
					 list __attribute__ ((unused)) l_in,
					 pv_results __attribute__ ((unused)) *pv_res,
					 pv_context __attribute__ ((unused)) *ctxt)
{
  pips_internal_error("not a C intrinsic");
}

static void default_intrinsic_to_post_pv(entity __attribute__ ((unused)) func,
					 list func_args, list l_in,
					 pv_results * pv_res, pv_context *ctxt)
{
  list l_in_cur = l_in;
  FOREACH(EXPRESSION, arg, func_args)
    {
      /* free the result paths of the previous expression evaluation */
      free_pv_results_paths(pv_res);
      expression_to_post_pv(arg, l_in_cur, pv_res, ctxt);
      l_in_cur = pv_res->l_out;
    }
}


void intrinsic_to_post_pv(entity func, list func_args, list l_in,
			  pv_results * pv_res, pv_context *ctxt)
{
  const char* func_name = entity_local_name(func);
  pips_debug(1, "begin for %s\n", func_name);

  IntrinsicToPostPVDescriptor *pid = IntrinsicToPostPVDescriptorTable;

  while (pid->name != NULL)
    {
      if (strcmp(pid->name, func_name) == 0)
	{
	  (*(pid->to_post_pv_function))(func, func_args, l_in, pv_res, ctxt);
	  pips_debug_pv_results(2, "resulting pv_res:", *pv_res);
	  pips_debug(1, "end\n");
	  return;
	}
      pid += 1;
  }

  pips_internal_error("unknown intrinsic %s", func_name);
  pips_debug(1, "end\n");
  return;
}

