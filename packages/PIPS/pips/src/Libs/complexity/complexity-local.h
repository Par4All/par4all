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
/*
 *
 *  COMPLEXITY EVALUATION    P. Berthomier   21-08-90
 *
 *                           L. Zhou         21-03-90
 *
 * Modifications:
 *  - deletion of includes in this include file; Francois Irigoin,
 *    13 March 1991
 *
 *  - deletion of #define for operators ;
 *    (all the operators and the instrinsics are defined inside of ri-util-local.h); Molka Becher, 08.03.2011
 */
#include "matrice.h"

#define COMPLEXITY_PACKAGE_NAME "COMPLEXITY"
#define COMPLEXITY_UNDEFINED complexity_undefined
/*
#define COMPLEXITY_UNDEFINED_SYMBOL "<Complexity undefined>"
*/
#define COMPLEXITY_UNDEFINED_P(c) ((c)==complexity_undefined)
#define COMPLEXITY_NOT_FOUND ((complexity) HASH_UNDEFINED_VALUE)

#define MAX_CONTROLS_IN_UNSTRUCTURED 100

/* pseudo-variable for unknown variables     */
#define UNKNOWN_VARIABLE_NAME "UNKNOWN_VARIABLE"

/* pseudo-variable for default iteration number of a loop */
#define UNKNOWN_RANGE_NAME "UNKNOWN_RANGE"

#define TCST_NAME "_TCST_"

/* Prefix added to a variable name when its value is unknown but has to
 * be used in a complexity formulae
 */
#define UNKNOWN_VARIABLE_VALUE_PREFIX "U_"

/* defined complexity data file names here. LZ 13/03/92 */
#define COST_DATA "operation index memory trigo transcend overhead"

/* defines for "complexity_fprint" calls */
#define DO_PRINT_STATS true                 
#define DONT_PRINT_STATS false           
#define PRINT_LOCAL_NAMES true
#define PRINT_GLOBAL_NAMES false

#define HASH_LOOP_INDEX ((char *) 4)     /* Used as `value' field in the hash-table */
#define HASH_USER_VARIABLE ((char *) 8)  /* "hash_complexity_parameters"... useful? */
#define HASH_FORMAL_PARAM ((char *) 12)
#define HASH_COMMON_VARIABLE ((char *) 16) /* used for variable in common. LZ 04/12/91 */

#define ZERO_BYTE 0                      
#define INT_NBYTES 4                     /* number of bytes used by a standard int        */
#define FLOAT_NBYTES 4                   /* number of bytes used by a single-precision    */
#define DOUBLE_NBYTES 8                  /* number of bytes used by a double-precision    */
#define COMPLEX_NBYTES 2*FLOAT_NBYTES    /* complex, single-precision */
#define DCOMPLEX_NBYTES 2*DOUBLE_NBYTES  /* complex, double-precision */
#define MAKE_INT_BASIC make_basic(is_basic_int, (void *) INT_NBYTES)
#define MAKE_ADDRESS_BASIC make_basic(is_basic_int, (void *) DEFAULT_POINTER_TYPE_SIZE)
#define MAKE_FLOAT_BASIC make_basic(is_basic_float, (void *) FLOAT_NBYTES)
#define MAKE_DOUBLE_BASIC make_basic(is_basic_float, (void *) DOUBLE_NBYTES)
#define MAKE_COMPLEX_BASIC make_basic(is_basic_complex, (void *) COMPLEX_NBYTES)
#define MAKE_DCOMPLEX_BASIC make_basic(is_basic_complex, (void *) DCOMPLEX_NBYTES)
#define MAKE_STRING_BASIC make_basic(is_basic_string, make_value(is_value_unknown, UU))

#define hash_contains_p(htp, key) (hash_get(htp, key) != HASH_UNDEFINED_VALUE)
#define hash_contains_formal_param_p(htp, key) (hash_get(htp, key) == HASH_FORMAL_PARAM)
#define hash_contains_user_var_p(htp, key) (hash_get(htp, key) == HASH_USER_VARIABLE)
#define hash_contains_common_var_p(htp, key) (hash_get(htp, key) == HASH_COMMON_VARIABLE)
#define hash_contains_loop_index_p(htp, key) (hash_get(htp, key) == HASH_LOOP_INDEX)


/* defines for "expression_to_polynome" parameters */
#define KEEP_SYMBOLS true
#define DONT_KEEP_SYMBOLS false
#define MAXIMUM_VALUE  1
#define MINIMUM_VALUE -1
#define EXACT_VALUE    0
#define TAKE_MAX(m) ((m) == MAXIMUM_VALUE)
#define TAKE_MIN(m) ((m) == MINIMUM_VALUE)
#define KEEP_EXACT(m) ((m) == EXACT_VALUE)

/* Intrinsics costs defines */

typedef struct intrinsic_cost_rec {
    char *name;
    _int min_basic_result;
    _int min_nbytes_result;
    _int int_cost;
    _int float_cost;
    _int double_cost;
    _int complex_cost;
    _int dcomplex_cost;
} intrinsic_cost_record;

#define LOOP_INIT_OVERHEAD "LOOP-INIT-OVERHEAD"
#define LOOP_BRANCH_OVERHEAD "LOOP-BRANCH-OVERHEAD"
#define CONDITION_OVERHEAD "CONDITION-OVERHEAD"

#define CALL_ZERO_OVERHEAD      "CALL-ZERO-OVERHEAD"      
#define CALL_ONE_OVERHEAD       "CALL-ONE-OVERHEAD"       
#define CALL_TWO_OVERHEAD       "CALL-TWO-OVERHEAD"       
#define CALL_THREE_OVERHEAD     "CALL-THREE-OVERHEAD"     
#define CALL_FOUR_OVERHEAD      "CALL-FOUR-OVERHEAD"      
#define CALL_FIVE_OVERHEAD      "CALL-FIVE-OVERHEAD"      
#define CALL_SIX_OVERHEAD       "CALL-SIX-OVERHEAD"       
#define CALL_SEVEN_OVERHEAD     "CALL-SEVEN-OVERHEAD"   

/* TYPE_CAST_COST added to handle cast case ; Molka Becher  */
#define TYPE_CAST_COST  "TypeCast"  

/* the above two lines are added for 6th cost file, overhead. LZ 280993 */
/* overhead is divided into two. init and branch 081093 */

#define MEMORY_READ_NAME "MEMORY-READ"
#define ONE_INDEX_NAME   "ONE-INDEX"    /* to count indexation costs in multi-dim arrays */
#define TWO_INDEX_NAME   "TWO-INDEX"    /* to count indexation costs in multi-dim arrays */
#define THREE_INDEX_NAME "THREE-INDEX"
#define FOUR_INDEX_NAME  "FOUR-INDEX"
#define FIVE_INDEX_NAME  "FIVE-INDEX"
#define SIX_INDEX_NAME   "SIX-INDEX"
#define SEVEN_INDEX_NAME "SEVEN-INDEX"
#define STRING_INTRINSICS_COST 1
#define LOGICAL_INTRINSICS_COST 1

#define DONT_COUNT_THAT     0,   0,   0,   0,  0
#define EMPTY_COST 0,0,0,0,0
#define DISCRIMINE_TYPES    1,  10, 100,1000,10000
#define REAL_INTRINSIC    100, 100, 100, 100, 100
#define DOUBLE_INTRINSIC  200, 200, 200, 200, 200
#define COMPLEX_INTRINSIC 400, 400, 400, 400, 400

#define MEMORY_READ_COST  DONT_COUNT_THAT
#define MEMORY_WRITE_COST DONT_COUNT_THAT
#define PLUS_MINUS_COST   1, 10, 20, 20, 40
/*
#define MULTIPLY_COST     50, 50, 100, 100, 200
#define DIVIDE_COST       50, 50, 100, 100, 200
*/
#define MULTIPLY_COST     PLUS_MINUS_COST
#define DIVIDE_COST       PLUS_MINUS_COST
#define POWER_COST        100, 100, 200, 200, 400

#define TRANSC_COST       REAL_INTRINSIC
#define DTRANSC_COST      DOUBLE_INTRINSIC
#define CTRANSC_COST      COMPLEX_INTRINSIC
#define TRIGO_COST        REAL_INTRINSIC
#define DTRIGO_COST       DOUBLE_INTRINSIC
#define CTRIGO_COST       COMPLEX_INTRINSIC
#define TRIGOH_COST       REAL_INTRINSIC
#define DTRIGOH_COST      DOUBLE_INTRINSIC
#define CTRIGOH_COST      COMPLEX_INTRINSIC

#define TWO_INDEX_COST    DONT_COUNT_THAT
#define THREE_INDEX_COST  DONT_COUNT_THAT
#define FOUR_INDEX_COST   DONT_COUNT_THAT
#define FIVE_INDEX_COST   DONT_COUNT_THAT
#define SIX_INDEX_COST    DONT_COUNT_THAT
#define SEVEN_INDEX_COST  DONT_COUNT_THAT

