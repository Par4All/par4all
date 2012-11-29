/**
 *
 *   \file             step_common.h
 *   \author           Abdellah Kouadri.
 *                     Daniel Millot.
 *                     Frédérique Silber-Chaussumier.
 *                     Alain Muller
 *   \date             04/02/2010
 *   \version          2.0
 *   \brief            Some useful declarations and definitions
 *
 */

#ifndef STEP_COMMON_H_
#define STEP_COMMON_H_

/**
 * Used in genereted file
 */
#ifndef MIN
#define MIN(a,b) ((a)<(b))?(a):(b)
#endif
#ifndef MAX
#define MAX(a,b) ((a)>(b))?(a):(b)
#endif

#define STEP_MAX_NB_LOOPSLICES 16
#define STEP_INDEX_SLICE_LOW    1
#define STEP_INDEX_SLICE_UP     2


/**
 * Supported languages
 */
#define STEP_C                 0
#define STEP_FORTRAN           1

/**
 * Handled construction
 */


#define STEP_PARALLEL          100
#define STEP_DO                101
#define STEP_PARALLEL_DO       102
#define STEP_MASTER            103
//#define STEP_CRITICAL          104
#define STEP_BARRIER           105
#define STEP_SINGLE            106
#define STEP_THREADPRIVATE     107

#define STEP_NOWAIT            0
#define STEP_WAIT              1

/**
 * Data types actually handled by the runtime
 */
#define STEP_INTEGER           0
#define STEP_REAL              1
#define STEP_DOUBLE_PRECISION  2
#define STEP_COMPLEX           3

#define STEP_INTEGER1          4
#define STEP_INTEGER2          5
#define STEP_INTEGER4          6
#define STEP_INTEGER8          7
#define STEP_REAL4             8
#define STEP_REAL8             9
#define STEP_REAL16           10
#define STEP_COMPLEX8         11
#define STEP_COMPLEX16        12

#define STEP_TYPE_UNDEFINED   13

/**
* Supported communication algorithms
*/
#define STEP_TAG_DEFAULT       0

#define STEP_NBLOCKING_ALG     0
//#define STEP_BLOCKING_ALG_1  1
//#define STEP_BLOCKING_ALG_2  2
//#define STEP_BLOCKING_ALG_3  3
//#define STEP_BLOCKING_ALG_4  4
//#define STEP_ONETOALL_BCAST  0


/**
* Supported reduction operators
*/
#define STEP_PROD_REDUCE     0
#define STEP_MAX_REDUCE      1
#define STEP_MIN_REDUCE      2
#define STEP_SUM_REDUCE      3
#define STEP_UNDEF_REDUCE    4
//#define STEP_LAND            4
//#define STEP_BAND            5
//#define STEP_LOR             6
//#define STEP_BOR             7
//#define STEP_LXOR            8
//#define STEP_BXOR            9
//#define STEP_MINLOC         10
//#define STEP_MAXLOC         11

#endif
