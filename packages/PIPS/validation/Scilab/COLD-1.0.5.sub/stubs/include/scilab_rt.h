/*------------------------------------- -*- C -*-
 *
 *  (c) HPC Project - 2010-2011
 *
 */



// to avoid an undefined vprintf when compiling with gcc -std=c99
#define __USE_MINGW_ANSI_STDIO 0

// to use strdup from string.h (glibc)
#define _XOPEN_SOURCE 500


#include <assert.h>
#include <complex.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>

#include "scilab_rt_pred_rand.h"

#define MAX(a,b) ((b) > (a) ? (b) : (a))
#define MIN(a,b) ((b) < (a) ? (b) : (a))
#define DEG2RAD(d) (d*SCILAB_PI/180.0)
#define RAD2DEG(d) (d*180.0/SCILAB_PI)
#define SIGN(d) ((d > 0) ? 1 : ((d < 0) ? -1 : 0))
#define ZSIGN(d) (d/cabs(d))
#define RAND1() ((double)(rand())/(double)RAND_MAX)
#define ZRAND1() (RAND1() + RAND1() * I)
#define PREDICTEDRAND1() ((double)(_pred_rand())/(double)_predicted_max)
#define PREDICTEDZRAND1() (PREDICTEDRAND1() + PREDICTEDRAND1() * I)
#define ROUND(a) ((a >= 0) ? (int)(a + 0.5) : (int)(a - 0.5))
#define BITAND(a,b) (a & b)
#define BOOL2S(a) ((a == 0) ? 0 : 1)
#define FIX(a) ((a >= 0) ? floor(a) : ceil(a))


extern const int __SCILAB_RT_FALSE__;
extern const int __SCILAB_RT_TRUE__;

extern const double SCILAB_E;
extern const double SCILAB_PI;
extern const double SCILAB_EPS;


/* Most of the following could go to scilab_rt_init.h. It stays here for
   full exposure */

// this works on IEEE little-endian machines only

/* Note that any symbol beginning with __ is reserved by the C norm and
   should not be used... */

typedef union { unsigned char __c[8]; double __d; } __double_union_t__;

extern __double_union_t__ __scilab_nan__;
extern __double_union_t__ __scilab_inf__;
extern __double_union_t__ __huge_val_max;
extern __double_union_t__ __huge_val_min;

#define   SCILAB_NAN      (__scilab_nan__.__d)
#define   SCILAB_INF      (__scilab_inf__.__d)
#define __HUGE_VAL_MAX__  (__huge_val_max.__d)
#define __HUGE_VAL_MIN__  (__huge_val_min.__d)
#define __INT_VAL_MAX__   INT_MAX
#define __INT_VAL_MIN__   INT_MIN

extern int __scilab_exit__;
extern int __scilab_verbose__;

extern int __scilab_is_running__;

#include "scilab_rt_abs.h"
#include "scilab_rt_acos.h"
#include "scilab_rt_acosd.h"
#include "scilab_rt_acosh.h"
#include "scilab_rt_add.h"
#include "scilab_rt_and.h"
#include "scilab_rt_asin.h"
#include "scilab_rt_asind.h"
#include "scilab_rt_asinh.h"
#include "scilab_rt_assign.h"
#include "scilab_rt_atan.h"
#include "scilab_rt_atand.h"
#include "scilab_rt_atanh.h"
#include "scilab_rt_bitand.h"
#include "scilab_rt_bool2s.h"
#include "scilab_rt_ceil.h"
#include "scilab_rt_chol.h"
#include "scilab_rt_clock.h"
#include "scilab_rt_conj.h"
#include "scilab_rt_cos.h"
#include "scilab_rt_cosd.h"
#include "scilab_rt_cosh.h"
#include "scilab_rt_cumsum.h"
#include "scilab_rt_date.h"
#include "scilab_rt_datefind.h"
#include "scilab_rt_datenum.h"
#include "scilab_rt_datevec.h"
#include "scilab_rt_det.h"
#include "scilab_rt_diag.h"
#include "scilab_rt_diff.h"
#include "scilab_rt_disp.h"
#include "scilab_rt_display.h"
#include "scilab_rt_div.h"
#include "scilab_rt_eltdiv.h"
#include "scilab_rt_eltmul.h"
#include "scilab_rt_eltpow.h"
#include "scilab_rt_eomday.h"
#include "scilab_rt_eq.h"
#include "scilab_rt_etime.h"
#include "scilab_rt_exit.h"
#include "scilab_rt_exp.h"
#include "scilab_rt_eye.h"
#include "scilab_rt_fft.h"
#include "scilab_rt_fix.h"
#include "scilab_rt_floor.h"
#include "scilab_rt_ge.h"
#include "scilab_rt_getdate.h"
#include "scilab_rt_grand.h"
#include "scilab_rt_gsort.h"
#include "scilab_rt_gt.h"
#include "scilab_rt_halt.h"
#include "scilab_rt_havewindow.h"
#include "scilab_rt_hess.h"
#include "scilab_rt_ifft.h"
#include "scilab_rt_imag.h"
#include "scilab_rt_init.h"
#include "scilab_rt_int32.h"
#include "scilab_rt_interp1.h"
#include "scilab_rt_inttrap.h"
#include "scilab_rt_inv.h"
#include "scilab_rt_io_m.h"
#include "scilab_rt_le.h"
#include "scilab_rt_leftdivide.h"
#include "scilab_rt_length.h"
#include "scilab_rt_lines.h"
#include "scilab_rt_linspace.h"
#include "scilab_rt_log.h"
#include "scilab_rt_log2.h"
#include "scilab_rt_lsq.h"
#include "scilab_rt_lt.h"
#include "scilab_rt_matrix.h"
#include "scilab_rt_max.h"
#include "scilab_rt_memory_management.h"
#include "scilab_rt_meshgrid.h"
#include "scilab_rt_mean.h"
#include "scilab_rt_min.h"
#include "scilab_rt_modulo.h"
#include "scilab_rt_mul.h"
#include "scilab_rt_ne.h"
#include "scilab_rt_norm.h"
#include "scilab_rt_not.h"
#include "scilab_rt_now.h"
#include "scilab_rt_ones.h"
#include "scilab_rt_or.h"
#include "scilab_rt_pmodulo.h"
#include "scilab_rt_pow.h"
#include "scilab_rt_predicted_rand.h"
#include "scilab_rt_qr.h"
#include "scilab_rt_rand.h"
#include "scilab_rt_rand_parallel.h"
#include "scilab_rt_rand_parallel_intern.h"
#include "scilab_rt_rand_parallel_internH.h"
#include "scilab_rt_real.h"
#include "scilab_rt_reduce_to_bool.h"
#include "scilab_rt_round.h"
#include "scilab_rt_schur.h"
#include "scilab_rt_send_to_scilab.h"
#include "scilab_rt_sign.h"
#include "scilab_rt_sin.h"
#include "scilab_rt_sind.h"
#include "scilab_rt_sinh.h"
#include "scilab_rt_size.h"
#include "scilab_rt_sleep.h"
#include "scilab_rt_spec.h"
#include "scilab_rt_sqrt.h"
#include "scilab_rt_stacksize.h"
#include "scilab_rt_start_scilab.h"
#include "scilab_rt_stdev.h"
#include "scilab_rt_strings.h"
#include "scilab_rt_squeeze.h"
#include "scilab_rt_sum.h"
#include "scilab_rt_svd.h"
#include "scilab_rt_sub.h"
#include "scilab_rt_tan.h"
#include "scilab_rt_tand.h"
#include "scilab_rt_tanh.h"
#include "scilab_rt_terminate.h"
#include "scilab_rt_terminate_scilab.h"
#include "scilab_rt_tictoc.h"
#include "scilab_rt_transpose.h"
#include "scilab_rt_transposeConjugate.h"
#include "scilab_rt_variance.h"
#include "scilab_rt_weekday.h"
#include "scilab_rt_yearfrac.h"
#include "scilab_rt_zeros.h"

#include "scilab_rt_RTgenerated.h"

