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
/* package generic effects :  Be'atrice Creusillet 5/97
 *
 * File: intrinsics.c
 * ~~~~~~~~~~~~~~~~~~
 *
 * This File contains the generic functions necessary for the computation of
 * all types of proper effects and proper references of intrinsics.
 *
 *
 * Modifications :
 * --------------
 *
 * Molka Becher (MB), May-June 2010
 *
 * - Reordering of the existing intrinsics according to ISO/IEC 9899
 * - Add of missing C Intrinsics
 * - Add of generic_string_effects and memmove_effects for handling string.h
 * effects
 */

#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <unistd.h>

#include "genC.h"

#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "database.h"

#include "ri-util.h"
#include "effects-util.h"
#include "constants.h"
#include "misc.h"
#include "text-util.h"
#include "text.h"

#include "properties.h"

#include "transformer.h"
#include "semantics.h"
#include "resources.h"

#include "effects-generic.h"
//#include "effects-simple.h"


/********************************************************* LOCAL FUNCTIONS */

static list time_effects(entity, list );
static list no_write_effects(entity e,list args);
static list safe_c_read_only_effects(entity e,list args);
static list address_expression_effects(entity e,list args);
static list conditional_effects(entity e,list args);
static list address_of_effects(entity e,list args);
static list affect_effects(entity e,list args);
static list update_effects(entity e,list args);
static list unique_update_effects(entity e,list args);
static list assign_substring_effects(entity e,list args);
static list substring_effect(entity e,list args);
static list make_io_read_write_memory_effects(entity e, list args);
static list io_effects(entity e, list args);
static list c_io_effects(entity e, list args);
static list read_io_effects(entity e, list args);
static list effects_of_ioelem(expression exp, tag act);
static list effects_of_C_ioelem(expression exp, tag act);
static list effects_of_iolist(list exprs, tag act);
static list effects_of_implied_do(expression exp, tag act);
static list generic_io_effects(entity e,list args, bool system_p);
static list unix_io_effects(entity e,list args);
static list any_rgs_effects(entity e,list args, bool init_p);
static list rgs_effects(entity e,list args);
static list rgsi_effects(entity e,list args);
static list any_heap_effects(entity e,list args);
static list va_list_effects(entity e, list args);
static list search_or_sort_effects(entity e, list args);
/* MB */
static list generic_string_effects(entity e,list args);
static list memmove_effects(entity e, list args);
static list strtoxxx_effects(entity e, list args);
static list strdup_effects(entity, list);



/* the following data structure indicates wether an io element generates
a read effects or a write effect. the kind of effect depends on the
instruction type: for instance, access generates a read effect if used
within an open statement, and a write effect when used inside an inquire
statement */

typedef struct IoElementDescriptor {
    string StmtName;
    string IoElementName;
    tag ReadOrWrite, MayOrMust;
} IoElementDescriptor;

static IoElementDescriptor IoElementDescriptorUndefined;

static IoElementDescriptor IoElementDescriptorTable[] = {
  {"OPEN",      "UNIT=",        is_action_read,  is_approximation_exact},
  {"OPEN",      "ERR=",         is_action_read,  is_approximation_may},
  {"OPEN",      "FILE=",        is_action_read,  is_approximation_exact},
  {"OPEN",      "STATUS=",      is_action_read,  is_approximation_may},
  {"OPEN",      "ACCESS=",      is_action_read,  is_approximation_exact},
  {"OPEN",      "FORM=",        is_action_read,  is_approximation_exact},
  {"OPEN",      "RECL=",        is_action_read,  is_approximation_exact},
  {"OPEN",      "BLANK=",       is_action_read,  is_approximation_may},
  {"OPEN",      "IOSTAT=",      is_action_write, is_approximation_may},

  {"CLOSE",     "UNIT=",        is_action_read,  is_approximation_exact},
  {"CLOSE",     "ERR=",         is_action_read,  is_approximation_may},
  {"CLOSE",     "STATUS=",      is_action_read,  is_approximation_may},
  {"CLOSE",     "IOSTAT=",      is_action_write, is_approximation_may},

  {"INQUIRE",   "UNIT=",        is_action_read,  is_approximation_exact},
  {"INQUIRE",   "ERR=",         is_action_read,  is_approximation_may},
  {"INQUIRE",   "FILE=",        is_action_read,  is_approximation_exact},
  {"INQUIRE",   "IOSTAT=",      is_action_write, is_approximation_exact},
  {"INQUIRE",   "EXIST=",       is_action_write, is_approximation_exact},
  {"INQUIRE",   "OPENED=",      is_action_write, is_approximation_exact},
  {"INQUIRE",   "NUMBER=",      is_action_write, is_approximation_exact},
  {"INQUIRE",   "NAMED=",       is_action_write, is_approximation_exact},
  {"INQUIRE",   "NAME=",        is_action_write, is_approximation_exact},
  {"INQUIRE",   "ACCESS=",      is_action_write, is_approximation_exact},
  {"INQUIRE",   "SEQUENTIAL=",  is_action_write, is_approximation_exact},
  {"INQUIRE",   "DIRECT=",      is_action_write, is_approximation_exact},
  {"INQUIRE",   "FORM=",        is_action_write, is_approximation_exact},
  {"INQUIRE",   "FORMATTED=",   is_action_write, is_approximation_exact},
  {"INQUIRE",   "UNFORMATTED=", is_action_write, is_approximation_exact},
  {"INQUIRE",   "RECL=",        is_action_write, is_approximation_exact},
  {"INQUIRE",   "NEXTREC=",     is_action_write, is_approximation_exact},
  {"INQUIRE",   "BLANK=",       is_action_write, is_approximation_exact},

  {"BACKSPACE", "UNIT=",        is_action_read,  is_approximation_exact},
  {"BACKSPACE", "ERR=",         is_action_read,  is_approximation_may},
  {"BACKSPACE", "IOSTAT=",      is_action_write, is_approximation_may},

  {"ENDFILE",   "UNIT=",        is_action_read,  is_approximation_exact},
  {"ENDFILE",   "ERR=",         is_action_read,  is_approximation_may},
  {"ENDFILE",   "IOSTAT=",      is_action_write, is_approximation_may},

  {"REWIND",    "UNIT=",        is_action_read,  is_approximation_exact},
  {"REWIND",    "ERR=",         is_action_read,  is_approximation_may},
  {"REWIND",    "IOSTAT=",      is_action_write, is_approximation_may},

  {"READ",      "FMT=",         is_action_read,  is_approximation_exact},
  {"READ",      "UNIT=",        is_action_read,  is_approximation_exact},
  {"READ",      "REC=",         is_action_read,  is_approximation_exact},
  {"READ",      "ERR=",         is_action_read,  is_approximation_may},
  {"READ",      "END=",         is_action_read,  is_approximation_exact},
  {"READ",      "IOSTAT=",      is_action_write, is_approximation_may},
  {"READ",      "IOLIST=",      is_action_write, is_approximation_exact},

  {"WRITE",     "ADVANCE=",     is_action_read,  is_approximation_exact},
  {"WRITE",     "FMT=",         is_action_read,  is_approximation_exact},
  {"WRITE",     "UNIT=",        is_action_read,  is_approximation_exact},
  {"WRITE",     "REC=",         is_action_read,  is_approximation_exact},
  {"WRITE",     "ERR=",         is_action_read,  is_approximation_may},
  {"WRITE",     "END=",         is_action_read,  is_approximation_exact},
  {"WRITE",     "IOSTAT=",      is_action_write, is_approximation_may},
  {"WRITE",     "IOLIST=",      is_action_read,  is_approximation_exact},

  /* C IO intrinsics arranged in the order of the standard ISO/IEC 9899:TC2. MB */

  /* The field IoElementName is used to describe the function's pattern
     defined according to the standard ISO/IEC 9899 (BC, july 2009) :
     n      when there is only the read effect on the value of the actual
            argument.
     r,w,x  for read, write, or read and write effects on the object
            pointed to by the actual argument.
     R,W    are used for formatted ios, because we need a specific handling
            for char * arguments subsequent to the format specifier.
     *      means that the last effect is repeated for the last arguments
            (varargs).
     s      for a FILE * argument ("s" stands for "stream").
     f      for an integer file descriptor (unix io system calls).
     v      for a va_list argument (this could be enhanced in the future
            to distinguish between read and write effects on the components
            of the va_list).

     The tag fields are not relevant.
  */

  /* Input/Output <stdio.h> */

  {FCLOSE_FUNCTION_NAME,        "s",       is_action_read, is_approximation_exact},
  {FOPEN_FUNCTION_NAME,         "rr",      is_action_read, is_approximation_exact},
  {FPRINTF_FUNCTION_NAME,       "srR*",    is_action_read, is_approximation_exact},
  {FSCANF_FUNCTION_NAME,        "srW*",    is_action_read, is_approximation_exact},
  {ISOC99_FSCANF_FUNCTION_NAME, "srW*",    is_action_read, is_approximation_exact},
  {PRINTF_FUNCTION_NAME,        "rR*",     is_action_read, is_approximation_exact},
  {SCANF_FUNCTION_NAME,         "rW*",     is_action_read, is_approximation_exact},
  {ISOC99_SCANF_FUNCTION_NAME,   "rW*",    is_action_read, is_approximation_exact},
  {SNPRINTF_FUNCTION_NAME,      "wnrR*",   is_action_read, is_approximation_exact},
  {SPRINTF_FUNCTION_NAME,       "wrR*",    is_action_read, is_approximation_exact},
  {SSCANF_FUNCTION_NAME,        "rrW*",    is_action_read, is_approximation_exact},
  {ISOC99_SSCANF_FUNCTION_NAME, "rrW*",    is_action_read, is_approximation_exact},
  {VFPRINTF_FUNCTION_NAME,      "srv",     is_action_read, is_approximation_exact},
  {VFSCANF_FUNCTION_NAME,       "srv",     is_action_read, is_approximation_exact},
  {ISOC99_VFSCANF_FUNCTION_NAME,"srv",     is_action_read, is_approximation_exact},
  {VPRINTF_FUNCTION_NAME,       "rv",      is_action_read, is_approximation_exact},
  {VSCANF_FUNCTION_NAME,        "rv",      is_action_read, is_approximation_exact},
  {ISOC99_VSCANF_FUNCTION_NAME, "rv",      is_action_read, is_approximation_exact},
  {VSNPRINTF_FUNCTION_NAME,     "wnrv",    is_action_read, is_approximation_exact},
  {VSPRINTF_FUNCTION_NAME,      "wrv",     is_action_read, is_approximation_exact},
  {VSSCANF_FUNCTION_NAME,       "rrv",     is_action_read, is_approximation_exact},
  {ISOC99_VSSCANF_FUNCTION_NAME,"rrv",     is_action_read, is_approximation_exact},
  {FGETC_FUNCTION_NAME,         "s",       is_action_read, is_approximation_exact},
  {FGETS_FUNCTION_NAME,         "wns",     is_action_read, is_approximation_exact},
  {FPUTC_FUNCTION_NAME,         "ns",      is_action_read, is_approximation_exact},
  {FPUTS_FUNCTION_NAME,         "rs",      is_action_read, is_approximation_exact},
  {GETC_FUNCTION_NAME,          "s",       is_action_read, is_approximation_exact},
  {_IO_GETC_FUNCTION_NAME,      "s",       is_action_read, is_approximation_exact},
  {GETCHAR_FUNCTION_NAME,       "",        is_action_read, is_approximation_exact},
  {GETS_FUNCTION_NAME,          "w",       is_action_read, is_approximation_exact},
  {PUTC_FUNCTION_NAME,          "ns",      is_action_read, is_approximation_exact},
  {_IO_PUTC_FUNCTION_NAME,      "ns",      is_action_read, is_approximation_exact},
  {PUTCHAR_FUNCTION_NAME,       "n",       is_action_read, is_approximation_exact},
  {PUTS_FUNCTION_NAME,          "r",       is_action_read, is_approximation_exact},
  {UNGETC_FUNCTION_NAME,        "ns",      is_action_read, is_approximation_exact},
  {FREAD_FUNCTION_NAME,         "wnns",    is_action_read, is_approximation_exact},
  {FWRITE_FUNCTION_NAME,        "rnns",    is_action_read, is_approximation_exact},
  {FGETPOS_FUNCTION_NAME,       "sw",      is_action_read, is_approximation_exact},
  {FSEEK_FUNCTION_NAME,         "snn",     is_action_read, is_approximation_exact},
  {FSETPOS_FUNCTION_NAME,       "sr",      is_action_read, is_approximation_exact},
  {FTELL_FUNCTION_NAME,         "s",       is_action_read, is_approximation_exact},
  {C_REWIND_FUNCTION_NAME,      "s",       is_action_read, is_approximation_exact},
  {CLEARERR_FUNCTION_NAME,      "s",       is_action_read, is_approximation_exact},
  {FEOF_FUNCTION_NAME,          "s",       is_action_read, is_approximation_exact},
  {FERROR_FUNCTION_NAME,        "s",       is_action_read, is_approximation_exact},
  {PERROR_FUNCTION_NAME,        "r",       is_action_read, is_approximation_exact},


  /* UNIX IO system calls */

  {C_OPEN_FUNCTION_NAME,        "nn",      is_action_read, is_approximation_exact},
  {CREAT_FUNCTION_NAME,         "nn",      is_action_read, is_approximation_exact},
  {C_CLOSE_FUNCTION_NAME,       "f",       is_action_read, is_approximation_exact},
  {C_WRITE_FUNCTION_NAME,       "frr",     is_action_read, is_approximation_exact},
  {C_READ_FUNCTION_NAME,        "fwn",     is_action_read, is_approximation_exact},
  {LINK_FUNCTION_NAME,          "r*",      is_action_read, is_approximation_exact},
  {SYMLINK_FUNCTION_NAME,       "r*",      is_action_read, is_approximation_exact},
  {UNLINK_FUNCTION_NAME,        "r",       is_action_read, is_approximation_exact},
  {FCNTL_FUNCTION_NAME,         "fnn*",    is_action_read, is_approximation_exact},
  {FSYNC_FUNCTION_NAME,         "f",       is_action_read, is_approximation_exact},
  {FDATASYNC_FUNCTION_NAME,     "f",       is_action_read, is_approximation_exact},
  {IOCTL_FUNCTION_NAME,         "fn*",     is_action_read, is_approximation_exact},
  {SELECT_FUNCTION_NAME,        "nrrrr",   is_action_read, is_approximation_exact},
  {PSELECT_FUNCTION_NAME,       "nrrrrw",  is_action_read, is_approximation_exact},
  {FSTAT_FUNCTION_NAME,         "nw",      is_action_read, is_approximation_exact},


  /* wchar.h */

  {FWSCANF_FUNCTION_NAME,       "srW*",    is_action_read, is_approximation_exact},
  {SWSCANF_FUNCTION_NAME,       "rrW*",    is_action_read, is_approximation_exact},
  {WSCANF_FUNCTION_NAME,        "rW*",     is_action_read, is_approximation_exact},

  /* BSD err.h */
  {ERR_FUNCTION_NAME,		"rrR*",	   is_action_read, is_approximation_exact},
  {ERRX_FUNCTION_NAME,		"rrR*",	   is_action_read, is_approximation_exact},
  {WARN_FUNCTION_NAME,		"rR*",	   is_action_read, is_approximation_exact},
  {WARNX_FUNCTION_NAME,		"rR*",	   is_action_read, is_approximation_exact},
  {VERR_FUNCTION_NAME,		"rrR*",	   is_action_read, is_approximation_exact},
  {VERRX_FUNCTION_NAME,		"rrR*",	   is_action_read, is_approximation_exact},
  {VWARN_FUNCTION_NAME,		"rR*",	   is_action_read, is_approximation_exact},
  {VWARNX_FUNCTION_NAME,	"rR*",	   is_action_read, is_approximation_exact},

  /* Fortran extensions for asynchronous IO's */

  {BUFFERIN_FUNCTION_NAME,      "xrwr",    is_action_read, is_approximation_exact},
  {BUFFEROUT_FUNCTION_NAME,     "xrrr",    is_action_read, is_approximation_exact},


  {0,                            0,        0,              0}
};


/* the following data structure describes an intrinsic function: its
name and the function to apply on a call to this intrinsic to get the
effects of the call */

/* These intrinsics are arranged in the order of the standard ISO/IEC 9899:TC2. MB */

typedef struct IntrinsicDescriptor
{
    string name;
    list (*effects_function)();
} IntrinsicDescriptor;

static IntrinsicDescriptor IntrinsicEffectsDescriptorTable[] = {
  {PLUS_OPERATOR_NAME,                     no_write_effects},
  {MINUS_OPERATOR_NAME,                    no_write_effects},
  {DIVIDE_OPERATOR_NAME,                   no_write_effects},
  {MULTIPLY_OPERATOR_NAME,                 no_write_effects},
  {INVERSE_OPERATOR_NAME,                  no_write_effects},
  {UNARY_MINUS_OPERATOR_NAME,              no_write_effects},
  {POWER_OPERATOR_NAME,                    no_write_effects},
  {EQUIV_OPERATOR_NAME,                    no_write_effects},
  {NON_EQUIV_OPERATOR_NAME,                no_write_effects},
  {OR_OPERATOR_NAME,                       no_write_effects},
  {AND_OPERATOR_NAME,                      no_write_effects},
  {LESS_THAN_OPERATOR_NAME,                no_write_effects},
  {GREATER_THAN_OPERATOR_NAME,             no_write_effects},
  {LESS_OR_EQUAL_OPERATOR_NAME,            no_write_effects},
  {GREATER_OR_EQUAL_OPERATOR_NAME,         no_write_effects},
  {EQUAL_OPERATOR_NAME,                    no_write_effects},
  {NON_EQUAL_OPERATOR_NAME,                no_write_effects},
  {CONCATENATION_FUNCTION_NAME,            no_write_effects},
  {NOT_OPERATOR_NAME,                      no_write_effects},

  {CONTINUE_FUNCTION_NAME,                 no_write_effects},
  {"ENDDO",                                no_write_effects},
  {PAUSE_FUNCTION_NAME,                    make_io_read_write_memory_effects},
  {RETURN_FUNCTION_NAME,                   no_write_effects},
  {STOP_FUNCTION_NAME,                     make_io_read_write_memory_effects},
  {END_FUNCTION_NAME,                      no_write_effects},
  {FORMAT_FUNCTION_NAME,                   no_write_effects},

  { IMPLIED_COMPLEX_NAME,                  no_write_effects},
  { IMPLIED_DCOMPLEX_NAME,                 no_write_effects},

  {INT_GENERIC_CONVERSION_NAME,            no_write_effects},
  {IFIX_GENERIC_CONVERSION_NAME,           no_write_effects},
  {IDINT_GENERIC_CONVERSION_NAME,          no_write_effects},
  {REAL_GENERIC_CONVERSION_NAME,           no_write_effects},
  {FLOAT_GENERIC_CONVERSION_NAME,          no_write_effects},
  {DFLOAT_GENERIC_CONVERSION_NAME,         no_write_effects},
  {SNGL_GENERIC_CONVERSION_NAME,           no_write_effects},
  {DBLE_GENERIC_CONVERSION_NAME,           no_write_effects},
  {DREAL_GENERIC_CONVERSION_NAME,          no_write_effects}, /* Added for Arnauld Leservot */
  {CMPLX_GENERIC_CONVERSION_NAME,          no_write_effects},
  {DCMPLX_GENERIC_CONVERSION_NAME,         no_write_effects},
  {INT_TO_CHAR_CONVERSION_NAME,            no_write_effects},
  {CHAR_TO_INT_CONVERSION_NAME,            no_write_effects},
  {AINT_CONVERSION_NAME,                   no_write_effects},
  {DINT_CONVERSION_NAME,                   no_write_effects},
  {ANINT_CONVERSION_NAME,                  no_write_effects},
  {DNINT_CONVERSION_NAME,                  no_write_effects},
  {NINT_CONVERSION_NAME,                   no_write_effects},
  {IDNINT_CONVERSION_NAME,                 no_write_effects},
  {IABS_OPERATOR_NAME,                     no_write_effects},
  {ABS_OPERATOR_NAME,                      no_write_effects},
  {DABS_OPERATOR_NAME,                     no_write_effects},
  {CABS_OPERATOR_NAME,                     no_write_effects},
  {CDABS_OPERATOR_NAME,                    no_write_effects},

  {MODULO_OPERATOR_NAME,                   no_write_effects},
  {REAL_MODULO_OPERATOR_NAME,              no_write_effects},
  {DOUBLE_MODULO_OPERATOR_NAME,            no_write_effects},
  {ISIGN_OPERATOR_NAME,                    no_write_effects},
  {SIGN_OPERATOR_NAME,                     no_write_effects},
  {DSIGN_OPERATOR_NAME,                    no_write_effects},
  {IDIM_OPERATOR_NAME,                     no_write_effects},
  {DIM_OPERATOR_NAME,                      no_write_effects},
  {DDIM_OPERATOR_NAME,                     no_write_effects},
  {DPROD_OPERATOR_NAME,                    no_write_effects},
  {MAX_OPERATOR_NAME,                      no_write_effects},
  {MAX0_OPERATOR_NAME,                     no_write_effects},
  {AMAX1_OPERATOR_NAME,                    no_write_effects},
  {DMAX1_OPERATOR_NAME,                    no_write_effects},
  {AMAX0_OPERATOR_NAME,                    no_write_effects},
  {MAX1_OPERATOR_NAME,                     no_write_effects},
  {MIN_OPERATOR_NAME,                      no_write_effects},
  {MIN0_OPERATOR_NAME,                     no_write_effects},
  {AMIN1_OPERATOR_NAME,                    no_write_effects},
  {DMIN1_OPERATOR_NAME,                    no_write_effects},
  {AMIN0_OPERATOR_NAME,                    no_write_effects},
  {MIN1_OPERATOR_NAME,                     no_write_effects},
  {PIPS_C_MIN_OPERATOR_NAME,               no_write_effects},
  {LENGTH_OPERATOR_NAME,                   no_write_effects},
  {INDEX_OPERATOR_NAME,                    no_write_effects},
  {AIMAG_CONVERSION_NAME,                  no_write_effects},
  {DIMAG_CONVERSION_NAME,                  no_write_effects},
  {CONJG_OPERATOR_NAME,                    no_write_effects},
  {DCONJG_OPERATOR_NAME,                   no_write_effects},
  {SQRT_OPERATOR_NAME,                     no_write_effects},
  {DSQRT_OPERATOR_NAME,                    no_write_effects},
  {CSQRT_OPERATOR_NAME,                    no_write_effects},

  {EXP_OPERATOR_NAME,                      no_write_effects},
  {DEXP_OPERATOR_NAME,                     no_write_effects},
  {CEXP_OPERATOR_NAME,                     no_write_effects},
  {LOG_OPERATOR_NAME,                      no_write_effects},
  {ALOG_OPERATOR_NAME,                     no_write_effects},
  {DLOG_OPERATOR_NAME,                     no_write_effects},
  {CLOG_OPERATOR_NAME,                     no_write_effects},
  {LOG10_OPERATOR_NAME,                    no_write_effects},
  {ALOG10_OPERATOR_NAME,                   no_write_effects},
  {DLOG10_OPERATOR_NAME,                   no_write_effects},
  {SIN_OPERATOR_NAME,                      no_write_effects},
  {DSIN_OPERATOR_NAME,                     no_write_effects},
  {CSIN_OPERATOR_NAME,                     no_write_effects},
  {COS_OPERATOR_NAME,                      no_write_effects},
  {DCOS_OPERATOR_NAME,                     no_write_effects},
  {CCOS_OPERATOR_NAME,                     no_write_effects},
  {TAN_OPERATOR_NAME,                      no_write_effects},
  {DTAN_OPERATOR_NAME,                     no_write_effects},
  {ASIN_OPERATOR_NAME,                     no_write_effects},
  {DASIN_OPERATOR_NAME,                    no_write_effects},
  {ACOS_OPERATOR_NAME,                     no_write_effects},
  {DACOS_OPERATOR_NAME,                    no_write_effects},
  {ATAN_OPERATOR_NAME,                     no_write_effects},
  {DATAN_OPERATOR_NAME,                    no_write_effects},
  {ATAN2_OPERATOR_NAME,                    no_write_effects},
  {DATAN2_OPERATOR_NAME,                   no_write_effects},
  {SINH_OPERATOR_NAME,                     no_write_effects},
  {DSINH_OPERATOR_NAME,                    no_write_effects},
  {COSH_OPERATOR_NAME,                     no_write_effects},
  {DCOSH_OPERATOR_NAME,                    no_write_effects},
  {TANH_OPERATOR_NAME,                     no_write_effects},
  {DTANH_OPERATOR_NAME,                    no_write_effects},

  {LGE_OPERATOR_NAME,                      no_write_effects},
  {LGT_OPERATOR_NAME,                      no_write_effects},
  {LLE_OPERATOR_NAME,                      no_write_effects},
  {LLT_OPERATOR_NAME,                      no_write_effects},

  {LIST_DIRECTED_FORMAT_NAME,              no_write_effects},
  {UNBOUNDED_DIMENSION_NAME,               no_write_effects},

  {ASSIGN_OPERATOR_NAME,                   affect_effects},

  /* Fortran IO related intrinsic */
  {WRITE_FUNCTION_NAME,                    io_effects},
  {REWIND_FUNCTION_NAME,                   io_effects},
  {BACKSPACE_FUNCTION_NAME,                io_effects},
  {OPEN_FUNCTION_NAME,                     io_effects},
  {CLOSE_FUNCTION_NAME,                    io_effects},
  {INQUIRE_FUNCTION_NAME,                  io_effects},
  {READ_FUNCTION_NAME,                     read_io_effects},
  {BUFFERIN_FUNCTION_NAME,                 c_io_effects},
  {BUFFEROUT_FUNCTION_NAME,                c_io_effects},
  {ENDFILE_FUNCTION_NAME,                  io_effects},
  {IMPLIED_DO_NAME,                        effects_of_implied_do},

  {SUBSTRING_FUNCTION_NAME,                substring_effect},
  {ASSIGN_SUBSTRING_FUNCTION_NAME,         assign_substring_effects},

  /* These operators are used within the OPTIMIZE transformation in
     order to manipulate operators such as n-ary add and multiply or
     multiply-add operators ( JZ - sept 98) */
  {EOLE_SUM_OPERATOR_NAME,                 no_write_effects },
  {EOLE_PROD_OPERATOR_NAME,                no_write_effects },
  {EOLE_FMA_OPERATOR_NAME,                 no_write_effects },

  {IMA_OPERATOR_NAME,                      no_write_effects },
  {IMS_OPERATOR_NAME,                      no_write_effects },


  /* Bit manipulation F90 functions. ISO/IEC 1539 : 1991  Amira Mensi */

  {ISHFT_OPERATOR_NAME,                    no_write_effects },
  {ISHFTC_OPERATOR_NAME,                   no_write_effects },
  {IBITS_OPERATOR_NAME,                    no_write_effects },
  {MVBITS_OPERATOR_NAME,                   no_write_effects },
  {BTEST_OPERATOR_NAME,                    no_write_effects },
  {IBCLR_OPERATOR_NAME,                    no_write_effects },
  {BIT_SIZE_OPERATOR_NAME,                 no_write_effects },
  {IBSET_OPERATOR_NAME,                    no_write_effects },
  {IAND_OPERATOR_NAME,                     no_write_effects },
  {IEOR_OPERATOR_NAME,                     no_write_effects },
  {IOR_OPERATOR_NAME,                      no_write_effects },

  /* Here are C intrinsics.*/

  /* ISO 6.5.2.3 structure and union members */
  {FIELD_OPERATOR_NAME,                    address_expression_effects},
  {POINT_TO_OPERATOR_NAME,                 address_expression_effects},

  /* ISO 6.5.2.4 postfix increment and decrement operators, real or pointer type operand */
  {POST_INCREMENT_OPERATOR_NAME,           unique_update_effects},
  {POST_DECREMENT_OPERATOR_NAME,           unique_update_effects},

  /* ISO 6.5.3.1 prefix increment and decrement operators, real or pointer type operand */
  {PRE_INCREMENT_OPERATOR_NAME,            unique_update_effects},
  {PRE_DECREMENT_OPERATOR_NAME,            unique_update_effects},

  /* ISO 6.5.3.2 address and indirection operators, add pointer type */
  {ADDRESS_OF_OPERATOR_NAME,               address_of_effects},
  {DEREFERENCING_OPERATOR_NAME,            address_expression_effects},

  /* ISO 6.5.3.3 unary arithmetic operators */
  {UNARY_PLUS_OPERATOR_NAME,               no_write_effects},
  // {"-unary",                            no_write_effects},UNARY_MINUS_OPERATOR already exist (FORTRAN)
  {BITWISE_NOT_OPERATOR_NAME,              no_write_effects},
  {C_NOT_OPERATOR_NAME,                    no_write_effects},

  {C_MODULO_OPERATOR_NAME,                 no_write_effects},

  /* ISO 6.5.6 additive operators, arithmetic types or pointer + integer type*/
  {PLUS_C_OPERATOR_NAME,                   no_write_effects},
  {MINUS_C_OPERATOR_NAME,                  no_write_effects},

  /* ISO 6.5.7 bitwise shift operators*/
  {LEFT_SHIFT_OPERATOR_NAME,               no_write_effects},
  {RIGHT_SHIFT_OPERATOR_NAME,              no_write_effects},

  /* ISO 6.5.8 relational operators,arithmetic or pointer types */
  {C_LESS_THAN_OPERATOR_NAME,              no_write_effects},
  {C_GREATER_THAN_OPERATOR_NAME,           no_write_effects},
  {C_LESS_OR_EQUAL_OPERATOR_NAME,          no_write_effects},
  {C_GREATER_OR_EQUAL_OPERATOR_NAME,       no_write_effects},

  /* ISO 6.5.9 equality operators, return 0 or 1*/
  {C_EQUAL_OPERATOR_NAME,                  no_write_effects},
  {C_NON_EQUAL_OPERATOR_NAME,              no_write_effects},

 /* ISO 6.5.10 bitwise AND operator */
  {BITWISE_AND_OPERATOR_NAME,              no_write_effects},

 /* ISO 6.5.11 bitwise exclusive OR operator */
  {BITWISE_XOR_OPERATOR_NAME,              no_write_effects},

  /* ISO 6.5.12 bitwise inclusive OR operator */
  {BITWISE_OR_OPERATOR_NAME,               no_write_effects},

  /* ISO 6.5.13 logical AND operator */
  {C_AND_OPERATOR_NAME,                    no_write_effects},

  /* ISO 6.5.14 logical OR operator */
  {C_OR_OPERATOR_NAME,                     no_write_effects},

  /* ISO 6.5.15 conditional operator */
  {CONDITIONAL_OPERATOR_NAME,              conditional_effects},

  /* ISO 6.5.16.2 compound assignments*/
  {MULTIPLY_UPDATE_OPERATOR_NAME,          update_effects},
  {DIVIDE_UPDATE_OPERATOR_NAME,            update_effects},
  {MODULO_UPDATE_OPERATOR_NAME,            update_effects},
  {PLUS_UPDATE_OPERATOR_NAME,              update_effects},
  {MINUS_UPDATE_OPERATOR_NAME,             update_effects},
  {LEFT_SHIFT_UPDATE_OPERATOR_NAME,        update_effects},
  {RIGHT_SHIFT_UPDATE_OPERATOR_NAME,       update_effects},
  {BITWISE_AND_UPDATE_OPERATOR_NAME,       update_effects},
  {BITWISE_XOR_UPDATE_OPERATOR_NAME,       update_effects},
  {BITWISE_OR_UPDATE_OPERATOR_NAME,        update_effects},

  /* ISO 6.5.17 comma operator */
  {COMMA_OPERATOR_NAME,                    no_write_effects},

  {BREAK_FUNCTION_NAME,                    no_write_effects},
  {CASE_FUNCTION_NAME,                     no_write_effects},
  {DEFAULT_FUNCTION_NAME,                  no_write_effects},
  {C_RETURN_FUNCTION_NAME,                 no_write_effects},

  /* intrinsic to handle C initialization */

  {BRACE_INTRINSIC,                        no_write_effects},


  /* assert.h */
  /* These intrinsics are added with no_write_effects to work with C.
     The real effects must be studied !!! I do not have time for the moment */

  {ASSERT_FUNCTION_NAME,                   no_write_effects},
  {ASSERT_FAIL_FUNCTION_NAME,              no_write_effects}, /* in fact, IO effect, does not return */

  /* #include <complex.h> */
  {CACOS_OPERATOR_NAME,                    no_write_effects},
  {CACOSF_OPERATOR_NAME,                   no_write_effects},
  {CACOSL_OPERATOR_NAME,                   no_write_effects},
  {CASIN_OPERATOR_NAME,                    no_write_effects},
  {CASINF_OPERATOR_NAME,                   no_write_effects},
  {CASINL_OPERATOR_NAME,                   no_write_effects},
  {CATAN_OPERATOR_NAME,                    no_write_effects},
  {CATANF_OPERATOR_NAME,                   no_write_effects},
  {CATANL_OPERATOR_NAME,                   no_write_effects},
  {C_CCOS_OPERATOR_NAME,                   no_write_effects},
  {CCOSF_OPERATOR_NAME,                    no_write_effects},
  {CCOSL_OPERATOR_NAME,                    no_write_effects},
  {C_CSIN_OPERATOR_NAME,                   no_write_effects},
  {CSINF_OPERATOR_NAME,                    no_write_effects},
  {CSINL_OPERATOR_NAME,                    no_write_effects},
  {CTAN_OPERATOR_NAME,                     no_write_effects},
  {CTANF_OPERATOR_NAME,                    no_write_effects},
  {CTANL_OPERATOR_NAME,                    no_write_effects},
  {CACOSH_OPERATOR_NAME,                   no_write_effects},
  {CACOSHF_OPERATOR_NAME,                  no_write_effects},
  {CACOSHL_OPERATOR_NAME,                  no_write_effects},
  {CASINH_OPERATOR_NAME,                   no_write_effects},
  {CASINHF_OPERATOR_NAME,                  no_write_effects},
  {CASINHL_OPERATOR_NAME,                  no_write_effects},
  {CATANH_OPERATOR_NAME,                   no_write_effects},
  {CATANHF_OPERATOR_NAME,                  no_write_effects},
  {CATANHL_OPERATOR_NAME,                  no_write_effects},
  {CCOSH_OPERATOR_NAME,                    no_write_effects},
  {CCOSHF_OPERATOR_NAME,                   no_write_effects},
  {CCOSHL_OPERATOR_NAME,                   no_write_effects},
  {CSINH_OPERATOR_NAME,                    no_write_effects},
  {CSINHF_OPERATOR_NAME,                   no_write_effects},
  {CSINHL_OPERATOR_NAME,                   no_write_effects},
  {CTANH_OPERATOR_NAME,                    no_write_effects},
  {CTANHF_OPERATOR_NAME,                   no_write_effects},
  {CTANHL_OPERATOR_NAME,                   no_write_effects},
  {C_CEXP_OPERATOR_NAME,                   no_write_effects},
  {CEXPF_OPERATOR_NAME,                    no_write_effects},
  {CEXPL_OPERATOR_NAME,                    no_write_effects},
  {C_CLOG_OPERATOR_NAME,                   no_write_effects},
  {CLOGF_OPERATOR_NAME,                    no_write_effects},
  {CLOGL_OPERATOR_NAME,                    no_write_effects},
  {C_CABS_OPERATOR_NAME,                   no_write_effects},
  {CABSF_OPERATOR_NAME,                    no_write_effects},
  {CABSL_OPERATOR_NAME,                    no_write_effects},
  {CPOW_OPERATOR_NAME,                     no_write_effects},
  {CPOWF_OPERATOR_NAME,                    no_write_effects},
  {CPOWL_OPERATOR_NAME,                    no_write_effects},
  {C_CSQRT_OPERATOR_NAME,                  no_write_effects},
  {CSQRTF_OPERATOR_NAME,                   no_write_effects},
  {CSQRTL_OPERATOR_NAME,                   no_write_effects},
  {CARG_OPERATOR_NAME,                     no_write_effects},
  {CARGF_OPERATOR_NAME,                    no_write_effects},
  {CARGL_OPERATOR_NAME,                    no_write_effects},
  {CIMAG_OPERATOR_NAME,                    no_write_effects},
  {GCC_CIMAG_OPERATOR_NAME,                no_write_effects},
  {CIMAGF_OPERATOR_NAME,                   no_write_effects},
  {CIMAGL_OPERATOR_NAME,                   no_write_effects},
  {CONJ_OPERATOR_NAME,                     no_write_effects},
  {CONJF_OPERATOR_NAME,                    no_write_effects},
  {CONJL_OPERATOR_NAME,                    no_write_effects},
  {CPROJ_OPERATOR_NAME,                    no_write_effects},
  {CPROJF_OPERATOR_NAME,                   no_write_effects},
  {CPROJL_OPERATOR_NAME,                   no_write_effects},
  {CREAL_OPERATOR_NAME,                    no_write_effects},
  {GCC_CREAL_OPERATOR_NAME,                no_write_effects},
  {CREALF_OPERATOR_NAME,                   no_write_effects},
  {CREALL_OPERATOR_NAME,                   no_write_effects},

  /* #include <ctype.h>*/

  {ISALNUM_OPERATOR_NAME,                  no_write_effects},
  {ISALPHA_OPERATOR_NAME,                  no_write_effects},
  {ISBLANK_OPERATOR_NAME,                  no_write_effects},
  {ISCNTRL_OPERATOR_NAME,                  no_write_effects},
  {ISDIGIT_OPERATOR_NAME,                  no_write_effects},
  {ISGRAPH_OPERATOR_NAME,                  no_write_effects},
  {ISLOWER_OPERATOR_NAME,                  no_write_effects},
  {ISPRINT_OPERATOR_NAME,                  no_write_effects},
  {ISPUNCT_OPERATOR_NAME,                  no_write_effects},
  {ISSPACE_OPERATOR_NAME,                  no_write_effects},
  {ISUPPER_OPERATOR_NAME,                  no_write_effects},
  {ISXDIGIT_OPERATOR_NAME,                 no_write_effects},
  {TOLOWER_OPERATOR_NAME,                  no_write_effects},
  {TOUPPER_OPERATOR_NAME,                  no_write_effects},

  /* errno.h */
  // MB: errno is usually an extern int variable, but *errno() is allowed (ISO section 7.5 in C99)
  {"errno",                                no_write_effects},

  /* fenv.h */

  {FECLEAREXCEPT_FUNCTION_NAME,            no_write_effects},
  {FERAISEEXCEPT_FUNCTION_NAME,            no_write_effects},
  {FESETEXCEPTFLAG_FUNCTION_NAME,          no_write_effects},
  {FETESTEXCEPT_FUNCTION_NAME,             no_write_effects},
  {FEGETROUND_FUNCTION_NAME,               no_write_effects},
  {FESETROUND_FUNCTION_NAME,               no_write_effects},
  // fenv_t *
  // {FESETENV_FUNCTION_NAME,                 no_write_effects},
  //{FEUPDATEENV_FUNCTION_NAME,              no_write_effects},


  /* inttypes.h */
  {IMAXABS_FUNCTION_NAME,                  no_write_effects},
  {IMAXDIV_FUNCTION_NAME,                  no_write_effects},

  /* locale.h */
  {SETLOCALE_FUNCTION_NAME,                no_write_effects},
  {"localeconv",                           no_write_effects},

  /* #include <math.h>*/

  {FPCLASSIFY_OPERATOR_NAME,               no_write_effects},
  {ISFINITE_OPERATOR_NAME,                 no_write_effects},
  {ISINF_OPERATOR_NAME,                    no_write_effects},
  {ISNAN_OPERATOR_NAME,                    no_write_effects},
  {ISNANL_OPERATOR_NAME,                   no_write_effects},
  {ISNANF_OPERATOR_NAME,                   no_write_effects},
  {ISNORMAL_OPERATOR_NAME,                 no_write_effects},
  {SIGNBIT_OPERATOR_NAME,                  no_write_effects},
  {C_ACOS_OPERATOR_NAME,                   no_write_effects},
  {ACOSF_OPERATOR_NAME,                    no_write_effects},
  {ACOSL_OPERATOR_NAME,                    no_write_effects},
  {C_ASIN_OPERATOR_NAME,                   no_write_effects},
  {ASINF_OPERATOR_NAME,                    no_write_effects},
  {ASINL_OPERATOR_NAME,                    no_write_effects},
  {C_ATAN_OPERATOR_NAME,                   no_write_effects},
  {ATANF_OPERATOR_NAME,                    no_write_effects},
  {ATANL_OPERATOR_NAME,                    no_write_effects},
  {C_ATAN2_OPERATOR_NAME,                  no_write_effects},
  {ATAN2F_OPERATOR_NAME,                   no_write_effects},
  {ATAN2L_OPERATOR_NAME,                   no_write_effects},
  {C_COS_OPERATOR_NAME,                    no_write_effects},
  {COSF_OPERATOR_NAME,                     no_write_effects},
  {COSL_OPERATOR_NAME,                     no_write_effects},
  {C_SIN_OPERATOR_NAME,                    no_write_effects},
  {SINF_OPERATOR_NAME,                     no_write_effects},
  {SINL_OPERATOR_NAME,                     no_write_effects},
  {C_TAN_OPERATOR_NAME,                    no_write_effects},
  {TANF_OPERATOR_NAME,                     no_write_effects},
  {TANL_OPERATOR_NAME,                     no_write_effects},
  {C_ACOSH_OPERATOR_NAME ,                 no_write_effects},
  {ACOSHF_OPERATOR_NAME ,                  no_write_effects},
  {ACOSHL_OPERATOR_NAME ,                  no_write_effects},
  {C_ASINH_OPERATOR_NAME,                  no_write_effects},
  {ASINHF_OPERATOR_NAME,                   no_write_effects},
  {ASINHL_OPERATOR_NAME,                   no_write_effects},
  {C_ATANH_OPERATOR_NAME,                  no_write_effects},
  {ATANHF_OPERATOR_NAME,                   no_write_effects},
  {ATANHL_OPERATOR_NAME,                   no_write_effects},
  {C_COSH_OPERATOR_NAME,                   no_write_effects},
  {COSHF_OPERATOR_NAME,                    no_write_effects},
  {COSHL_OPERATOR_NAME,                    no_write_effects},
  {C_SINH_OPERATOR_NAME,                   no_write_effects},
  {SINHF_OPERATOR_NAME,                    no_write_effects},
  {SINHL_OPERATOR_NAME,                    no_write_effects},
  {C_TANH_OPERATOR_NAME,                   no_write_effects},
  {TANHF_OPERATOR_NAME,                    no_write_effects},
  {TANHL_OPERATOR_NAME,                    no_write_effects},
  {C_EXP_OPERATOR_NAME,                    no_write_effects},
  {EXPF_OPERATOR_NAME,                     no_write_effects},
  {EXPL_OPERATOR_NAME,                     no_write_effects},
  {EXP2_OPERATOR_NAME,                     no_write_effects},
  {EXP2F_OPERATOR_NAME,                    no_write_effects},
  {EXP2L_OPERATOR_NAME,                    no_write_effects},
  {EXPM1_OPERATOR_NAME,                    no_write_effects},
  {EXPM1F_OPERATOR_NAME,                   no_write_effects},
  {EXPM1L_OPERATOR_NAME,                   no_write_effects},
  //frexp has a write effect not defined correctly. MB
  {FREXP_OPERATOR_NAME,                    no_write_effects},
  {ILOGB_OPERATOR_NAME,                    no_write_effects},
  {ILOGBF_OPERATOR_NAME,                   no_write_effects},
  {ILOGBL_OPERATOR_NAME,                   no_write_effects},
  {LDEXP_OPERATOR_NAME,                    no_write_effects},
  {LDEXPF_OPERATOR_NAME,                   no_write_effects},
  {LDEXPL_OPERATOR_NAME,                   no_write_effects},
  {C_LOG_OPERATOR_NAME,                    no_write_effects},
  {LOGF_OPERATOR_NAME,                     no_write_effects},
  {LOGL_OPERATOR_NAME,                     no_write_effects},
  {C_LOG10_OPERATOR_NAME,                  no_write_effects},
  {LOG10F_OPERATOR_NAME,                   no_write_effects},
  {LOG10L_OPERATOR_NAME,                   no_write_effects},
  {LOG1P_OPERATOR_NAME,                    no_write_effects},
  {LOG1PF_OPERATOR_NAME,                   no_write_effects},
  {LOG1PL_OPERATOR_NAME,                   no_write_effects},
  {LOG2_OPERATOR_NAME,                     no_write_effects},
  {LOG2F_OPERATOR_NAME,                    no_write_effects},
  {LOG2L_OPERATOR_NAME,                    no_write_effects},
  {LOGB_OPERATOR_NAME,                     no_write_effects},
  {LOGBF_OPERATOR_NAME,                    no_write_effects},
  {LOGBL_OPERATOR_NAME,                    no_write_effects},
  //modf & modff have write effects not defined correctly. MB
  {MODF_OPERATOR_NAME,                     no_write_effects},
  {MODFF_OPERATOR_NAME,                    no_write_effects},
  {SCALBN_OPERATOR_NAME,                   no_write_effects},
  {SCALBNF_OPERATOR_NAME,                  no_write_effects},
  {SCALBNL_OPERATOR_NAME,                  no_write_effects},
  {SCALB_OPERATOR_NAME,                    no_write_effects}, /* POSIX.1-2001, The scalb function is the BSD name for ldexp */
  {SCALBLN_OPERATOR_NAME,                  no_write_effects},
  {SCALBLNF_OPERATOR_NAME,                 no_write_effects},
  {SCALBLNL_OPERATOR_NAME,                 no_write_effects},
  {CBRT_OPERATOR_NAME,                     no_write_effects},
  {CBRTF_OPERATOR_NAME,                    no_write_effects},
  {CBRTL_OPERATOR_NAME,                    no_write_effects},
  {FABS_OPERATOR_NAME,                     no_write_effects},
  {FABSF_OPERATOR_NAME,                    no_write_effects},
  {FABSL_OPERATOR_NAME,                    no_write_effects},
  {HYPOT_OPERATOR_NAME,                    no_write_effects},
  {HYPOTF_OPERATOR_NAME,                   no_write_effects},
  {HYPOTL_OPERATOR_NAME,                   no_write_effects},
  {POW_OPERATOR_NAME,                      no_write_effects},
  {POWF_OPERATOR_NAME,                     no_write_effects},
  {POWL_OPERATOR_NAME,                     no_write_effects},
  {C_SQRT_OPERATOR_NAME,                   no_write_effects},
  {SQRTF_OPERATOR_NAME,                    no_write_effects},
  {SQRTL_OPERATOR_NAME,                    no_write_effects},
  {ERF_OPERATOR_NAME,                      no_write_effects},
  {ERFF_OPERATOR_NAME,                     no_write_effects},
  {ERFL_OPERATOR_NAME,                     no_write_effects},
  {ERFC_OPERATOR_NAME,                     no_write_effects},
  {ERFCF_OPERATOR_NAME,                    no_write_effects},
  {ERFCL_OPERATOR_NAME,                    no_write_effects},
  {GAMMA_OPERATOR_NAME,                    no_write_effects}, /* GNU C Library */
  {LGAMMA_OPERATOR_NAME,                   no_write_effects},
  {LGAMMAF_OPERATOR_NAME,                  no_write_effects},
  {LGAMMAL_OPERATOR_NAME,                  no_write_effects},
  {TGAMMA_OPERATOR_NAME,                   no_write_effects},
  {TGAMMAF_OPERATOR_NAME,                  no_write_effects},
  {TGAMMAL_OPERATOR_NAME,                  no_write_effects},
  {CEIL_OPERATOR_NAME,                     no_write_effects},
  {CEILF_OPERATOR_NAME,                    no_write_effects},
  {CEILL_OPERATOR_NAME,                    no_write_effects},
  {FLOOR_OPERATOR_NAME,                    no_write_effects},
  {FLOORF_OPERATOR_NAME,                   no_write_effects},
  {FLOORL_OPERATOR_NAME,                   no_write_effects},
  {NEARBYINT_OPERATOR_NAME,                no_write_effects},
  {NEARBYINTF_OPERATOR_NAME,               no_write_effects},
  {NEARBYINTL_OPERATOR_NAME,               no_write_effects},
  {RINT_OPERATOR_NAME,                     no_write_effects},
  {RINTF_OPERATOR_NAME,                    no_write_effects},
  {RINTL_OPERATOR_NAME,                    no_write_effects},
  {LRINT_OPERATOR_NAME,                    no_write_effects},
  {LRINTF_OPERATOR_NAME,                   no_write_effects},
  {LRINTL_OPERATOR_NAME,                   no_write_effects},
  {LLRINT_OPERATOR_NAME,                   no_write_effects},
  {LLRINTF_OPERATOR_NAME,                  no_write_effects},
  {LLRINTL_OPERATOR_NAME,                  no_write_effects},
  {ROUND_OPERATOR_NAME,                    no_write_effects},
  {ROUNDF_OPERATOR_NAME,                   no_write_effects},
  {ROUNDL_OPERATOR_NAME,                   no_write_effects},
  {LROUND_OPERATOR_NAME,                   no_write_effects},
  {LROUNDF_OPERATOR_NAME,                  no_write_effects},
  {LROUNDL_OPERATOR_NAME,                  no_write_effects},
  {LLROUND_OPERATOR_NAME,                   no_write_effects},
  {LLROUNDF_OPERATOR_NAME,                  no_write_effects},
  {LLROUNDL_OPERATOR_NAME,                  no_write_effects},
  {TRUNC_OPERATOR_NAME,                    no_write_effects},
  {TRUNCF_OPERATOR_NAME,                   no_write_effects},
  {TRUNCL_OPERATOR_NAME,                   no_write_effects},
  {FMOD_OPERATOR_NAME,                     no_write_effects},
  {FMODF_OPERATOR_NAME,                    no_write_effects},
  {FMODL_OPERATOR_NAME,                    no_write_effects},
  {REMAINDER_OPERATOR_NAME,                no_write_effects},
  {REMAINDERF_OPERATOR_NAME,               no_write_effects},
  {REMAINDERL_OPERATOR_NAME,               no_write_effects},
  {COPYSIGN_OPERATOR_NAME,                 no_write_effects},
  {COPYSIGNF_OPERATOR_NAME,                no_write_effects},
  {COPYSIGNL_OPERATOR_NAME,                no_write_effects},
  {NAN_OPERATOR_NAME,                      no_write_effects},
  {NANF_OPERATOR_NAME,                     no_write_effects},
  {NANL_OPERATOR_NAME,                     no_write_effects},
  {NEXTAFTER_OPERATOR_NAME,                no_write_effects},
  {NEXTAFTERF_OPERATOR_NAME,               no_write_effects},
  {NEXTAFTERL_OPERATOR_NAME,               no_write_effects},
  {NEXTTOWARD_OPERATOR_NAME,               no_write_effects},
  {NEXTTOWARDF_OPERATOR_NAME,              no_write_effects},
  {NEXTTOWARDL_OPERATOR_NAME,              no_write_effects},
  {FDIM_OPERATOR_NAME,                     no_write_effects},
  {FDIMF_OPERATOR_NAME,                    no_write_effects},
  {FDIML_OPERATOR_NAME,                    no_write_effects},
  {FMAX_OPERATOR_NAME,                     no_write_effects},
  {FMAXF_OPERATOR_NAME,                    no_write_effects},
  {FMAXL_OPERATOR_NAME,                    no_write_effects},
  {FMIN_OPERATOR_NAME,                     no_write_effects},
  {FMINF_OPERATOR_NAME,                    no_write_effects},
  {FMINL_OPERATOR_NAME,                    no_write_effects},
  {FMA_OPERATOR_NAME,                      no_write_effects},
  {FMAF_OPERATOR_NAME,                     no_write_effects},
  {FMAL_OPERATOR_NAME,                     no_write_effects},
  {ISGREATER_OPERATOR_NAME,                no_write_effects},
  {ISGREATEREQUAL_OPERATOR_NAME,           no_write_effects},
  {ISLESS_OPERATOR_NAME,                   no_write_effects},
  {ISLESSEQUAL_OPERATOR_NAME,              no_write_effects},
  {ISLESSGREATER_OPERATOR_NAME,            no_write_effects},
  {ISUNORDERED_OPERATOR_NAME,              no_write_effects},


  /*#include <setjmp.h>*/

  {"setjmp",                               no_write_effects},
  {"__setjmp",                             no_write_effects},
  {"longjmp",                              no_write_effects}, // control effect 7.13 in C99
  {"__longjmp",                            no_write_effects},
  {"sigsetjmp",                            no_write_effects}, //POSIX.1-2001
  {"siglongjmp",                           no_write_effects}, //POSIX.1-2001


  /* signal.h 7.14 */
  {SIGFPE_OPERATOR_NAME,                   no_write_effects}, //macro
  {SIGNAL_OPERATOR_NAME,                   no_write_effects},
  {RAISE_FUNCTION_NAME,                    no_write_effects},


  /* stdarg.h */

  {BUILTIN_VA_START,                       va_list_effects},
  {BUILTIN_VA_END,                         va_list_effects},
  {BUILTIN_VA_COPY,                        va_list_effects},
  /* va_arg is not a standard call; it is directly represented in PIPS
     internal representation. */

  /*#include <stdio.h>*/
  // IO functions
  {REMOVE_FUNCTION_NAME,                   no_write_effects},
  {RENAME_FUNCTION_NAME,                   no_write_effects},
  {TMPFILE_FUNCTION_NAME,                  no_write_effects},
  {TMPNAM_FUNCTION_NAME,                   no_write_effects},
  {FCLOSE_FUNCTION_NAME,                   c_io_effects},
  {FFLUSH_FUNCTION_NAME,                   no_write_effects},
  {FOPEN_FUNCTION_NAME,                    c_io_effects},
  {FREOPEN_FUNCTION_NAME,                  no_write_effects},
  {SETBUF_FUNCTION_NAME,                   no_write_effects},
  {SETVBUF_FUNCTION_NAME ,                 no_write_effects},
  {FPRINTF_FUNCTION_NAME,                  c_io_effects},
  {FSCANF_FUNCTION_NAME,                   c_io_effects},
  {ISOC99_FSCANF_FUNCTION_NAME,            c_io_effects},
  {PRINTF_FUNCTION_NAME,                   c_io_effects},
  {SCANF_FUNCTION_NAME,                    c_io_effects},
  {ISOC99_SCANF_FUNCTION_NAME,             c_io_effects},
  {SNPRINTF_FUNCTION_NAME,                 c_io_effects},
  {SPRINTF_FUNCTION_NAME,                  c_io_effects},
  {SSCANF_FUNCTION_NAME,                   c_io_effects},
  {ISOC99_SSCANF_FUNCTION_NAME,            c_io_effects},
  {VFPRINTF_FUNCTION_NAME,                 c_io_effects},
  {VFSCANF_FUNCTION_NAME,                  c_io_effects},
  {ISOC99_VFSCANF_FUNCTION_NAME,           c_io_effects},
  {VPRINTF_FUNCTION_NAME,                  c_io_effects},
  {VSCANF_FUNCTION_NAME,                   c_io_effects},
  {ISOC99_VSCANF_FUNCTION_NAME,            c_io_effects},
  {VSNPRINTF_FUNCTION_NAME,                c_io_effects},
  {VSPRINTF_FUNCTION_NAME,                 c_io_effects},
  {VSSCANF_FUNCTION_NAME,                  c_io_effects},
  {ISOC99_VSSCANF_FUNCTION_NAME,           c_io_effects},
  {FGETC_FUNCTION_NAME,                    c_io_effects},
  {FGETS_FUNCTION_NAME,                    c_io_effects},
  {FPUTC_FUNCTION_NAME,                    c_io_effects},
  {FPUTS_FUNCTION_NAME,                    c_io_effects},
  {GETC_FUNCTION_NAME,                     c_io_effects},
  {_IO_GETC_FUNCTION_NAME,                 c_io_effects},
  {GETCHAR_FUNCTION_NAME,                  c_io_effects},
  {GETS_FUNCTION_NAME,                     c_io_effects},
  {PUTC_FUNCTION_NAME,                     c_io_effects},
  {_IO_PUTC_FUNCTION_NAME,                 c_io_effects},
  {PUTCHAR_FUNCTION_NAME,                  c_io_effects},
  {PUTS_FUNCTION_NAME,                     c_io_effects},
  {UNGETC_FUNCTION_NAME,                   c_io_effects},
  {FREAD_FUNCTION_NAME,                    c_io_effects},
  {FWRITE_FUNCTION_NAME,                   c_io_effects},
  {FGETPOS_FUNCTION_NAME,                  c_io_effects},
  {FSEEK_FUNCTION_NAME,                    c_io_effects},
  {FSETPOS_FUNCTION_NAME,                  c_io_effects},
  {FTELL_FUNCTION_NAME,                    c_io_effects},
  {C_REWIND_FUNCTION_NAME,                 c_io_effects},
  {CLEARERR_FUNCTION_NAME,                 c_io_effects},
  {FEOF_FUNCTION_NAME,                     c_io_effects},
  {FERROR_FUNCTION_NAME,                   c_io_effects},
  {PERROR_FUNCTION_NAME,                   c_io_effects},


  /* #include <stdlib.h> */
  {ATOF_FUNCTION_NAME,                     safe_c_read_only_effects},
  {ATOI_FUNCTION_NAME,                     safe_c_read_only_effects},
  {ATOL_FUNCTION_NAME,                     safe_c_read_only_effects},
  {ATOLL_FUNCTION_NAME,                    safe_c_read_only_effects},
  {STRTOD_FUNCTION_NAME,                   strtoxxx_effects},
  {STRTOF_FUNCTION_NAME,                   strtoxxx_effects},
  {STRTOL_FUNCTION_NAME,                   strtoxxx_effects},
  {STRTOLL_FUNCTION_NAME,                  strtoxxx_effects},
  {STRTOUL_FUNCTION_NAME,                  strtoxxx_effects},
  {STRTOULL_FUNCTION_NAME,                 strtoxxx_effects},
  {RAND_FUNCTION_NAME,                     rgs_effects},
  {SRAND_FUNCTION_NAME,                    rgsi_effects},
  {CALLOC_FUNCTION_NAME,                   no_write_effects},
  {FREE_FUNCTION_NAME,                     any_heap_effects},
  {MALLOC_FUNCTION_NAME,                   any_heap_effects},
  {REALLOC_FUNCTION_NAME,                  any_heap_effects},
  /* SG: I am setting an any_heap_effects for alloca, which is over pessimistic ... */
  {ALLOCA_FUNCTION_NAME,                   any_heap_effects},
  {ABORT_FUNCTION_NAME,                    no_write_effects},
  {ATEXIT_FUNCTION_NAME,                   no_write_effects},
  {EXIT_FUNCTION_NAME,                     no_write_effects},
  {_EXIT_FUNCTION_NAME,                    no_write_effects},
  {GETENV_FUNCTION_NAME,                   no_write_effects},
  {SYSTEM_FUNCTION_NAME,                   no_write_effects},
  {BSEARCH_FUNCTION_NAME,                  search_or_sort_effects},
  {QSORT_FUNCTION_NAME,                    search_or_sort_effects},
  {C_ABS_FUNCTION_NAME,                    no_write_effects},
  {LABS_FUNCTION_NAME,                     no_write_effects},
  {LLABS_FUNCTION_NAME,                    no_write_effects},
  {DIV_FUNCTION_NAME,                      no_write_effects},
  {LDIV_FUNCTION_NAME,                     no_write_effects},
  {LLDIV_FUNCTION_NAME,                    no_write_effects},
  {MBLEN_FUNCTION_NAME,                    no_write_effects},
  {MBTOWC_FUNCTION_NAME,                   no_write_effects},
  {WCTOMB_FUNCTION_NAME,                   no_write_effects},
  {MBSTOWCS_FUNCTION_NAME,                 no_write_effects},
  {WCSTOMBS_FUNCTION_NAME,                 no_write_effects},

   /*#include <string.h>*/

  {MEMCPY_FUNCTION_NAME,                   generic_string_effects},
  {MEMMOVE_FUNCTION_NAME,                  generic_string_effects},
  {STRCPY_FUNCTION_NAME,                   generic_string_effects},
  {STRDUP_FUNCTION_NAME,                   strdup_effects}, /* according to man page,  _BSD_SOURCE || _POSIX_C_SOURCE >= 200809L */
  {STRNCPY_FUNCTION_NAME,                  generic_string_effects},
  {STRCAT_FUNCTION_NAME,                   generic_string_effects},
  {STRNCAT_FUNCTION_NAME,                  generic_string_effects},
  {MEMCMP_FUNCTION_NAME,                   no_write_effects},
  {STRCMP_FUNCTION_NAME,                   no_write_effects},
  {STRCOLL_FUNCTION_NAME,                  no_write_effects},
  {STRNCMP_FUNCTION_NAME,                  no_write_effects},
  {STRXFRM_FUNCTION_NAME,                  generic_string_effects},
  {MEMCHR_FUNCTION_NAME,                   no_write_effects},
  {STRCHR_FUNCTION_NAME,                   no_write_effects},
  {STRCSPN_FUNCTION_NAME,                  no_write_effects},
  {STRPBRK_FUNCTION_NAME,                  no_write_effects},
  {STRRCHR_FUNCTION_NAME,                  no_write_effects},
  {STRSPN_FUNCTION_NAME,                   no_write_effects},
  {STRSTR_FUNCTION_NAME,                   no_write_effects},
  {STRTOK_FUNCTION_NAME,                   no_write_effects},
  {MEMSET_FUNCTION_NAME,                   generic_string_effects},
  {STRERROR_FUNCTION_NAME,                 no_write_effects},
  {STRERROR_R_FUNCTION_NAME,               no_write_effects},
  {STRLEN_FUNCTION_NAME,                   safe_c_read_only_effects},

  /*#include <time.h>*/
  {TIME_FUNCTION_NAME,                     time_effects},
  {LOCALTIME_FUNCTION_NAME,                time_effects},
  {DIFFTIME_FUNCTION_NAME,                 no_write_effects},
  {GETTIMEOFDAY_FUNCTION_NAME,             time_effects},
  {CLOCK_GETTIME_FUNCTION_NAME,            time_effects},
  {CLOCK_FUNCTION_NAME,                    time_effects},
  {SECOND_FUNCTION_NAME,                   time_effects}, //gfortran extension

  /*#include <wchar.h>*/
  {FWSCANF_FUNCTION_NAME,                  c_io_effects},
  {SWSCANF_FUNCTION_NAME,                  c_io_effects},
  {WSCANF_FUNCTION_NAME,                   c_io_effects},

  /* #include <wctype.h> */
  {ISWALNUM_OPERATOR_NAME,                 no_write_effects},
  {ISWALPHA_OPERATOR_NAME,                 no_write_effects},
  {ISWBLANK_OPERATOR_NAME,                 no_write_effects},
  {ISWCNTRL_OPERATOR_NAME,                 no_write_effects},
  {ISWDIGIT_OPERATOR_NAME,                 no_write_effects},
  {ISWGRAPH_OPERATOR_NAME,                 no_write_effects},
  {ISWLOWER_OPERATOR_NAME,                 no_write_effects},
  {ISWPRINT_OPERATOR_NAME,                 no_write_effects},
  {ISWPUNCT_OPERATOR_NAME,                 no_write_effects},
  {ISWSPACE_OPERATOR_NAME,                 no_write_effects},
  {ISWUPPER_OPERATOR_NAME,                 no_write_effects},
  {ISWXDIGIT_OPERATOR_NAME,                no_write_effects},
  {ISWCTYPE_OPERATOR_NAME,                 no_write_effects},
  {WCTYPE_OPERATOR_NAME,                   no_write_effects},
  {TOWLOWER_OPERATOR_NAME,                 no_write_effects},
  {TOWUPPER_OPERATOR_NAME,                 no_write_effects},
  {TOWCTRANS_OPERATOR_NAME,                no_write_effects},
  {WCTRANS_OPERATOR_NAME,                  no_write_effects},



  //not found in standard C99 (in GNU C Library)
  {ISASCII_OPERATOR_NAME,                  no_write_effects}, //This function is a BSD extension and is also an SVID extension.
  {TOASCII_OPERATOR_NAME,                  no_write_effects}, //This function is a BSD extension and is also an SVID extension.
  {_TOLOWER_OPERATOR_NAME,                 no_write_effects}, //This function is provided for compatibility with the SVID
  {_TOUPPER_OPERATOR_NAME,                 no_write_effects}, //This function is provided for compatibility with the SVID

  /* Part of the binary standard */
  {CTYPE_B_LOC_OPERATOR_NAME,              no_write_effects},



  {"__flt_rounds",                         no_write_effects},

  {"_sysconf",                             no_write_effects},
  {"wdinit",                               no_write_effects},
  {"wdchkind",                             no_write_effects},
  {"wdbindf",                              no_write_effects},
  {"wddelim",                              no_write_effects},
  {"mcfiller",                             no_write_effects},
  {"mcwrap",                               no_write_effects},

  //GNU C Library
  {"dcgettext",                            no_write_effects},
  {"dgettext",                             no_write_effects},
  {"gettext",                              no_write_effects},
  {"textdomain",                           no_write_effects},
  {"bindtextdomain",                       no_write_effects},


  /* not found in C99 standard (in GNU C Library) */

  {J0_OPERATOR_NAME,                       no_write_effects},
  {J1_OPERATOR_NAME,                       no_write_effects},
  {JN_OPERATOR_NAME,                       no_write_effects},
  {Y0_OPERATOR_NAME,                       no_write_effects},
  {Y1_OPERATOR_NAME,                       no_write_effects},
  {YN_OPERATOR_NAME,                       no_write_effects},

  //In the System V math library
  {MATHERR_OPERATOR_NAME,                  no_write_effects},

  //This function exists mainly for use in certain standardized tests of IEEE 754 conformance.
  {SIGNIFICAND_OPERATOR_NAME,              no_write_effects},

  /* netdb.h not in C99 standard (in GNU library) */
  {__H_ERRNO_LOCATION_OPERATOR_NAME,       no_write_effects},

  /* bits/errno.h */
  {__ERRNO_LOCATION_OPERATOR_NAME,         no_write_effects},


  //Posix LEGACY Std 1003.1
  {ECVT_FUNCTION_NAME,                     no_write_effects},
  {FCVT_FUNCTION_NAME,                     no_write_effects},
  {GCVT_FUNCTION_NAME,                     no_write_effects},


  /* Random number generators in stdlib.h Conforming to SVr4, POSIX.1-2001 but not in C99 */

  {RANDOM_FUNCTION_NAME,                   rgs_effects},
  {SRANDOM_FUNCTION_NAME,                  rgsi_effects},
  {DRAND48_FUNCTION_NAME,                  rgs_effects},
  {ERAND48_FUNCTION_NAME,                  rgs_effects},
  {JRAND48_FUNCTION_NAME,                  rgs_effects},
  {LRAND48_FUNCTION_NAME,                  rgs_effects},
  {MRAND48_FUNCTION_NAME,                  rgs_effects},
  {NRAND48_FUNCTION_NAME,                  rgs_effects},
  {SRAND48_FUNCTION_NAME,                  rgsi_effects},
  {SEED48_FUNCTION_NAME,                   rgsi_effects},
  {LCONG48_FUNCTION_NAME,                  rgsi_effects},


  //Posix
  {POSIX_MEMALIGN_FUNCTION_NAME,           any_heap_effects},
  {MEMALIGN_FUNCTION_NAME,                 any_heap_effects},
  {ATOQ_FUNCTION_NAME,                     no_write_effects},
  {LLTOSTR_FUNCTION_NAME,                  safe_c_effects},
  {ULLTOSTR_FUNCTION_NAME,                 no_write_effects},

  //MB: not found in C99 standard. POSIX.2
  {__FILBUF_FUNCTION_NAME,                 no_write_effects},
  {__FILSBUF_FUNCTION_NAME,                no_write_effects},
  {SETBUFFER_FUNCTION_NAME,                no_write_effects},
  {SETLINEBUF_FUNCTION_NAME,               no_write_effects},
  {FDOPEN_FUNCTION_NAME,                   no_write_effects},
  {CTERMID_FUNCTION_NAME,                  no_write_effects},
  {FILENO_FUNCTION_NAME,                   no_write_effects},
  {POPEN_FUNCTION_NAME,                    no_write_effects},
  {CUSERID_FUNCTION_NAME,                  no_write_effects},
  {TEMPNAM_FUNCTION_NAME,                  no_write_effects},
  {GETOPT_FUNCTION_NAME,                   no_write_effects}, // unistd.h,
							      // side effects
  {GETOPT_LONG_FUNCTION_NAME,              no_write_effects}, // getopt.h,
							      // side effects
  {GETOPT_LONG_ONLY_FUNCTION_NAME,         no_write_effects}, // same
							      // as above
  {GETSUBOPT_FUNCTION_NAME,                no_write_effects},
  {GETW_FUNCTION_NAME,                     no_write_effects},
  {PUTW_FUNCTION_NAME,                     no_write_effects},
  {PCLOSE_FUNCTION_NAME,                   no_write_effects},
  {FSEEKO_FUNCTION_NAME,                   no_write_effects},
  {FTELLO_FUNCTION_NAME,                   no_write_effects},
  {FOPEN64_FUNCTION_NAME,                  no_write_effects},
  {FREOPEN64_FUNCTION_NAME,                no_write_effects},
  {TMPFILE64_FUNCTION_NAME,                no_write_effects},
  {FGETPOS64_FUNCTION_NAME,                no_write_effects},
  {FSETPOS64_FUNCTION_NAME,                no_write_effects},
  {FSEEKO64_FUNCTION_NAME,                 no_write_effects},
  {FTELLO64_FUNCTION_NAME,                 no_write_effects},

  //MB: not found in C99
  {SINGLE_TO_DECIMAL_OPERATOR_NAME,        no_write_effects},
  {DOUBLE_TO_DECIMAL_OPERATOR_NAME,        no_write_effects},
  {EXTENDED_TO_DECIMAL_OPERATOR_NAME,      no_write_effects},
  {QUADRUPLE_TO_DECIMAL_OPERATOR_NAME,     no_write_effects},
  {DECIMAL_TO_SINGLE_OPERATOR_NAME,        no_write_effects},
  {DECIMAL_TO_DOUBLE_OPERATOR_NAME,        no_write_effects},
  {DECIMAL_TO_EXTENDED_OPERATOR_NAME,      no_write_effects},
  {DECIMAL_TO_QUADRUPLE_OPERATOR_NAME,     no_write_effects},
  {STRING_TO_DECIMAL_OPERATOR_NAME,        no_write_effects},
  {FUNC_TO_DECIMAL_OPERATOR_NAME,          no_write_effects},
  {FILE_TO_DECIMAL_OPERATOR_NAME,          no_write_effects},
  //3bsd
  {SECONVERT_OPERATOR_NAME,                no_write_effects},
  {SFCONVERT_OPERATOR_NAME,                no_write_effects},
  {SGCONVERT_OPERATOR_NAME,                no_write_effects},
  {ECONVERT_OPERATOR_NAME,                 no_write_effects},
  {FCONVERT_OPERATOR_NAME,                 no_write_effects},
  {GCONVERT_OPERATOR_NAME,                 no_write_effects},
  {QECONVERT_OPERATOR_NAME,                no_write_effects},
  {QFCONVERT_OPERATOR_NAME,                no_write_effects},
  {QGCONVERT_OPERATOR_NAME,                no_write_effects},

  /* C IO system functions in man -S 2 unistd.h */

  {C_OPEN_FUNCTION_NAME,                   unix_io_effects},
  {CREAT_FUNCTION_NAME,                    unix_io_effects},
  {C_CLOSE_FUNCTION_NAME,                  unix_io_effects},
  {C_WRITE_FUNCTION_NAME,                  unix_io_effects},
  {C_READ_FUNCTION_NAME,                   unix_io_effects},
  {LINK_FUNCTION_NAME,                     c_io_effects},
  {SYMLINK_FUNCTION_NAME,                  c_io_effects},
  {UNLINK_FUNCTION_NAME,                   c_io_effects},

  {FCNTL_FUNCTION_NAME,                    unix_io_effects},
  {FSYNC_FUNCTION_NAME,                    unix_io_effects},
  {FDATASYNC_FUNCTION_NAME,                unix_io_effects},
  {IOCTL_FUNCTION_NAME,                    unix_io_effects},
  {SELECT_FUNCTION_NAME,                   unix_io_effects},
  {PSELECT_FUNCTION_NAME,                  unix_io_effects},
  {STAT_FUNCTION_NAME,                     no_write_effects}, /* sys/stat.h */
  {FSTAT_FUNCTION_NAME,                    unix_io_effects},
  {LSTAT_FUNCTION_NAME,                    no_write_effects},



  /*  {char *getenv(const char *, 0, 0},
      {long int labs(long, 0, 0},
      {ldiv_t ldiv(long, long, 0, 0},*/

  /* F95 */
  {ALLOCATE_FUNCTION_NAME,                 any_heap_effects},
  {DEALLOCATE_FUNCTION_NAME,               any_heap_effects},
  {ETIME_FUNCTION_NAME,                    no_write_effects},
  {DTIME_FUNCTION_NAME,                    no_write_effects},
  {CPU_TIME_FUNCTION_NAME,                 no_write_effects},

  /* F2003 */
  {C_LOC_FUNCTION_NAME,                    no_write_effects},

  /* BSD <err.h> */
  /* SG: concerning the err* family of functions, they also exit() from the program
   * This is not represented in the EXIT_FUNCTION_NAME description, so neither it is here
   * but it seems an error to me */
  {ERR_FUNCTION_NAME,                      c_io_effects},
  {ERRX_FUNCTION_NAME,                     c_io_effects},
  {WARN_FUNCTION_NAME,                     c_io_effects},
  {WARNX_FUNCTION_NAME,                    c_io_effects},
  {VERR_FUNCTION_NAME,                     c_io_effects},
  {VERRX_FUNCTION_NAME,                    c_io_effects},
  {VWARN_FUNCTION_NAME,                    c_io_effects},
  {VWARNX_FUNCTION_NAME,                   c_io_effects},

  /*Conforming to 4.3BSD, POSIX.1-2001.*/
  /* POSIX.1-2001 declares this function obsolete; use nanosleep(2) instead.*/
  /*POSIX.1-2008 removes the specification of usleep()*/
  {USLEEP_FUNCTION_NAME,                    no_write_effects},

  /* _POSIX_C_SOURCE >= 199309L */
  {NANOSLEEP_FUNCTION_NAME,                 safe_c_effects},

  /* assembly code */
  {ASM_FUNCTION_NAME,                       make_anywhere_read_write_memory_effects},

  /* PIPS internal intrinsics */
  {PIPS_MEMORY_BARRIER_OPERATOR_NAME,       make_anywhere_read_write_memory_effects},
  {PIPS_IO_BARRIER_OPERATOR_NAME,           make_io_read_write_memory_effects},

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



/* list generic_proper_effects_of_intrinsic(entity e, list args)
 * @return the corresponding list of effects.
 * @param e, an intrinsic function name,
 * @param args, the list or arguments.
 * modifies : nothing.
 * comment  :
 */
list
generic_proper_effects_of_intrinsic(entity e, list args)
{
    const char* s = entity_local_name(e);
    IntrinsicDescriptor *pid = IntrinsicEffectsDescriptorTable;
    list lr;

    pips_debug(3, "begin\n");

    while (pid->name != NULL) {
        if (strcmp(pid->name, s) == 0) {
                lr = (*(pid->effects_function))(e, args);
                pips_debug(3, "end\n");
                return(lr);
            }

        pid += 1;
    }

    pips_internal_error("unknown intrinsic %s", s);

    return(NIL);
}

static list
no_write_effects(entity e __attribute__ ((__unused__)),list args)
{
    list lr;

    debug(5, "no_write_effects", "begin\n");
    lr = generic_proper_effects_of_expressions(args);
    debug(5, "no_write_effects", "end\n");
    return(lr);
}


/**
   assumes may read and write effects on the objects pointed to by actual arguments

   Hopefully also used for empty functions, i.e. functions with an
   empty statement, which, most of the time, are unknown functions.
 */
list
safe_c_effects(entity e __attribute__ ((__unused__)),list args)
{
  list lw = NIL, lr = NIL;

  pips_debug(5, "begin\n");
  lr = generic_proper_effects_of_expressions(args);
  FOREACH(EXPRESSION, arg, args)
    {
      lw = gen_nconc(lw, c_actual_argument_to_may_summary_effects(arg, 'x'));
    }
  pips_debug(5, "end\n");
  lr = gen_nconc(lr, lw);
  return(lr);
}

static list
safe_c_read_only_effects(entity e __attribute__ ((__unused__)),list args)
{
  list lw = NIL, lr = NIL;

  pips_debug(5, "begin\n");
  lr = generic_proper_effects_of_expressions(args);
  FOREACH(EXPRESSION, arg, args)
    {
      lw = gen_nconc(lw, c_actual_argument_to_may_summary_effects(arg, 'r'));
    }
  pips_debug(5, "end\n");
  lr = gen_nconc(lr, lw);
  return(lr);
}


static list
address_expression_effects(entity op, list args)
{
    list le = list_undefined;
    expression ne = make_call_expression(op, args);
    call nc = syntax_call(expression_syntax(ne));

    pips_debug(5, "begin\n");
    le = generic_proper_effects_of_address_expression(ne, false);
    call_function(nc) = entity_undefined; // useless because of persistance
    call_arguments(nc) = NIL;
    free_expression(ne);
    pips_debug(5, "end\n");
    return le;
}

static list
conditional_effects(entity e __attribute__ ((__unused__)),list args)
{
  list le;

  pips_debug(5, "begin\n");
  expression cond = EXPRESSION(CAR(args));
  expression et = EXPRESSION(CAR(CDR(args)));
  expression ef = EXPRESSION(CAR(CDR(CDR(args))));

  list lc = generic_proper_effects_of_expression(cond);
  list lt = generic_proper_effects_of_expression(et);
  list lf = generic_proper_effects_of_expression(ef);

  le = (*effects_test_union_op)(lt, lf, effects_scalars_and_same_action_p);

  pips_debug_effects(8, "Effects for the two branches:\n", le);

  le = (*effects_union_op)(le, lc, effects_scalars_and_same_action_p);

  pips_debug(5, "end\n");
  return le;
}

static list
address_of_effects(entity f __attribute__ ((__unused__)),list args)
{
    list lr;
    expression e = EXPRESSION(CAR(args));
    syntax s = expression_syntax(e);
    pips_debug(5, "begin\n");
    pips_assert("address of has only one argument", gen_length(args)==1);

    if( syntax_reference_p(s))
    {
        reference r = syntax_reference(s);
        list i = reference_indices(r);
        lr = generic_proper_effects_of_expressions(i);
	if (!get_bool_property("MEMORY_EFFECTS_ONLY"))
	  lr = CONS(EFFECT, make_declaration_effect(reference_variable(r), false), lr);
    }
    else
      {
	list l_eff = NIL;
	lr = generic_proper_effects_of_complex_address_expression(e, &l_eff, false);
	/* there is no effect on the argument of & */
	effects_free(l_eff);
      }

    pips_debug_effects(5, "end\n", lr);
    return(lr);
}

/* @return the corresponding list of effects.
 * @param args, the list or arguments.
 * @param update_p, set to true if the operator is an update operator (i.e +=)
 * @param unique_p, set to true if the operator is an unique operator (i.e ++)
 * Three different cases are handled:
 * the standard assignement: x = y;
 * the assignement with update, x+=y, which implies first a read of the lhs
 * the update, x++, which has no rhs
 */
static list any_affect_effects(entity e __attribute__ ((__unused__)),
			       list args,
			       bool update_p,
			       bool unique_p)
{
  list le = NIL;
  expression lhs = EXPRESSION(CAR(args));

  pips_debug(5, "begin\n");

  if (update_p)
    {
      pips_debug(5, "update_p is true\n");
      le = generic_proper_effects_of_expression(lhs);
    }

  le = gen_nconc(le, generic_proper_effects_of_any_lhs(lhs));

  if(!unique_p)
    {
      pips_debug(5, "unique_p is false\n");

      expression rhs = EXPRESSION(CAR(CDR(args)));
      le = gen_nconc(le, generic_proper_effects_of_expression(rhs));
    }

  ifdebug(5)
    {
      pips_debug(5, "end with effects :\n");
      (*effects_prettyprint_func)(le);
    }

  return le;
}

static list affect_effects(entity e __attribute__ ((__unused__)),list args)
{
  return any_affect_effects(e, args, false, false);
}

static list
update_effects(entity e __attribute__ ((__unused__)),list args)
{
  return any_affect_effects(e, args, true, false);
}

static list
unique_update_effects(entity e __attribute__ ((__unused__)),list args)
{
  return any_affect_effects(e, args, true, true);
}

static list
assign_substring_effects(entity e __attribute__ ((__unused__)), list args)
{
    list le = NIL;

    expression lhs = EXPRESSION(CAR(args));
    syntax s = expression_syntax(lhs);
    expression l = EXPRESSION(CAR(CDR(args)));
    expression u = EXPRESSION(CAR(CDR(CDR(args))));
    expression rhs = EXPRESSION(CAR(CDR(CDR(CDR(args)))));

    pips_debug(5, "begin\n");

    if (! syntax_reference_p(s))
            pips_internal_error("not a reference");


    le = generic_proper_effects_of_written_reference(syntax_reference(s));
    le = gen_nconc(le, generic_proper_effects_of_expression(l));
    le = gen_nconc(le, generic_proper_effects_of_expression(u));

    le = gen_nconc(le, generic_proper_effects_of_expression(rhs));

    pips_debug(5, "end\n");
    return(le);
}

static list
substring_effect(entity e __attribute__ ((__unused__)), list args)
{
    list le = NIL;
    expression expr = EXPRESSION(CAR(args));
    expression l = EXPRESSION(CAR(CDR(args)));
    expression u = EXPRESSION(CAR(CDR(CDR(args))));

    pips_debug(5, "begin\n");

    if (! syntax_reference_p(expression_syntax(expr)))
            pips_internal_error("not a reference");

    le = generic_proper_effects_of_expression(expr);
    le = gen_nconc(le, generic_proper_effects_of_expression(l));
    le = gen_nconc(le, generic_proper_effects_of_expression(u));

    pips_debug(5, "end\n");

    return(le);
}

static
 IoElementDescriptor*
SearchIoElement(const char *s, const char *i)
{
   IoElementDescriptor *p = IoElementDescriptorTable;

      while (p->StmtName != NULL) {
        if (strcmp(p->StmtName, s) == 0 && strcmp(p->IoElementName, i) == 0)
                return(p);
        p += 1;
    }

    pips_internal_error("unknown io element %s %s", s, i);
    /* Never reaches this point. Only to avoid a warning at compile time. BC. */
    return(&IoElementDescriptorUndefined);
}

/* return the appropriate C IO function.Amira Mensi*/
static
 IoElementDescriptor*
SearchCIoElement(const char *s)
{
   IoElementDescriptor *p = IoElementDescriptorTable;

      while (p->StmtName != NULL) {
        if (strcmp(p->StmtName, s) == 0)
                return(p);
        p += 1;
    }

    pips_internal_error("unknown io element %s", s);

    return(&IoElementDescriptorUndefined);
}


static list
make_io_read_write_memory_effects(entity e __attribute__ ((__unused__)), list args __attribute__ ((__unused__)))
{
    /* Fortran standard deliberately does not define the exact output
       device of a PAUSE or STOP statement. See Page B-6 in ANSI X3.9-1978
       FORTRAN 77. We assume a WRITE on stderr, i.e. unit 0 on UNIX, if
       one argument is available or not. */
    list le = NIL;
    entity private_io_entity;
    reference ref;
    list indices = NIL;

    indices = CONS(EXPRESSION,
                   int_to_expression(STDERR_LUN),
                   NIL);

    private_io_entity = FindEntity(IO_EFFECTS_PACKAGE_NAME,
       IO_EFFECTS_ARRAY_NAME);

    pips_assert("io_effects", private_io_entity != entity_undefined);

    ref = make_reference(private_io_entity,indices);
    le = gen_nconc(le, generic_proper_effects_of_read_reference(ref));
    le = gen_nconc(le, generic_proper_effects_of_written_reference(ref));

    return le;
}

static list read_io_effects(entity e, list args)
{
  /* READ is a special case , for example

     N = 5
     READ *,N, (T(I),I=1,N)

     There is write effect on N in the READ statement, so we have
     to project N from the proper region of the READ statement, then add
     the precondition after.

     The correct region in this case is :

     <T(PHI1)-MAY-{1 <= PHI1, N==5} >*/


  /*  get standard io effects

      Attention : in this list of io_effects, there are already
      preconditions  so the too late application of reverse transformer
      still gives the false result !!!

      io_effects => generic_proper_effects_of_expression => ....
      effects_precondition_composition_op

 */

  list le = io_effects(e, args);

  /* get current transformer */
  statement s = effects_private_current_stmt_head();
  transformer t = (*load_transformer_func)(s);

  if (!transformer_undefined_p(t)) /* hummm... */
  {
    /* reverse-apply transformer to le. */
    le = (*effects_transformer_composition_op)(le, t);
  }

  return le;
}

static list io_effects(entity e, list args)
{
    list le = NIL, pc, lep;

    pips_debug(5, "begin\n");

    for (pc = args; pc != NIL; pc = CDR(pc)) {
        IoElementDescriptor *p;
        entity ci;
        syntax s = expression_syntax(EXPRESSION(CAR(pc)));

        pips_assert("syntax is a call", syntax_call_p(s));

        ci = call_function(syntax_call(s));
        p = SearchIoElement(entity_local_name(e), entity_local_name(ci));

        pc = CDR(pc);

        if (strcmp(p->IoElementName, "IOLIST=") == 0) {
            lep = effects_of_iolist(pc, p->ReadOrWrite);
        }
        else {
            lep = effects_of_ioelem(EXPRESSION(CAR(pc)),
                                    p->ReadOrWrite);
        }

        if (p->MayOrMust == is_approximation_may)
            effects_to_may_effects(lep);

        le = gen_nconc(le, lep);

        /* effects effects on logical units - taken from effects/io.c */
        if ((get_bool_property ("PRETTYPRINT_IO_EFFECTS")) &&
            (pc != NIL) &&
            (strcmp(p->IoElementName, "UNIT=") == 0))
        {
            /* We simulate actions on files by read/write actions
               to a static integer array
               GO:
               It is necessary to do a read and and write action to
               the array, because it updates the file-pointer so
               it reads it and then writes it ...*/
            entity private_io_entity;
            reference ref;
            list indices = NIL;
            expression unit = EXPRESSION(CAR(pc));

            if(expression_list_directed_p(unit)) {
                if(same_string_p(entity_local_name(e), READ_FUNCTION_NAME))
                    unit = int_to_expression(STDIN_LUN);
                else if(same_string_p(entity_local_name(e), WRITE_FUNCTION_NAME))
                    unit = int_to_expression(STDOUT_LUN);
                else
                    pips_internal_error("Which logical unit?");
            }

            indices = gen_nconc(indices, CONS(EXPRESSION, unit, NIL));

            private_io_entity = FindEntity(IO_EFFECTS_PACKAGE_NAME,
                 IO_EFFECTS_ARRAY_NAME);

            pips_assert("private_io_entity is defined", private_io_entity != entity_undefined);

            ref = make_reference(private_io_entity, indices);
            le = gen_nconc(le, generic_proper_effects_of_read_reference(ref));
            le = gen_nconc(le, generic_proper_effects_of_written_reference(copy_reference(ref)));
        }
    }

    pips_debug(5, "end\n");

    return(le);
}

/*generic_io_effects to encompass the system functions and the C
  libray IO functions. Amira Mensi*/

static list generic_io_effects(entity e, list args, bool system_p)
{
  /*
    return (  unix_io_effects( e,  args));
    else
    return ( c_io_effects( e,  args));
  */
  list le = NIL, lep;
  entity private_io_entity;
  //reference ref;
  list indices = NIL;
  IoElementDescriptor *p;
  int lenght=0;
  int i=0;
  /* it really is an IO, not a string operation */
  //bool file_p = true;

  expression unit = expression_undefined;

  pips_debug(5, "begin\n");

  p = SearchCIoElement(entity_local_name(e));
  lenght=strlen(p->IoElementName);

  FOREACH(EXPRESSION, arg, args)
    {
      //if we have * as last argument, we repeat the effect of the
      //penultimate argument for the rest of the arguments

      if(p->IoElementName[lenght-1]=='*' && i>=lenght-1)
	lep = effects_of_C_ioelem(arg, p->IoElementName[lenght-2]);
      else
	lep = effects_of_C_ioelem(arg, p->IoElementName[i]);

      i=i+1;

      if (p->MayOrMust == is_approximation_may)
	effects_to_may_effects(lep);

      le = gen_nconc(le, lep);

      ifdebug(8)
	{
	  pips_debug(8, "effects for argument %s :\n",
		     words_to_string(words_expression(arg,NIL)));
	  (*effects_prettyprint_func)(lep);
	}
    }

  /* special cases */
  /* We simulate actions on files by read/write actions
     to a special static integer array.
     GO:
     It is necessary to do a read and and write action to
     the array, because it updates the file-pointer so
     it reads it and then writes it ...*/

  if(!system_p)
    {
      reference std_ref = reference_undefined;
      /* FILE * file descriptors are used */
      if(ENTITY_PRINTF_P(e) || ENTITY_PUTCHAR_P(e) ||
	 ENTITY_PUTS_P(e)|| ENTITY_VPRINTF_P(e))
	{
	  // The output is written to stdout
	  entity std_ent =  get_stdout_entity();
	  if (entity_undefined_p(std_ent))
	    pips_user_irrecoverable_error("stdout is not defined: check if \"stdio.h\""
			    " is included)\n");
	  else
	    std_ref = make_reference(std_ent, NIL);

	  if (!get_bool_property("USER_EFFECTS_ON_STD_FILES"))
	    unit = int_to_expression(STDOUT_FILENO);
	  else
	    /* we cannot use STDOUT_FILENO because the stdout variable may have been modified by the user */
	    unit = make_unbounded_expression();
	}
      else if (ENTITY_SCANF_P(e) || ENTITY_ISOC99_SCANF_P(e)
	       || ENTITY_VSCANF_P(e) || ENTITY_ISOC99_VSCANF_P(e)
	       || ENTITY_GETS_P(e)
	       || ENTITY_GETCHAR_P(e))
	{
	  //The input is obtained from stdin
	  entity std_ent =  get_stdin_entity();
	  if (entity_undefined_p(std_ent))
	    pips_user_irrecoverable_error("stdin is not defined (check if <stdio.h> is included)\n");
	  else
	    std_ref = make_reference(std_ent, NIL);

	  if (!get_bool_property("USER_EFFECTS_ON_STD_FILES"))
	    unit = int_to_expression(STDIN_FILENO);
	  else
	    /* we cannot use STDIN_FILENO because the stdin variable may have been modified by the user */
	    unit = make_unbounded_expression();
	}
      else if (ENTITY_PERROR_P(e))
	{
	  entity std_ent =  get_stderr_entity();
	  if (entity_undefined_p(std_ent))
	    pips_user_irrecoverable_error("stderr is not defined (check if <stdio.h> is included)\n");
	  else
	    std_ref = make_reference(std_ent, NIL);

	  if (!get_bool_property("USER_EFFECTS_ON_STD_FILES"))
	    unit = int_to_expression(STDERR_FILENO);
	  else
	    /* we cannot use STDERR_FILENO because the stderr variable may have been modified by the user */
	    unit = make_unbounded_expression();
	}

      else if(ENTITY_FOPEN_P(e) || ENTITY_UNLINK_SYSTEM_P(e) || ENTITY_LINK_SYSTEM_P(e)
	      || ENTITY_SYMLINK_SYSTEM_P(e))
	// the fopen and link/unlink functions the path's file as arguments.
	unit = make_unbounded_expression();

      else if(ENTITY_BUFFERIN_P(e) || ENTITY_BUFFEROUT_P(e))
	// the first argument is an integer specifying the logical unit
	// The expression should be evaluated and used if an integer is returned
	unit = make_unbounded_expression();

      if (!reference_undefined_p(std_ref))
	{
	  effect fp_eff_w = (*reference_to_effect_func)(std_ref, make_action_write_memory(), false);
	  effect fp_eff_r = (*reference_to_effect_func)(copy_reference(std_ref), make_action_read_memory(), false);
	  list l_fp_eff = CONS(EFFECT, copy_effect(fp_eff_r), NIL);
	  effect_add_dereferencing_dimension(fp_eff_w);
	  effect_to_may_effect(fp_eff_w);
	  effect_add_dereferencing_dimension(fp_eff_r);
	  effect_to_may_effect(fp_eff_r);
	  l_fp_eff = gen_nconc(l_fp_eff, CONS(EFFECT, fp_eff_r, CONS(EFFECT, fp_eff_w, NIL)));
	  le = gen_nconc(le, l_fp_eff);
	}

    }
  else
    {
      if(ENTITY_SELECT_SYSTEM_P(e) || ENTITY_PSELECT_SYSTEM_P(e))
	{
	  /* Several file descriptors are read */
	  ;
	}
    }

  if(!expression_undefined_p(unit))
    {
    reference ref1 = reference_undefined;
    effect eff1 = effect_undefined;
    reference ref2 = reference_undefined;
    effect eff2 = effect_undefined;

    indices = CONS(EXPRESSION, unit, NIL);

    private_io_entity = FindEntity(IO_EFFECTS_PACKAGE_NAME, IO_EFFECTS_ARRAY_NAME);

    pips_assert("private_io_entity is defined",
		private_io_entity != entity_undefined);

    ref1 = make_reference(private_io_entity, indices);
    ref2 = copy_reference(ref1);
    /* FI: I would like not to use "preference" instead of
       "reference", but this causes a bug in cumulated effects and I
       do not have time to chase it. */
    eff1 = (*reference_to_effect_func)(ref1, make_action_read_memory(), false);
    eff2 = (*reference_to_effect_func)(ref2, make_action_write_memory(), false);

    if(unbounded_expression_p(unit))
      {
	effect_approximation_tag(eff1) = is_approximation_may;
	effect_approximation_tag(eff2) = is_approximation_may;
      }

    ifdebug(8) print_reference(ref1);
    le = gen_nconc(le, CONS(EFFECT, eff1, CONS(EFFECT, eff2, NIL)));
  }

  pips_debug(5, "end\n");

  return(le);
}

/* unix_io_effects to manage the IO system's functions */
static list unix_io_effects(entity e, list args)
{
  return generic_io_effects(e, args, true);
}

/* c_io_effects to handle the effects of functions of the "stdio.h" library. Amira Mensi*/
static list c_io_effects(entity e, list args)
{
  return generic_io_effects(e, args, false);
}


static list strdup_effects(entity e, list args) {
    list le = safe_c_read_only_effects(e, args);
    le = gen_nconc(any_heap_effects(e,args), le);
    return le;
}

/*Molka Becher: generic_string_effects to encompass the C string.h intrinsics*/
/*
  @brief handles several C intrinsics from string.h.
 */
static list
generic_string_effects(entity e, list args)
{
  list le = NIL;
  const char * lname = entity_local_name(e);

  pips_debug(5, "begin\n");

  // first read effects for the evaluation of arguments
  le = generic_proper_effects_of_expressions(args);

  // then effects on special entities for memmove intrinsic
  if (same_string_p(lname,MEMMOVE_FUNCTION_NAME))
    {
      le = gen_nconc(le,memmove_effects(e, args));
    }

  // finally write effects on some elements of first argument depending on third argument
  // if the main effect is not on a char *, or if we don't know the number of copied elements,
  // we generate may effects on all reachable paths
  // from the main effect, without going through pointers.

  expression arg1 = EXPRESSION(CAR(args));

  if (expression_call_p(arg1)
      && call_constant_p(expression_call(arg1)))
    {
      pips_user_error("constant expression as first argument not allowed "
		      "for intrinsic %s\n", entity_name(e));
    }
  else
    {
      list l_eff1 = NIL; /* main effects on first arg */
      list l_tmp = generic_proper_effects_of_complex_address_expression(arg1, &l_eff1, true);
      gen_full_free_list(l_tmp);

      FOREACH(EFFECT, eff1, l_eff1)
	{
	  if (!anywhere_effect_p(eff1))
	    {
	      if (gen_length(args) == 3) // we may know the number of elements
		{
		  /* first check whether the argument main effect is on a char *. */
		  type t = simple_effect_reference_type(effect_any_reference(eff1));
		  variable v = type_variable_p(t) ? type_variable(t) : variable_undefined;
		  if (!variable_undefined_p(v)
		      && ((basic_pointer_p(variable_basic(v))
			   && ENDP(variable_dimensions(v))
			   && char_type_p(basic_pointer(variable_basic(v))))
			  ||
			  (char_type_p(t) && (gen_length(variable_dimensions(v)) == 1))))
		    {
		      /* Effect is on eff11_ref[0..arg3-1]*/
		      expression arg3 = EXPRESSION(CAR(CDR(CDR(args))));
		      expression argm1 = MakeBinaryCall(entity_intrinsic(MINUS_OPERATOR_NAME),
							copy_expression(arg3),
							int_to_expression(1));
		      range r = make_range(int_to_expression(0), argm1, int_to_expression(1));
		      expression ie = make_expression(make_syntax_range(r),
						      make_normalized_complex());
		      (*effect_add_expression_dimension_func)(eff1, ie);
		      le = gen_nconc(le, CONS(EFFECT, eff1, NIL));
		    }
		  else
		    {
		      le = gen_nconc(le, c_actual_argument_to_may_summary_effects(arg1, 'w'));
		      free_effect(eff1);
		    }
		}
	      else
		{
		  le = gen_nconc(le, c_actual_argument_to_may_summary_effects(arg1, 'w'));
		  free_effect(eff1);
		}
	    }
	  else
	    le = gen_nconc(le, CONS(EFFECT, eff1, NIL)); /* write is after reads */
	}
      gen_free_list(l_eff1);
    }

  // and on the same number of elements of the second one for all handled intrinsics
  // except memset.
  if (!same_string_p(lname,MEMSET_FUNCTION_NAME))
    {
      /* this is almost the same code as for arg1 just before,
	 maybe this could be factorized. */
      expression arg2 = EXPRESSION(CAR(CDR(args)));
      //type t2 = expression_to_type(arg2);
      if (expression_call_p(arg2)
	  && call_constant_p(expression_call(arg2)))
	{
	  pips_debug(5, "constant expression as ssecond argument -> no effect");
	}
      else
	{
	  list l_eff2; /* main effects on second arg */
	  list l_tmp = generic_proper_effects_of_complex_address_expression(arg2, &l_eff2, false);
	  gen_full_free_list(l_tmp);

	  FOREACH(EFFECT, eff2, l_eff2)
	    {
	      if (!anywhere_effect_p(eff2))
		{
		  if (gen_length(args) == 3) // we may know the number of elements
		    {
		      /* first check whether the argument main effect is on a char *. */
		      type t = simple_effect_reference_type(effect_any_reference(eff2));
		      variable v = type_variable_p(t) ? type_variable(t) : variable_undefined;
		      if (!variable_undefined_p(v)
			  && ((basic_pointer_p(variable_basic(v))
			       && ENDP(variable_dimensions(v))
			       && char_type_p(basic_pointer(variable_basic(v))))
			      ||
			      (char_type_p(t) && (gen_length(variable_dimensions(v)) == 1))))
			{
			  /* Effect is on eff12_ref[0..arg3-1]*/
			  expression arg3 = EXPRESSION(CAR(CDR(CDR(args))));
			  expression argm1 = MakeBinaryCall(entity_intrinsic(MINUS_OPERATOR_NAME),
							    copy_expression(arg3),
							    int_to_expression(1));
			  range r = make_range(int_to_expression(0), argm1, int_to_expression(1));
			  expression ie = make_expression(make_syntax_range(r),
							  make_normalized_complex());
			  (*effect_add_expression_dimension_func)(eff2, ie);
			  le = gen_nconc(le, CONS(EFFECT, eff2, NIL));
			}
		      else
			{
			  le = gen_nconc(le, c_actual_argument_to_may_summary_effects(arg2, 'r'));
			  free_effect(eff2);
			}
		    }
		  else
		    {
		      le = gen_nconc(le, c_actual_argument_to_may_summary_effects(arg2, 'r'));
		      free_effect(eff2);
		    }
		}
	      else
		le = gen_nconc(le, CONS(EFFECT, eff2, NIL));
	    }
	  gen_free_list(l_eff2);
	}
    }

  pips_debug_effects(5,"returning:\n", le);
  pips_debug(5, "end\n");

  return(le);
}

/* To handle the effects of random functions. Amira Mensi*/
static list any_rgs_effects(entity e __attribute__ ((__unused__)), list args, bool init_p)
{
  list le = NIL;
  list lep = NIL;
  entity private_rgs_entity = entity_undefined;
  reference ref;
  list indices = NIL;

  pips_debug(5, "begin\n");

  MAP(EXPRESSION,exp,{
    lep = generic_proper_effects_of_expression(exp);
    le = gen_nconc(le, lep);
    //ifdebug(8) print_effects(le);
    //ifdebug(8) print_effects(lep);
  }, args);

  private_rgs_entity = FindEntity(RAND_EFFECTS_PACKAGE_NAME,
     RAND_GEN_EFFECTS_NAME);

  pips_assert("gen_seed_effects", private_rgs_entity != entity_undefined);

  ref = make_reference(private_rgs_entity, indices);

  ifdebug(8) print_reference(ref);

  /* Read first. */
  if(init_p != true){
    le = gen_nconc(le, generic_proper_effects_of_read_reference(ref));
  }

  /* Init or write back. */
  le = gen_nconc(le, generic_proper_effects_of_written_reference(ref));

  pips_debug(5, "end\n");

  return(le);
}

/* The seed is written for initialization */
static list rgsi_effects(entity e, list args)
{
  return any_rgs_effects( e, args, true);
}

/* The seed is read and then written */
static list rgs_effects(entity e, list args)
{
  return any_rgs_effects( e, args, false);
}

/* To handle the effects of heap related functions.
   CreateHeapAbstractState() is done in bootstrap.c */
static list any_heap_effects(entity e, list args)
{
  list le = NIL;
  list lep = NIL;
  entity malloc_entity = entity_undefined;
  reference ref;
  effect malloc_effect;

  pips_debug(5, "begin for function \"%s\"\n", entity_user_name(e));

  MAP(EXPRESSION,exp,{
    lep = generic_proper_effects_of_expression(exp);
    le = gen_nconc(le, lep);
    //ifdebug(8) print_effects(le);
    //ifdebug(8) print_effects(lep);
  }, args);

  malloc_entity = FindEntity(MALLOC_EFFECTS_PACKAGE_NAME,
     MALLOC_EFFECTS_NAME);

  pips_assert("malloc entity pre-exists", !entity_undefined_p(malloc_entity));

  ref = make_reference(malloc_entity, NIL);

  ifdebug(8) print_reference(ref);

  /* Read first. */
  malloc_effect = (*reference_to_effect_func)(ref, make_action_read_memory(), false);

  le = CONS(EFFECT, malloc_effect, le);

  /* Write back. */
  malloc_effect = (*reference_to_effect_func)(copy_reference(ref), make_action_write_memory(), false);
  le = CONS(EFFECT, malloc_effect, le);

  pips_debug_effects(5, "output effects :\n", le);

  return(le);
}

/* Molka Becher : To handle the effects of memmove function. Memmove acts as if it uses
   a temporary array to copy characters from one object to another. C99
   Note : CreateMemmoveAbstractState() is defined in bootstrap.c  */
static list memmove_effects(entity e, list args __attribute__ ((unused)))
{
  list le = NIL;
  entity memmove_entity = entity_undefined;
  reference ref;
  effect memmove_effect;

  pips_debug(5, "begin for function \"%s\"\n", entity_user_name(e));

/*   MAP(EXPRESSION,exp,{ */
/*     lep = generic_proper_effects_of_expression(exp); */
/*     le = gen_nconc(le, lep); */
/*     //ifdebug(8) print_effects(le); */
/*     //ifdebug(8) print_effects(lep); */
/*   }, args); */

  memmove_entity = FindEntity(MEMMOVE_EFFECTS_PACKAGE_NAME,
     MEMMOVE_EFFECTS_NAME);

  pips_assert("memmove entity pre-exists", !entity_undefined_p(memmove_entity));

  ref = make_reference(memmove_entity, NIL);

  ifdebug(8) print_reference(ref);

  /* Write First. */
  memmove_effect = (*reference_to_effect_func)(copy_reference(ref), make_action_write_memory(), false);
  le = CONS(EFFECT, memmove_effect, le);

  /* Read Back. */
  memmove_effect = (*reference_to_effect_func)(copy_reference(ref), make_action_read_memory(), false);

  le = CONS(EFFECT, memmove_effect, le);


  pips_debug_effects(5, "output effects :\n", le);

  return(le);
}

static entity dummy_c_io_ptr = entity_undefined;

/* Intrinsics do not have formal parameters. This function returns a
   char * needed to analyze calls to C IO functions. */
static entity make_dummy_io_ptr()
{
  if(entity_undefined_p(dummy_c_io_ptr)) {
    type pt = make_scalar_integer_type(1); /* char */
    type t = make_type_variable(make_variable(make_basic_pointer(pt), NIL, NIL));

   dummy_c_io_ptr =
      make_entity(AddPackageToName(IO_EFFECTS_PACKAGE_NAME,
				   IO_EFFECTS_PTR_NAME),
		  t,
		  make_storage(is_storage_ram,
			       make_ram(FindEntity(TOP_LEVEL_MODULE_NAME,
							      IO_EFFECTS_PACKAGE_NAME),
					FindEntity(IO_EFFECTS_PACKAGE_NAME,
							      STATIC_AREA_LOCAL_NAME),
					0, NIL)),
		  make_value_unknown());
  }

  return dummy_c_io_ptr;
}

static list effects_of_any_ioelem(expression exp, tag act, bool is_fortran)
{
  list le = NIL;
  //syntax s = expression_syntax(exp);

  pips_debug(5, "begin with expression %s, act=%s\n",
	     words_to_string(words_expression(exp,NIL)),
	     (act == is_action_write) ? "w" :
	     ((act == 'x') ? "read-write" : "r"));

  if (act == 'w' || act == 'x' || act == is_action_write) {
    pips_debug(6, "is_action_write or read-write\n");

    if(is_fortran)
      le = generic_proper_effects_of_any_lhs(exp);
    else {
      pips_internal_error("we should never get here: there is effects_of_c_ioelem for that purpose");
      /* C language */
      /* FI: we lack information about the number of elements written */
      /* This is not generic! */
      entity ioptr = make_dummy_io_ptr();
      reference r = make_reference(ioptr, CONS(EXPRESSION, make_unbounded_expression(), NIL));
      effect eff = (*reference_to_effect_func)(r, make_action_write_memory(),false);
      effect_approximation_tag(eff) = is_approximation_may;
      /* FI: this is really not generic! removed because should be useless BC. */
      // le = c_summary_effect_to_proper_effects(eff, exp);
      /* FI: We also need the read effects implied by the evaluation
	 of exp... but I do not know any function available to do
	 that. generic_proper_effects_of_expression() is ging to
	 return a bit too much. */


      if(false) {
	/* FI: short term simplification... We need pointers for side effects... */
	syntax s = expression_syntax(exp);

	if(syntax_reference_p(s)) {
	  /* This is not possible as parameters are passed by value,
	     except if an address is passed, for instance an array or a
	     pointer */
	  reference r = syntax_reference(s);
	  entity v = reference_variable(r);
	  type t = entity_type(v);
	  type ut = ultimate_type(t); /* FI: is ultimate_type() enough? */

	  if(type_variable_p(ut)) {
	    variable vt = type_variable(ut);

	    if(!ENDP(variable_dimensions(vt))) {
	      /* Fine, this is an array */
	      le = generic_proper_effects_of_any_lhs(exp);
	      /* Is it fully written? More information about the IO and
		 about the array would be needed to make the
		 decision...*/
	      pips_assert("Only one effect for this reference?", gen_length(le)==1);
	      effects_to_may_effects(le);
	    }
	    else if(basic_pointer_p(variable_basic(vt))) {
	      /* This is going to be called for the FILE descriptors */
	      /* pips_internal_error("Effects thru pointers not implemented yet"); */
	      pips_user_warning("Effects thru pointers not implemented yet\n");
	    }
	    else
	      pips_user_error("IO element update for \"\%s\": an address should be passed",
			      entity_user_name(v));
	  }
	  else {
	    pips_user_error("IO element update for \"\%s\": incompatible type",
			    entity_user_name(v));
	  }
	}
	else if(syntax_call_p(s)) {
	  call c = syntax_call(s);
	  entity op = call_function(c);
	  list nargs = call_arguments(c);

	  if(ENTITY_DEREFERENCING_P(op)) {
	    pips_internal_error("Effects thru dereferenced pointers not implemented yet");
	  }
	  if(ENTITY_ADDRESS_OF_P(op)) {
	    expression e = EXPRESSION(CAR(nargs));

	    pips_assert("one argument", gen_length(nargs)==1);
	    le = generic_proper_effects_of_any_lhs(e);
	  }
	  else {
	    pips_internal_error("Operator \"\%s\" not handled", entity_name(op));
	  }
	}
      }
    }
  }

  if(act ==  'r' || act == 'x' || act == is_action_read) {
    pips_debug(6, "is_action_read or read-write\n");
    le = gen_nconc(le, generic_proper_effects_of_expression(exp));
  }


  pips_debug(5, "end\n");

  return le;
}

static list effects_of_ioelem(expression exp, tag act)
{
  return effects_of_any_ioelem(exp, act, true);
}

/**
   Computes the effects of an io C intrinsic function on the actual
   argument arg according to the action tag act.
   proper effects on the evaluation of the argument are systematically added.

 @param arg is an actual argument of a C io intrinsic function
 @param act is an action tag as described in the  IoElementDescriptorTable.
        (its value can be either 'i', 's', 'r', 'w','x', 'R', 'W', 'v' or 'n').
 */
static list effects_of_C_ioelem(expression arg, tag act)
{
  list le = NIL; /* result list */
  expression unit = expression_undefined;
  list indices = NIL;
  reference ref1, ref2;
  effect eff1, eff2;
  bool must_p;
  entity private_io_entity;

  pips_debug(5, "begin with expression %s and tag %c\n",
	     words_to_string(words_expression(arg,NIL)), act);

  le = gen_nconc(le, generic_proper_effects_of_expression(arg));

  switch (act)
    {
    case 'f':
      unit = copy_expression(arg);
            /* We simulate actions on files by read/write actions
	       to a special static integer array.
	       GO:
	       It is necessary to do a read and write action to
	       the array, because it updates the file-pointer so
	       it reads it and then writes it ...
	    */
      indices = CONS(EXPRESSION, unit, NIL);
      private_io_entity = FindEntity(IO_EFFECTS_PACKAGE_NAME,
	 IO_EFFECTS_ARRAY_NAME);

      pips_assert("private_io_entity is defined\n",
		  private_io_entity != entity_undefined);

      ref1 = make_reference(private_io_entity, indices);
      ref2 = copy_reference(ref1);
      eff1 = (*reference_to_effect_func)(ref1, make_action_read_memory(), false);
      eff2 = (*reference_to_effect_func)(ref2, make_action_write_memory(), false);

      le = gen_nconc(le, CONS(EFFECT, eff1, CONS(EFFECT, eff2, NIL)));
      break;
    case 's':
      pips_debug(5, "stream or integer file descriptor case \n");


       /* first the effects on the file pointer */
       /* We should maybe check here that the argument has the right type (FILE *) */
       list l_fp_eff_r = NIL;
       list l_fp_eff = generic_proper_effects_of_complex_address_expression(arg, &l_fp_eff_r, false);

       FOREACH(EFFECT, fp_eff_r, l_fp_eff_r)
	 {
	   effect fp_eff_w = effect_undefined;
	   if (effect_undefined_p(fp_eff_r))
	     {
	       fp_eff_w = make_anywhere_effect(make_action_write_memory());
	       fp_eff_r = make_anywhere_effect(make_action_read_memory());
	     }
	   else
	     {
	       if( anywhere_effect_p(fp_eff_r))
		 fp_eff_w = make_anywhere_effect(make_action_write_memory());
	       else
		 {
		   /* the read effect on the file pointer will be added later */
		   /* l_fp_eff = gen_nconc(l_fp_eff, CONS(EFFECT, copy_effect(fp_eff_r), NIL)); */
		   effect_add_dereferencing_dimension(fp_eff_r);
		   effect_to_may_effect(fp_eff_r);
		   fp_eff_w = copy_effect(fp_eff_r);
		   effect_action(fp_eff_w) = make_action_write_memory();
		 }
	     }
	   l_fp_eff = gen_nconc(l_fp_eff, CONS(EFFECT, fp_eff_w, CONS(EFFECT, fp_eff_r, NIL)));

	   /* Then we simulate actions on files by read/write actions
	      to a special static integer array.
	      GO: It is necessary to do a read and write action to
	      the array, because it updates the file-pointer so
	      it reads it and then writes it ...
	      We try to identify if the stream points to a std file
	   */

	   if ((!get_bool_property("USER_EFFECTS_ON_STD_FILES")) && std_file_effect_p(fp_eff_r))
	     {
	       const char* s = entity_user_name(effect_entity(fp_eff_r));
	       if (same_string_p(s, "stdout"))
		 unit = int_to_expression(STDOUT_FILENO);
	       else if (same_string_p(s, "stdin"))
		 unit = int_to_expression(STDIN_FILENO);
	       else /* (same_string_p(s, "stderr")) */
		 unit = int_to_expression(STDERR_FILENO);
	       must_p = true;

	     }
	   else
	     {
	       unit = make_unbounded_expression();
	       must_p = false;
	     }
	   indices = CONS(EXPRESSION,
			  unit,
			  NIL);
	   private_io_entity = FindEntity(IO_EFFECTS_PACKAGE_NAME,
	      IO_EFFECTS_ARRAY_NAME);

	   pips_assert("private_io_entity is defined\n",
		       private_io_entity != entity_undefined);

	   ref1 = make_reference(private_io_entity, indices);
	   ref2 = copy_reference(ref1);
	   eff1 = (*reference_to_effect_func)(ref1, make_action_read_memory(), false);
	   eff2 = (*reference_to_effect_func)(ref2, make_action_write_memory(), false);

	   if(!must_p)
	     {
	       effect_approximation_tag(eff1) = is_approximation_may;
	       effect_approximation_tag(eff2) = is_approximation_may;
	     }
	   le = gen_nconc(le, CONS(EFFECT, eff1, CONS(EFFECT, eff2, NIL)));
	 }
       le = gen_nconc(le, l_fp_eff);

       break;
    case 'v':
      pips_debug(5, "va_list case \n");

      /* we generate a read and a write effect on the va_list :
         this is not really correct, since the va_list itself cannot be
	 modified, but it's internal fields only. BC.
      */

      le = gen_nconc(le, generic_proper_effects_of_any_lhs(arg));
      break;
    case 'r':
    case 'w':
    case 'x':
      pips_debug(5, "potential pointer \n");
      le = gen_nconc(le, c_actual_argument_to_may_summary_effects(arg, act));
      break;
    case 'R':
    case 'W':
      pips_debug(5, "formatted IOs\n");
      /* first check whether the argument is a char *. */
      type t = expression_to_type(arg);
      variable v = type_variable_p(t) ? type_variable(t) : variable_undefined;
      if (act == 'W'
	  ||
	  (!variable_undefined_p(v)
	   && ((basic_pointer_p(variable_basic(v))
		&& ENDP(variable_dimensions(v))
		&& char_type_p(basic_pointer(variable_basic(v))))
	       ||
	       (char_type_p(t) && (gen_length(variable_dimensions(v)) == 1)))))
	     {
	  le = gen_nconc(le, c_actual_argument_to_may_summary_effects(arg, tolower(act)));
	}
      break;
    case 'n':
      pips_debug(5, "only effects on actual argument evaluation\n");
      break;
    default :
      pips_internal_error("unknown tag");
    }

  return le;
}

static list
effects_of_iolist(list exprs, tag act)
{
    list lep = NIL;
    expression exp = EXPRESSION(CAR(exprs));

    pips_debug(5, "begin with exp = %s\n",
	       words_to_string(words_expression(exp,NIL)));

    if (expression_implied_do_p(exp))
        lep = effects_of_implied_do(exp, act);
    else
    {
       if (act == is_action_write)
        {
            syntax s = expression_syntax(exp);

            pips_debug(6, "is_action_write");
            /* pips_assert("effects_of_iolist", syntax_reference_p(s)); */
            if(syntax_reference_p(s))
              lep = generic_proper_effects_of_written_reference(syntax_reference(s));
            else
            {
              /* write action on a substring */
              if(syntax_call_p(s) &&
                 strcmp(entity_local_name(call_function(syntax_call(s))),
                        SUBSTRING_FUNCTION_NAME) == 0 )
              {
                expression e = EXPRESSION(CAR(call_arguments(syntax_call(s))));
                expression l = EXPRESSION(CAR(CDR(call_arguments(syntax_call(s)))));
                expression u = EXPRESSION(CAR(CDR(CDR(call_arguments(syntax_call(s))))));

                lep = generic_proper_effects_of_written_reference
                    (syntax_reference(expression_syntax(e)));
                lep = gen_nconc(lep, generic_proper_effects_of_expression(l));
                lep = gen_nconc(lep, generic_proper_effects_of_expression(u));
              }
              else {
                pips_internal_error("Impossible memory write effect!");
              }
            }
        }
        else {
            pips_debug(6, "is_action_read");
            lep = generic_proper_effects_of_expression(exp);
        }
    } /* if (expression_implied_do_p(exp)) */
    pips_debug(5, "end\n");

    return lep;
}

/* an implied do is a call to an intrinsic function named IMPLIED-DO;
 * its first argument is the loop index, the second one is a range, and the
 * remaining ones are expressions to be written or references to be read,
 * or another implied_do (BA).
 */

static list
effects_of_implied_do(expression exp, tag act)
{
    list le, lep, lr, args;
    expression arg1, arg2;
    entity index;
    range r;
    reference ref;
    transformer context = effects_private_current_context_head();
    transformer local_context = transformer_undefined;

    pips_assert("effects_of_implied_do", expression_implied_do_p(exp));

    pips_debug(5, "begin\n");

    args = call_arguments(syntax_call(expression_syntax(exp)));
    arg1 = EXPRESSION(CAR(args));       /* loop index */
    arg2 = EXPRESSION(CAR(CDR(args)));  /* range */

    pips_assert("effects_of_implied_do",
                syntax_reference_p(expression_syntax(arg1)));

    pips_assert("effects_of_implied_do",
                syntax_range_p(expression_syntax(arg2)));

    index = reference_variable(syntax_reference(expression_syntax(arg1)));
    ref = make_reference(index, NIL);

    r = syntax_range(expression_syntax(arg2));

    /* effects of implied do index
     * it is must_written but may read because the implied loop
     * might execute no iteration.
     */

    le = generic_proper_effects_of_written_reference(ref); /* the loop index is must-written */
    /* Read effects are masked by the first write to the implied-do loop variable */

    /* effects of implied-loop bounds and increment */
    le = gen_nconc(le, generic_proper_effects_of_expression(arg2));

    /* Do we use context information */
    if (! transformer_undefined_p(context))
    {
        transformer tmp_trans;
        Psysteme context_sc;

    /* the preconditions of the current statement don't include those
     * induced by the implied_do, because they are local to the statement.
     * But we need them to properly calculate the regions.
     * the solution is to add to the current context the preconditions
     * due to the current implied_do (think of nested implied_do).
     * Beware: the implied-do index variable may already appear
     * in the preconditions. So we have to eliminate it first.
     * the regions are calculated, and projected along the index.
     * BA, September 27, 1993.
     */

        local_context = transformer_dup(context);
        /* we first eliminate the implied-do index variable */
        context_sc = predicate_system(transformer_relation(local_context));
        if(base_contains_variable_p(context_sc->base, (Variable) index))
        {
            sc_and_base_projection_along_variable_ofl_ctrl(&context_sc,
                                                           (Variable) index,
                                                           NO_OFL_CTRL);
            predicate_system_(transformer_relation(local_context)) =
                newgen_Psysteme(context_sc);
        }
        /* tmp_trans simulates the transformer of the implied-do loop body */
        tmp_trans = transformer_identity();
        local_context = add_index_range_conditions(local_context, index, r,
                                                   tmp_trans);
        free_transformer(tmp_trans);
        transformer_arguments(local_context) =
            arguments_add_entity(transformer_arguments(local_context),
                                 entity_to_new_value(index));


        ifdebug(7) {
            pips_debug(7, "local context : \n%s\n",
                       precondition_to_string(local_context));
        }
    }
    else
        local_context = transformer_undefined;

    effects_private_current_context_push(local_context);

    MAP(EXPRESSION, expr,
    {
      syntax s = expression_syntax(expr);

      if (syntax_reference_p(s))
        if (act == is_action_write)
          lep = generic_proper_effects_of_written_reference(syntax_reference(s));
        else
          lep = generic_proper_effects_of_expression(expr);
      else
        if (syntax_range_p(s))
          lep = generic_proper_effects_of_range(syntax_range(s));
        else
          /* syntax_call_p(s) is true here */
          if (expression_implied_do_p(expr))
            lep = effects_of_implied_do(expr, act);
          else
            lep = generic_r_proper_effects_of_call(syntax_call(s));

      /* indices are removed from effects because this is a loop */
      lr = NIL;
      MAP(EFFECT, eff,
      {
        if (effect_entity(eff) != index)
          lr =  CONS(EFFECT, eff, lr);
        else if(act==is_action_write /* This is a read (FI: I do not
					understand this comment) */
                && action_write_p(effect_action(eff))) {
          pips_user_error("Index %s in implied DO is read. "
                          "Standard violation, see Section 12.8.2.3\n",
                          entity_local_name(index));
        }
        else
        {
          debug(5, "effects_of_implied_do", "index removed");
          free_effect(eff);
        }
      }, lep);
      gen_free_list(lep);
      lr = gen_nreverse(lr); /* preserve initial order??? */
      le = gen_nconc(le, lr);
    }, CDR(CDR(args)));


    (*effects_union_over_range_op)(le,
                                   index,
                                   r, descriptor_undefined);

    ifdebug(6) {
      pips_debug(6, "effects:\n");
      (*effects_prettyprint_func)(le);
      fprintf(stderr, "\n");
    }

    effects_private_current_context_pop();
    pips_debug(5, "end\n");

    return le;
}

/* Add a time effect, that is a read and a write on a particular hidden time variable */
static
list time_effects(entity e __attribute__ ((unused)), list args) {
	list le = NIL;
    FOREACH(EXPRESSION,arg,args) // true for at least gettimeofday, clock and time
        le = gen_nconc(le,
                c_actual_argument_to_may_summary_effects(arg, 'w')); // may be a must ?
    entity private_time_entity = FindEntity(TIME_EFFECTS_PACKAGE_NAME, TIME_EFFECTS_VARIABLE_NAME);

    pips_assert("private_time_entity is defined",
		!entity_undefined_p(private_time_entity));

    reference ref = reference_undefined;

    ref = make_reference(private_time_entity,NIL);

    le = gen_nconc(le, generic_proper_effects_of_read_reference(ref));
    le = gen_nconc(le, generic_proper_effects_of_written_reference(ref));
    /* SG: should we free ref? */

    if(get_bool_property("TIME_EFFECTS_USED")) {
      /* Barrier effect suggested by Mehdi Amini (Ticket 594) */
      effect re = make_anywhere_effect(make_action_read(make_action_kind_store()));
      effect we = make_anywhere_effect(make_action_write(make_action_kind_store()));
      le = CONS(EFFECT, re, le);
      le = CONS(EFFECT, we, le);
    }
    return le;
}

/**
   handling of variable argument list macros (va_end, va_start and va_copy)
   according to ISO/IEC 9899:1999.
   va_list parameters are considered as scalar variables. It is unclear
   for the moment whether we should use a more precise approach to simulate
   the effects of successive calls to va_arg.
   va_arg is directly represented in the PIPS internal representation
   (see domain syntax).
 */
static list va_list_effects(entity e, list args)
{
  list le = NIL;
  expression first_arg;
  pips_debug(5, "begin for function \"%s\"\n", entity_user_name(e));

  /* the first argument is always evaluated (read) and we simulate
     the written effects on the depths of the va_list by a write
     effect on the va_list itself.
  */
  first_arg = EXPRESSION(CAR(args));
  le = gen_nconc(le, generic_proper_effects_of_expression(first_arg));
  le = gen_nconc(le, generic_proper_effects_of_any_lhs(first_arg));

  /* but it is *may* written for va_end */
  if (ENTITY_VA_END_P(e))
    {
       FOREACH(EFFECT, eff, le)
	{
	  if (effect_write_p(eff))
	    effect_approximation_tag(eff) = is_approximation_may;
	}
    }

  if (ENTITY_VA_COPY_P(e))
    {
      /* the second argument is only read. In fact, in a more precise
       approach, we should simulate reads on the depths of the va_list.
      */
      expression second_arg = EXPRESSION(CAR(CDR(args)));
      le = gen_nconc(le, generic_proper_effects_of_expression(second_arg));
    }

  ifdebug(5)
    {
      pips_debug(5, "resulting effects: \n");
      (*effects_prettyprint_func)(le);
      fprintf(stderr, "\n");
    }
  pips_debug(5, "end\n");
  return le;
}

/**
   compute the effects of a call to a searching or sorting function

   The summary effects of the comparison function should be translated
   using the actual argument target array to infer the actual effects
   of the search or sort, especially in case of an array of structs for
   instance. Currently, conservative effects are generated, which means
   effects on all the paths reachable from the array elements.
 */
static list search_or_sort_effects(entity e, list args)
{
  list le = NIL;
  list l_tmp = NIL;
  expression base_exp = expression_undefined;
  expression nmemb_exp = expression_undefined;
  expression size_exp = expression_undefined;
  expression compar_func_exp = expression_undefined;

  if(ENTITY_BSEARCH_SYSTEM_P(e))
    {
      pips_debug(8, "bsearch function\n");
      expression key_exp = EXPRESSION(CAR(args));
      // the key is passed to the comparison function as an argument.
      // so all paths reachable from key may be read.
      le = c_actual_argument_to_may_summary_effects(key_exp, 'r');
      pips_debug_effects(8, "effects for the key:", le);
      POP(args);
    }
  else if(ENTITY_QSORT_SYSTEM_P(e))
    {
      pips_debug(8, "qsort function\n");
    }
  else
    pips_internal_error("unexpected searching or sorting function\n");

  base_exp = EXPRESSION(CAR(args));
  POP(args);
  nmemb_exp = EXPRESSION(CAR(args));
  POP(args);
  size_exp = EXPRESSION(CAR(args));
  POP(args);
  compar_func_exp = EXPRESSION(CAR(args));

  // each element of the array is passed to the comparison function
  // so all paths reachable from base may be read;
  // more precise effects could be generated by taking into account
  // the summary effects of the comparison function
  l_tmp = c_actual_argument_to_may_summary_effects(base_exp, 'r');
  pips_debug_effects(8, "read effects for the base array:", l_tmp);
  le = gen_nconc(l_tmp, le);

  // then each element of the array may be written,
  // but not beyond struct internal pointers
  list l_base_eff = NIL;
  list l_base_inter_eff =
    generic_proper_effects_of_complex_address_expression(base_exp, &l_base_eff, true);

  l_tmp = NIL;
  FOREACH(EFFECT, base_eff, l_base_eff)
    {
      bool to_be_freed = false;
      type base_eff_type = cell_to_type(effect_cell(base_eff), &to_be_freed);
      l_tmp = gen_nconc
	(l_tmp,
	 generic_effect_generate_all_accessible_paths_effects_with_level(base_eff,
									 base_eff_type,
									 'w',
									 true,
									 0,
									 false));
      if (to_be_freed) free_type(base_eff_type);
    }
  pips_debug_effects(8, "write effects for the base array:", l_tmp);
  le = gen_nconc(l_tmp, le);
  le = gen_nconc(l_base_inter_eff, le);

  //nmemb_exp and size_exp are read
  le = gen_nconc(generic_proper_effects_of_expression(nmemb_exp), le);
  le = gen_nconc(generic_proper_effects_of_expression(size_exp), le);

  // and the comparison function expression is evaluated
  le = gen_nconc(generic_proper_effects_of_expression(compar_func_exp), le);

  pips_debug_effects(8, "final effects:", le);

  return le;
}


/**
    generate effects for strtoxxx functions
 */
static list strtoxxx_effects(entity e __attribute__ ((unused)), list args)
{
  list le = NIL;
  expression nptr_exp = EXPRESSION(CAR(args));
  POP(args);
  expression endptr_exp = EXPRESSION(CAR(args));
  POP(args);
  expression base_exp = EXPRESSION(CAR(args));

  /* the first expression must be char * or char[], and all its components may be read */
  le = c_actual_argument_to_may_summary_effects(nptr_exp, 'r');
  pips_debug_effects(8, "effects on first argument\n", le);

  /* the second expression target is initialized if it is a non-null pointer */
  if (expression_address_of_p(endptr_exp))
    {
      call c = expression_call(endptr_exp);
      endptr_exp = EXPRESSION(CAR(call_arguments(c)));
      list lme_endptr = NIL;
      list le_endptr = generic_proper_effects_of_complex_address_expression(endptr_exp, &lme_endptr, true);
      pips_debug_effects(8, "main effects for address of expression\n", lme_endptr);
      le = gen_nconc(le, lme_endptr);
      le = gen_nconc(le, le_endptr);
    }
  else if (!expression_null_p(endptr_exp)) /* if its a NULL pointer, it is not assigned */
    {
      list lme_endptr = NIL;
      list le_endptr = generic_proper_effects_of_complex_address_expression(endptr_exp, &lme_endptr, true);
      pips_debug_effects(8, "main effects on second argument\n", lme_endptr);

      FOREACH(EFFECT, me, lme_endptr)
	{
	  /* there is a read effect on the main pointer */
	  effect me_dup = (*effect_dup_func)(me);
	  effect_to_read_effect(me_dup);
	  le = gen_nconc(le, effect_to_list(me_dup));
	  effect_add_dereferencing_dimension(me);
	  effect_to_may_effect(me); /* well, if its a NULL pointer, it is not assigned, but we may not know it */
	}
      pips_debug_effects(8, "main effects after adding dereferencing dimension\n", lme_endptr);
      le = gen_nconc(le, lme_endptr);
      le = gen_nconc(le, le_endptr);
    }
  /* the third expression is solely read */
  le = gen_nconc(le, generic_proper_effects_of_expression(base_exp));

  pips_debug_effects(8,"final_effects:", le);
  return le;
}
