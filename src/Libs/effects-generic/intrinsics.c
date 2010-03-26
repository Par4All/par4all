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
 */

#include <stdio.h>
#include <string.h>
#include <ctype.h>

#include "genC.h"

#include "linear.h"
#include "ri.h"
#include "database.h"

#include "ri-util.h"
#include "control.h"
#include "constants.h"
#include "misc.h"
#include "text-util.h"
#include "text.h"
#include "makefile.h"

#include "properties.h"
#include "pipsmake.h"

#include "transformer.h"
#include "semantics.h"
#include "pipsdbm.h"
#include "resources.h"

#include "effects-generic.h"
#include "effects-simple.h"


/********************************************************* LOCAL FUNCTIONS */

static list no_write_effects(entity e,list args);
static list safe_c_effects(entity e,list args);
static list address_expression_effects(entity e,list args);
static list conditional_effects(entity e,list args);
static list address_of_effects(entity e,list args);
static list affect_effects(entity e,list args);
static list update_effects(entity e,list args);
static list unique_update_effects(entity e,list args);
static list assign_substring_effects(entity e,list args);
static list substring_effect(entity e,list args);
static list some_io_effects(entity e, list args);
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
  {"OPEN",      "UNIT=",        is_action_read,  is_approximation_must},
  {"OPEN",      "ERR=",         is_action_read,  is_approximation_may},
  {"OPEN",      "FILE=",        is_action_read,  is_approximation_must},
  {"OPEN",      "STATUS=",      is_action_read,  is_approximation_may},
  {"OPEN",      "ACCESS=",      is_action_read,  is_approximation_must},
  {"OPEN",      "FORM=",        is_action_read,  is_approximation_must},
  {"OPEN",      "RECL=",        is_action_read,  is_approximation_must},
  {"OPEN",      "BLANK=",       is_action_read,  is_approximation_may},
  {"OPEN",      "IOSTAT=",      is_action_write, is_approximation_may},

  {"CLOSE",     "UNIT=",        is_action_read,  is_approximation_must},
  {"CLOSE",     "ERR=",         is_action_read,  is_approximation_may},
  {"CLOSE",     "STATUS=",      is_action_read,  is_approximation_may},
  {"CLOSE",     "IOSTAT=",      is_action_write, is_approximation_may},

  {"INQUIRE",   "UNIT=",        is_action_read,  is_approximation_must},
  {"INQUIRE",   "ERR=",         is_action_read,  is_approximation_may},
  {"INQUIRE",   "FILE=",        is_action_read,  is_approximation_must},
  {"INQUIRE",   "IOSTAT=",      is_action_write, is_approximation_must},
  {"INQUIRE",   "EXIST=",       is_action_write, is_approximation_must},
  {"INQUIRE",   "OPENED=",      is_action_write, is_approximation_must},
  {"INQUIRE",   "NUMBER=",      is_action_write, is_approximation_must},
  {"INQUIRE",   "NAMED=",       is_action_write, is_approximation_must},
  {"INQUIRE",   "NAME=",        is_action_write, is_approximation_must},
  {"INQUIRE",   "ACCESS=",      is_action_write, is_approximation_must},
  {"INQUIRE",   "SEQUENTIAL=",  is_action_write, is_approximation_must},
  {"INQUIRE",   "DIRECT=",      is_action_write, is_approximation_must},
  {"INQUIRE",   "FORM=",        is_action_write, is_approximation_must},
  {"INQUIRE",   "FORMATTED=",   is_action_write, is_approximation_must},
  {"INQUIRE",   "UNFORMATTED=", is_action_write, is_approximation_must},
  {"INQUIRE",   "RECL=",        is_action_write, is_approximation_must},
  {"INQUIRE",   "NEXTREC=",     is_action_write, is_approximation_must},
  {"INQUIRE",   "BLANK=",       is_action_write, is_approximation_must},

  {"BACKSPACE", "UNIT=",        is_action_read,  is_approximation_must},
  {"BACKSPACE", "ERR=",         is_action_read,  is_approximation_may},
  {"BACKSPACE", "IOSTAT=",      is_action_write, is_approximation_may},

  {"ENDFILE",   "UNIT=",        is_action_read,  is_approximation_must},
  {"ENDFILE",   "ERR=",         is_action_read,  is_approximation_may},
  {"ENDFILE",   "IOSTAT=",      is_action_write, is_approximation_may},

  {"REWIND",    "UNIT=",        is_action_read,  is_approximation_must},
  {"REWIND",    "ERR=",         is_action_read,  is_approximation_may},
  {"REWIND",    "IOSTAT=",      is_action_write, is_approximation_may},

  {"READ",      "FMT=",         is_action_read,  is_approximation_must},
  {"READ",      "UNIT=",        is_action_read,  is_approximation_must},
  {"READ",      "REC=",         is_action_read,  is_approximation_must},
  {"READ",      "ERR=",         is_action_read,  is_approximation_may},
  {"READ",      "END=",         is_action_read,  is_approximation_must},
  {"READ",      "IOSTAT=",      is_action_write, is_approximation_may},
  {"READ",      "IOLIST=",      is_action_write, is_approximation_must},

  {"WRITE",     "FMT=",         is_action_read,  is_approximation_must},
  {"WRITE",     "UNIT=",        is_action_read,  is_approximation_must},
  {"WRITE",     "REC=",         is_action_read,  is_approximation_must},
  {"WRITE",     "ERR=",         is_action_read,  is_approximation_may},
  {"WRITE",     "END=",         is_action_read,  is_approximation_must},
  {"WRITE",     "IOSTAT=",      is_action_write, is_approximation_may},
  {"WRITE",     "IOLIST=",      is_action_read,  is_approximation_must},

  /* C IO intrinsics */

  /* The field IoElementName is used to describe the function's pattern
     defined according to the standard ISO/IEC 9899 (BC, july 2009) :
     n      when there is only the read effect on the value of the actual
            argument.
     r,w,x  for read, write, or read and write effects on the object
            pointed to by the actual argument.
     *      means that the last effect is repeated for the last arguments
            (varargs).
     s      for a FILE * argument ("s" stands for "stream").
     f      for an integer file descriptor (unix io system calls).
     v      for a va_list argument (this could be enhanced in the future
            to distinguish between read and write effects on the components
            of the va_list).

     The tag fields are not relevant.

  */

  {PRINTF_FUNCTION_NAME,        "rn*",     is_action_read, is_approximation_must},
  {FPRINTF_FUNCTION_NAME,       "srn*",    is_action_read, is_approximation_must},
  {SCANF_FUNCTION_NAME,         "rw*",     is_action_read, is_approximation_must},
  {ISOC99_SCANF_FUNCTION_NAME,         "rw*",     is_action_read, is_approximation_must},
  {FSCANF_FUNCTION_NAME,        "srw*",    is_action_read, is_approximation_must},
  {ISOC99_FSCANF_FUNCTION_NAME,        "srw*",    is_action_read, is_approximation_must},
  {PUTS_FUNCTION_NAME,          "r",       is_action_read, is_approximation_must},
  {GETS_FUNCTION_NAME,          "w",       is_action_read, is_approximation_must},
  {FPUTS_FUNCTION_NAME,         "rs",     is_action_read, is_approximation_must},
  {FGETS_FUNCTION_NAME,         "wns",     is_action_read, is_approximation_must},
  {FOPEN_FUNCTION_NAME,         "rr",      is_action_read, is_approximation_must},
  {FCLOSE_FUNCTION_NAME,        "s",       is_action_read, is_approximation_must},
  {SNPRINTF_FUNCTION_NAME,      "wnrn*",    is_action_read, is_approximation_must},
  {SPRINTF_FUNCTION_NAME,       "wrn*",     is_action_read, is_approximation_must},
  {SSCANF_FUNCTION_NAME,        "rrw*",    is_action_read, is_approximation_must},
  {ISOC99_SSCANF_FUNCTION_NAME,        "rrw*",    is_action_read, is_approximation_must},
  {VFPRINTF_FUNCTION_NAME,      "srv",      is_action_read, is_approximation_must},
  {VFSCANF_FUNCTION_NAME,       "srv",     is_action_read, is_approximation_must},
  {ISOC99_VFSCANF_FUNCTION_NAME,       "srv",     is_action_read, is_approximation_must},
  {VPRINTF_FUNCTION_NAME,       "rv",      is_action_read, is_approximation_must},
  {VSNPRINTF_FUNCTION_NAME,     "wnrv",    is_action_read, is_approximation_must},
  {VSPRINTF_FUNCTION_NAME,      "wrv",     is_action_read, is_approximation_must},
  {VSSCANF_FUNCTION_NAME,       "rrv",     is_action_read, is_approximation_must},
  {ISOC99_VSSCANF_FUNCTION_NAME,       "rrv",     is_action_read, is_approximation_must},
  {VSCANF_FUNCTION_NAME,        "rv",      is_action_read, is_approximation_must},
  {ISOC99_VSCANF_FUNCTION_NAME,        "rv",      is_action_read, is_approximation_must},
  {FPUTC_FUNCTION_NAME,         "ns",      is_action_read, is_approximation_must},
  {GETC_FUNCTION_NAME,          "s",       is_action_read, is_approximation_must},
  {_IO_GETC_FUNCTION_NAME,      "s",       is_action_read, is_approximation_must},
  {FGETC_FUNCTION_NAME,         "s",       is_action_read, is_approximation_must},
  {GETCHAR_FUNCTION_NAME,       "",       is_action_read, is_approximation_must},
  {PUTC_FUNCTION_NAME,          "ns",      is_action_read, is_approximation_must},
  {_IO_PUTC_FUNCTION_NAME,      "ns",      is_action_read, is_approximation_must},
  {PUTCHAR_FUNCTION_NAME,       "n",       is_action_read, is_approximation_must},
  {UNGETC_FUNCTION_NAME,        "ns",      is_action_read, is_approximation_must},
  {FREAD_FUNCTION_NAME,         "wnns",    is_action_read, is_approximation_must},
  {FWRITE_FUNCTION_NAME,        "rnns",    is_action_read, is_approximation_must},
  {FGETPOS_FUNCTION_NAME,       "sw",      is_action_read, is_approximation_must},
  {FSEEK_FUNCTION_NAME,         "snn",     is_action_read, is_approximation_must},
  {FSETPOS_FUNCTION_NAME,       "sr",      is_action_read, is_approximation_must},
  {FTELL_FUNCTION_NAME,         "s",       is_action_read, is_approximation_must},
  {C_REWIND_FUNCTION_NAME,      "s",       is_action_read, is_approximation_must},
  {CLEARERR_FUNCTION_NAME,      "s",       is_action_read, is_approximation_must},
  {FEOF_FUNCTION_NAME,          "s",       is_action_read, is_approximation_must},
  {FERROR_FUNCTION_NAME,        "s",       is_action_read, is_approximation_must},
  {PERROR_FUNCTION_NAME,        "r",       is_action_read, is_approximation_must},

  /* UNIX IO system calls */

  {C_OPEN_FUNCTION_NAME,        "nn",     is_action_read, is_approximation_must},
  {CREAT_FUNCTION_NAME,         "nn",      is_action_read, is_approximation_must},
  {C_CLOSE_FUNCTION_NAME,       "f",       is_action_read, is_approximation_must},
  {C_WRITE_FUNCTION_NAME,       "frr",     is_action_read, is_approximation_must},
  {C_READ_FUNCTION_NAME,        "fwn",     is_action_read, is_approximation_must},
  {FCNTL_FUNCTION_NAME,         "fnn*",     is_action_read, is_approximation_must},
  {FSYNC_FUNCTION_NAME,         "f",       is_action_read, is_approximation_must},
  {FDATASYNC_FUNCTION_NAME,     "f",       is_action_read, is_approximation_must},
  {IOCTL_FUNCTION_NAME,         "fn*",     is_action_read, is_approximation_must},
  {SELECT_FUNCTION_NAME,        "nrrrr",   is_action_read, is_approximation_must},
  {PSELECT_FUNCTION_NAME,       "nrrrrw",  is_action_read, is_approximation_must},
  {FSTAT_FUNCTION_NAME,         "nw",      is_action_read, is_approximation_must},

  /* wchar.h */
  {FWSCANF_FUNCTION_NAME, "srw*",is_action_read, is_approximation_must},
  {SWSCANF_FUNCTION_NAME, "rrw*",is_action_read, is_approximation_must},
  {WSCANF_FUNCTION_NAME, "rw*",is_action_read, is_approximation_must},

  /* Fortran extensions for asynchronous IO's */

  {BUFFERIN_FUNCTION_NAME,      "xrwr",    is_action_read, is_approximation_must},
  {BUFFEROUT_FUNCTION_NAME,     "xrrr",    is_action_read, is_approximation_must},

 
  {0,                            0,        0,              0}
};


/* the following data structure describes an intrinsic function: its
name and the function to apply on a call to this intrinsic to get the
effects of the call */

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
  {PAUSE_FUNCTION_NAME,                    some_io_effects},
  {RETURN_FUNCTION_NAME,                   no_write_effects},
  {STOP_FUNCTION_NAME,                     some_io_effects},
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

  {SUBSTRING_FUNCTION_NAME,    substring_effect},
  {ASSIGN_SUBSTRING_FUNCTION_NAME, assign_substring_effects},

  /* These operators are used within the OPTIMIZE transformation in
     order to manipulate operators such as n-ary add and multiply or
     multiply-add operators ( JZ - sept 98) */
  {EOLE_SUM_OPERATOR_NAME,                 no_write_effects },
  {EOLE_PROD_OPERATOR_NAME,                no_write_effects },
  {EOLE_FMA_OPERATOR_NAME,                 no_write_effects },

  {IMA_OPERATOR_NAME,                      no_write_effects },
  {IMS_OPERATOR_NAME,                      no_write_effects },

  /* Bits manipulation functions. Amira Mensi */


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

  {FIELD_OPERATOR_NAME,                    address_expression_effects},
  {POINT_TO_OPERATOR_NAME,                 address_expression_effects},
  {POST_INCREMENT_OPERATOR_NAME,           unique_update_effects},
  {POST_DECREMENT_OPERATOR_NAME,           unique_update_effects},
  {PRE_INCREMENT_OPERATOR_NAME,            unique_update_effects},
  {PRE_DECREMENT_OPERATOR_NAME,            unique_update_effects},
  {ADDRESS_OF_OPERATOR_NAME,               address_of_effects},
  {DEREFERENCING_OPERATOR_NAME,            address_expression_effects},
  {UNARY_PLUS_OPERATOR_NAME,               no_write_effects},
  // {"-unary",                            no_write_effects},UNARY_MINUS_OPERATOR already exist (FORTRAN)
  {BITWISE_NOT_OPERATOR_NAME,              no_write_effects},
  {C_NOT_OPERATOR_NAME,                    no_write_effects},
  {C_MODULO_OPERATOR_NAME,                 no_write_effects},
  {PLUS_C_OPERATOR_NAME,                   no_write_effects},
  {MINUS_C_OPERATOR_NAME,                  no_write_effects},
  {LEFT_SHIFT_OPERATOR_NAME,               no_write_effects},
  {RIGHT_SHIFT_OPERATOR_NAME,              no_write_effects},
  {C_LESS_THAN_OPERATOR_NAME,              no_write_effects},
  {C_GREATER_THAN_OPERATOR_NAME,           no_write_effects},
  {C_LESS_OR_EQUAL_OPERATOR_NAME,          no_write_effects},
  {C_GREATER_OR_EQUAL_OPERATOR_NAME,       no_write_effects},
  {C_EQUAL_OPERATOR_NAME,                  no_write_effects},
  {C_NON_EQUAL_OPERATOR_NAME,              no_write_effects},
  {BITWISE_AND_OPERATOR_NAME,              no_write_effects},
  {BITWISE_XOR_OPERATOR_NAME,              no_write_effects},
  {BITWISE_OR_OPERATOR_NAME,               no_write_effects},
  {C_AND_OPERATOR_NAME,                    no_write_effects},
  {C_OR_OPERATOR_NAME,                     no_write_effects},
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
  {COMMA_OPERATOR_NAME,                    no_write_effects},
  {CONDITIONAL_OPERATOR_NAME,              conditional_effects},

  {BRACE_INTRINSIC,                        no_write_effects},
  {BREAK_FUNCTION_NAME,                    no_write_effects},
  {CASE_FUNCTION_NAME,                     no_write_effects},
  {DEFAULT_FUNCTION_NAME,                  no_write_effects},
  {C_RETURN_FUNCTION_NAME,                 no_write_effects},

  /* stdarg.h */

  {BUILTIN_VA_START,                       va_list_effects},
  {BUILTIN_VA_END,                         va_list_effects},
  {BUILTIN_VA_COPY,                        va_list_effects},
  /* va_arg is not a standard call; it is directly represented in PIPS
     internal representation. */

  /* These intrinsics are added with no_write_effects to work with C.
     The real effects must be studied !!! I do not have time for the moment */

  {"__assert",                             no_write_effects},
  {"__assert_fail",                        no_write_effects}, /* in fact, IO effect, does not return */

  /* #include <ctype.h>*/

  {ISALNUM_OPERATOR_NAME,                  no_write_effects},
  {ISALPHA_OPERATOR_NAME,                  no_write_effects},
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
  {ISASCII_OPERATOR_NAME,                  no_write_effects},
  {TOASCII_OPERATOR_NAME,                  no_write_effects},
  {_TOLOWER_OPERATOR_NAME,                 no_write_effects},
  {_TOUPPER_OPERATOR_NAME,                 no_write_effects},
  {CTYPE_B_LOC_OPERATOR_NAME,              no_write_effects},

  {"errno",                                no_write_effects},

  {"__flt_rounds",                         no_write_effects},

  {"_sysconf",                             no_write_effects},
  {"setlocale",                            no_write_effects},
  {"localeconv",                           no_write_effects},
  {"dcgettext",                            no_write_effects},
  {"dgettext",                             no_write_effects},
  {"gettext",                              no_write_effects},
  {"textdomain",                           no_write_effects},
  {"bindtextdomain",                       no_write_effects},
  {"wdinit",                               no_write_effects},
  {"wdchkind",                             no_write_effects},
  {"wdbindf",                              no_write_effects},
  {"wddelim",                              no_write_effects},
  {"mcfiller",                             no_write_effects},
  {"mcwrap",                               no_write_effects},

  /* #include <math.h>*/

  {C_ACOS_OPERATOR_NAME,                   no_write_effects},
  {C_ASIN_OPERATOR_NAME,                   no_write_effects},
  {C_ATAN_OPERATOR_NAME,                   no_write_effects},
  {C_ATAN2_OPERATOR_NAME,                  no_write_effects},
  {C_COS_OPERATOR_NAME,                    no_write_effects},
  {C_SIN_OPERATOR_NAME,                    no_write_effects},
  {C_TAN_OPERATOR_NAME,                    no_write_effects},
  {C_COSH_OPERATOR_NAME,                   no_write_effects},
  {C_SINH_OPERATOR_NAME,                   no_write_effects},
  {C_TANH_OPERATOR_NAME,                   no_write_effects},
  {C_EXP_OPERATOR_NAME,                    no_write_effects},
  {FREXP_OPERATOR_NAME,                    no_write_effects},
  {LDEXP_OPERATOR_NAME,                    no_write_effects},
  {C_LOG_OPERATOR_NAME,                    no_write_effects},
  {C_LOG10_OPERATOR_NAME,                  no_write_effects},
  {MODF_OPERATOR_NAME,                     no_write_effects},
  {POW_OPERATOR_NAME,                      no_write_effects},
  {C_SQRT_OPERATOR_NAME,                   no_write_effects},
  {CEIL_OPERATOR_NAME,                     no_write_effects},
  {FABS_OPERATOR_NAME,                     no_write_effects},
  {FLOOR_OPERATOR_NAME,                    no_write_effects},
  {FMOD_OPERATOR_NAME,                     no_write_effects},
  {ERF_OPERATOR_NAME,                      no_write_effects},
  {ERFC_OPERATOR_NAME,                     no_write_effects},
  {GAMMA_OPERATOR_NAME,                    no_write_effects},
  {HYPOT_OPERATOR_NAME,                    no_write_effects},
  {ISNAN_OPERATOR_NAME,                    no_write_effects},
  {J0_OPERATOR_NAME,                       no_write_effects},
  {J1_OPERATOR_NAME,                       no_write_effects},
  {JN_OPERATOR_NAME,                       no_write_effects},
  {LGAMMA_OPERATOR_NAME,                   no_write_effects},
  {Y0_OPERATOR_NAME,                       no_write_effects},
  {Y1_OPERATOR_NAME,                       no_write_effects},
  {YN_OPERATOR_NAME,                       no_write_effects},
  {C_ACOSH_OPERATOR_NAME ,                 no_write_effects},
  {C_ASINH_OPERATOR_NAME,                  no_write_effects},
  {C_ATANH_OPERATOR_NAME,                  no_write_effects},
  {CBRT_OPERATOR_NAME,                     no_write_effects},
  {LOGB_OPERATOR_NAME,                     no_write_effects},
  {NEXTAFTER_OPERATOR_NAME,                no_write_effects},
  {REMAINDER_OPERATOR_NAME,                no_write_effects},
  {SCALB_OPERATOR_NAME,                    no_write_effects},
  {EXPM1_OPERATOR_NAME,                    no_write_effects},
  {ILOGB_OPERATOR_NAME,                    no_write_effects},
  {LOG1P_OPERATOR_NAME,                    no_write_effects},
  {RINT_OPERATOR_NAME,                     no_write_effects},
  {MATHERR_OPERATOR_NAME,                  no_write_effects},
  {SIGNIFICAND_OPERATOR_NAME,              no_write_effects},
  {COPYSIGN_OPERATOR_NAME,                 no_write_effects},
  {SCALBN_OPERATOR_NAME,                   no_write_effects},
  {MODFF_OPERATOR_NAME,                    no_write_effects},
  {SIGFPE_OPERATOR_NAME,                   no_write_effects},
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
  {SECONVERT_OPERATOR_NAME,                no_write_effects},
  {SFCONVERT_OPERATOR_NAME,                no_write_effects},
  {SGCONVERT_OPERATOR_NAME,                no_write_effects},
  {ECONVERT_OPERATOR_NAME,                 no_write_effects},
  {FCONVERT_OPERATOR_NAME,                 no_write_effects},
  {GCONVERT_OPERATOR_NAME,                 no_write_effects},
  {QECONVERT_OPERATOR_NAME,                no_write_effects},
  {QFCONVERT_OPERATOR_NAME,                no_write_effects},
  {QGCONVERT_OPERATOR_NAME,                no_write_effects},

  /* netdb.h */
  {__H_ERRNO_LOCATION_OPERATOR_NAME,       no_write_effects},

  /* bits/errno.h */
  {__ERRNO_LOCATION_OPERATOR_NAME,         no_write_effects},

  /* signal.h */
  {SIGNAL_OPERATOR_NAME,                   no_write_effects},

  {ECVT_FUNCTION_NAME,                     no_write_effects},
  {FCVT_FUNCTION_NAME,                     no_write_effects},
  {GCVT_FUNCTION_NAME,                     no_write_effects},
  {ATOF_FUNCTION_NAME,                     no_write_effects},
  {STRTOD_FUNCTION_NAME,                   no_write_effects},
  /* Random number generators in stdlib.h */
  {RANDOM_FUNCTION_NAME,                   rgs_effects},
  {SRANDOM_FUNCTION_NAME,                  rgsi_effects},
  {RAND_FUNCTION_NAME,                     rgs_effects},
  {SRAND_FUNCTION_NAME,                    rgsi_effects},
  {DRAND48_FUNCTION_NAME,                  rgs_effects},
  {ERAND48_FUNCTION_NAME,                  rgs_effects},
  {JRAND48_FUNCTION_NAME,                  rgs_effects},
  {LRAND48_FUNCTION_NAME,                  rgs_effects},
  {MRAND48_FUNCTION_NAME,                  rgs_effects},
  {NRAND48_FUNCTION_NAME,                  rgs_effects},
  {SRAND48_FUNCTION_NAME,                  rgsi_effects},
  {SEED48_FUNCTION_NAME,                   rgsi_effects},
  {LCONG48_FUNCTION_NAME,                  rgsi_effects},

  /*#include <setjmp.h>*/

  {"setjmp",                               no_write_effects},
  {"__setjmp",                             no_write_effects},
  {"longjmp",                              no_write_effects},
  {"__longjmp",                            no_write_effects},
  {"sigsetjmp",                            no_write_effects},
  {"siglongjmp",                           no_write_effects},

  /*#include <stdio.h>*/
  // IO functions
  {FCLOSE_FUNCTION_NAME,                   c_io_effects},
  {FOPEN_FUNCTION_NAME,                    c_io_effects},
  {FPRINTF_FUNCTION_NAME,                  c_io_effects},
  {FSCANF_FUNCTION_NAME,                   c_io_effects},
  {ISOC99_FSCANF_FUNCTION_NAME,                   c_io_effects},
  {PRINTF_FUNCTION_NAME,                   c_io_effects},
  {SCANF_FUNCTION_NAME,                    c_io_effects},
  {ISOC99_SCANF_FUNCTION_NAME,                    c_io_effects},
  {SPRINTF_FUNCTION_NAME,                  c_io_effects},
  {SSCANF_FUNCTION_NAME,                   c_io_effects},
  {ISOC99_SSCANF_FUNCTION_NAME,                   c_io_effects},
  {VFPRINTF_FUNCTION_NAME,                 c_io_effects},
  {VPRINTF_FUNCTION_NAME,                  c_io_effects},
  {VFSCANF_FUNCTION_NAME,                  c_io_effects},
  {ISOC99_VFSCANF_FUNCTION_NAME,                  c_io_effects},
  {VSPRINTF_FUNCTION_NAME,                 c_io_effects},
  {VSNPRINTF_FUNCTION_NAME,                c_io_effects},
  {SNPRINTF_FUNCTION_NAME,                 c_io_effects},
  {VSSCANF_FUNCTION_NAME,                  c_io_effects},
  {ISOC99_VSSCANF_FUNCTION_NAME,                  c_io_effects},
  {VSCANF_FUNCTION_NAME,                   c_io_effects},
  {ISOC99_VSCANF_FUNCTION_NAME,                   c_io_effects},
  {FGETC_FUNCTION_NAME,                    c_io_effects},
  {FGETS_FUNCTION_NAME,                    c_io_effects},
  {FPUTC_FUNCTION_NAME,                    c_io_effects},
  {FPUTS_FUNCTION_NAME,                    c_io_effects},
  {GETC_FUNCTION_NAME,                     c_io_effects},
  {_IO_GETC_FUNCTION_NAME,                 c_io_effects},
  {PUTC_FUNCTION_NAME,                     c_io_effects},
  {_IO_PUTC_FUNCTION_NAME,                 c_io_effects},
  {GETCHAR_FUNCTION_NAME,                  c_io_effects},
  {PUTCHAR_FUNCTION_NAME,                  c_io_effects},
  {GETS_FUNCTION_NAME,                     c_io_effects},
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

  {REMOVE_FUNCTION_NAME,                           no_write_effects},
  {RENAME_FUNCTION_NAME,                           no_write_effects},
  {TMPFILE_FUNCTION_NAME,                          no_write_effects},
  {TMPNAM_FUNCTION_NAME,                           no_write_effects},
  {FFLUSH_FUNCTION_NAME,                           no_write_effects},
  {FREOPEN_FUNCTION_NAME,                          no_write_effects},
  {SETBUF_FUNCTION_NAME,                           no_write_effects},
  {SETVBUF_FUNCTION_NAME ,                         no_write_effects},

  {__FILBUF_FUNCTION_NAME,                         no_write_effects},
  {__FILSBUF_FUNCTION_NAME,                        no_write_effects},
  {SETBUFFER_FUNCTION_NAME,                        no_write_effects},
  {SETLINEBUF_FUNCTION_NAME,                       no_write_effects},
  {FDOPEN_FUNCTION_NAME,                           no_write_effects},
  {CTERMID_FUNCTION_NAME,                          no_write_effects},
  {FILENO_FUNCTION_NAME,                           no_write_effects},
  {POPEN_FUNCTION_NAME,                            no_write_effects},
  {CUSERID_FUNCTION_NAME,                          no_write_effects},
  {TEMPNAM_FUNCTION_NAME,                          no_write_effects},
  {GETOPT_FUNCTION_NAME,                           no_write_effects},
  {GETSUBOPT_FUNCTION_NAME,                        no_write_effects},
  {GETW_FUNCTION_NAME,                             no_write_effects},
  {PUTW_FUNCTION_NAME,                             no_write_effects},
  {PCLOSE_FUNCTION_NAME,                           no_write_effects},
  {FSEEKO_FUNCTION_NAME,                           no_write_effects},
  {FTELLO_FUNCTION_NAME,                           no_write_effects},
  {FOPEN64_FUNCTION_NAME,                          no_write_effects},
  {FREOPEN64_FUNCTION_NAME,                        no_write_effects},
  {TMPFILE64_FUNCTION_NAME,                        no_write_effects},
  {FGETPOS64_FUNCTION_NAME,                        no_write_effects},
  {FSETPOS64_FUNCTION_NAME,                        no_write_effects},
  {FSEEKO64_FUNCTION_NAME,                         no_write_effects},
  {FTELLO64_FUNCTION_NAME,                         no_write_effects},

  /* C IO system functions in man -S 2 unistd.h */

  {C_OPEN_FUNCTION_NAME,                           unix_io_effects},
  {CREAT_FUNCTION_NAME,                            unix_io_effects},
  {C_CLOSE_FUNCTION_NAME,                          unix_io_effects},
  {C_WRITE_FUNCTION_NAME,                          unix_io_effects},
  {C_READ_FUNCTION_NAME,                           unix_io_effects},
  {FCNTL_FUNCTION_NAME,                            unix_io_effects},
  {FSYNC_FUNCTION_NAME,                            unix_io_effects},
  {FDATASYNC_FUNCTION_NAME,                        unix_io_effects},
  {IOCTL_FUNCTION_NAME,                            unix_io_effects},
  {SELECT_FUNCTION_NAME,                           unix_io_effects},
  {PSELECT_FUNCTION_NAME,                          unix_io_effects},
  {STAT_FUNCTION_NAME,                             no_write_effects}, /* sys/stat.h */
  {FSTAT_FUNCTION_NAME,                            unix_io_effects},
  {LSTAT_FUNCTION_NAME,                            no_write_effects},

  /*#include <stdlib.h>*/
  {POSIX_MEMALIGN_FUNCTION_NAME,                   no_write_effects},
  {ABORT_FUNCTION_NAME,                            no_write_effects},
  {ABS_FUNCTION_NAME,                              no_write_effects},
  {ATEXIT_FUNCTION_NAME,                           no_write_effects},
  {ATOF_FUNCTION_NAME,                             no_write_effects},
  {ATOI_FUNCTION_NAME,                             no_write_effects},
  {ATOL_FUNCTION_NAME,                             no_write_effects},
  {ATOLL_FUNCTION_NAME,                            no_write_effects},
  {ATOQ_FUNCTION_NAME,                             no_write_effects},
  {BSEARCH_FUNCTION_NAME,                          no_write_effects},
  {CALLOC_FUNCTION_NAME,                           no_write_effects},
  {DIV_FUNCTION_NAME,                              no_write_effects},
  {EXIT_FUNCTION_NAME,                             no_write_effects},
  {FREE_FUNCTION_NAME,                             any_heap_effects},
  {LLABS_FUNCTION_NAME,                            no_write_effects},
  {LLDIV_FUNCTION_NAME,                            no_write_effects},
  {LLTOSTR_FUNCTION_NAME,                          safe_c_effects},
  {STRTOLL_FUNCTION_NAME,                          safe_c_effects},
  {STRTOULL_FUNCTION_NAME,                         safe_c_effects},
  {ULLTOSTR_FUNCTION_NAME,                          no_write_effects},


  /*  {char *getenv(const char *, 0, 0},
      {long int labs(long, 0, 0},
      {ldiv_t ldiv(long, long, 0, 0},*/

  {MALLOC_FUNCTION_NAME, any_heap_effects},
  {REALLOC_FUNCTION_NAME, any_heap_effects},

  /*#include <time.h>*/
  {TIME_FUNCTION_NAME,                           no_write_effects},

  /*#include <wchar.h>*/
  {FWSCANF_FUNCTION_NAME,                          c_io_effects},
  {SWSCANF_FUNCTION_NAME,                          c_io_effects},
  {WSCANF_FUNCTION_NAME,                           c_io_effects},

  /*#include <string.h>*/

  {MEMCPY_FUNCTION_NAME,                           no_write_effects},
  {MEMMOVE_FUNCTION_NAME,                          no_write_effects},
  {MEMCMP_FUNCTION_NAME,                           no_write_effects},
  {MEMSET_FUNCTION_NAME,                           no_write_effects},
  {STRCMP_FUNCTION_NAME,                           no_write_effects},
  {STRCPY_FUNCTION_NAME,                           no_write_effects},
  {STRNCPY_FUNCTION_NAME,                          no_write_effects},
  {STRCAT_FUNCTION_NAME,                           no_write_effects},
  {STRNCAT_FUNCTION_NAME,                          no_write_effects},
  {STRLEN_FUNCTION_NAME,                           no_write_effects},
  {STRCOLL_FUNCTION_NAME,                          no_write_effects},
  {STRNCMP_FUNCTION_NAME,                          no_write_effects},
  {STRXFRM_FUNCTION_NAME,                          no_write_effects},
  {MEMCHR_FUNCTION_NAME,                           no_write_effects},
  {STRCHR_FUNCTION_NAME,                           no_write_effects},
  {STRCSPN_FUNCTION_NAME,                          no_write_effects},
  {STRPBRK_FUNCTION_NAME,                          no_write_effects},
  {STRRCHR_FUNCTION_NAME,                          no_write_effects},
  {STRSPN_FUNCTION_NAME,                           no_write_effects},
  {STRSTR_FUNCTION_NAME,                           no_write_effects},
  {STRTOK_FUNCTION_NAME,                           no_write_effects},
  {STRERROR_FUNCTION_NAME,                         no_write_effects},
  {STRERROR_R_FUNCTION_NAME,                       no_write_effects},


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
    string s = entity_local_name(e);
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
   
    pips_error("generic_proper_effects_of_intrinsic", "unknown intrinsic %s\n", s);

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
 */
static list
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
address_expression_effects(entity op, list args)
{
    list le = list_undefined;
    expression ne = make_call_expression(op, args);
    call nc = syntax_call(expression_syntax(ne));

    pips_debug(5, "begin\n");
    le = generic_proper_effects_of_address_expression(ne, FALSE);
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

  le = (*effects_test_union_op)(lt, lf, effects_same_action_p);

  ifdebug(8) {
    pips_debug(8, "Effects for the two branches:\n");
    print_effects(le);
    (void) fprintf(stderr, "\n");
  }

  le = (*effects_union_op)(le, lc, effects_same_action_p);

  pips_debug(5, "end\n");
  return le;
}

static list
address_of_effects(entity f __attribute__ ((__unused__)),list args)
{
    list lr;
    expression e = EXPRESSION(CAR(args));
    syntax s = expression_syntax(e);
    reference r = syntax_reference(s);
    list i = reference_indices(r);

    pips_debug(5, "begin\n");
    pips_assert("address of has only one argument", gen_length(args)==1);
    /* FI: this is not true with "&c.a" */
    /*
    pips_assert("address of has only one argument and it is a reference",
		syntax_reference_p(s));
    */
    lr = generic_proper_effects_of_expressions(i);
    pips_debug(5, "end\n");
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
  return any_affect_effects(e, args, FALSE, FALSE);
}

static list
update_effects(entity e __attribute__ ((__unused__)),list args)
{
  return any_affect_effects(e, args, TRUE, FALSE);
}

static list
unique_update_effects(entity e __attribute__ ((__unused__)),list args)
{
  return any_affect_effects(e, args, TRUE, TRUE);
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
            pips_error("assign_substring_effects", "not a reference\n");


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
            pips_error("substring_effect", "not a reference\n");

    le = generic_proper_effects_of_expression(expr);
    le = gen_nconc(le, generic_proper_effects_of_expression(l));
    le = gen_nconc(le, generic_proper_effects_of_expression(u));

    pips_debug(5, "end\n");

    return(le);
}

static
 IoElementDescriptor*
SearchIoElement(char *s, char *i)
{
   IoElementDescriptor *p = IoElementDescriptorTable;

      while (p->StmtName != NULL) {
        if (strcmp(p->StmtName, s) == 0 && strcmp(p->IoElementName, i) == 0)
                return(p);
        p += 1;
    }

    pips_error("SearchIoElement", "unknown io element %s %s\n", s, i);
    /* Never reaches this point. Only to avoid a warning at compile time. BC. */
    return(&IoElementDescriptorUndefined);
}

/* return the appropriate C IO function.Amira Mensi*/
static
 IoElementDescriptor*
SearchCIoElement(char *s)
{
   IoElementDescriptor *p = IoElementDescriptorTable;

      while (p->StmtName != NULL) {
        if (strcmp(p->StmtName, s) == 0)
                return(p);
        p += 1;
    }

    pips_error("SearchCIoElement", "unknown io element %s\n", s);

    return(&IoElementDescriptorUndefined);
}


static list
some_io_effects(entity e __attribute__ ((__unused__)), list args __attribute__ ((__unused__)))
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

    private_io_entity = global_name_to_entity
      (IO_EFFECTS_PACKAGE_NAME,
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
                    pips_error("io_effects", "Which logical unit?\n");
            }

            indices = gen_nconc(indices, CONS(EXPRESSION, unit, NIL));

            private_io_entity = global_name_to_entity
                (IO_EFFECTS_PACKAGE_NAME,
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
  //bool file_p = TRUE;

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
      /* FILE * file descriptors are used */
      if(ENTITY_PRINTF_P(e) || ENTITY_PUTCHAR_P(e) ||
	 ENTITY_PUTS_P(e)|| ENTITY_VPRINTF_P(e))
	// The output is written into stdout
      unit = int_to_expression(STDOUT_FILENO);
    else if (ENTITY_SCANF_P(e) || ENTITY_GETS_P(e) || 
	     ENTITY_VSCANF_P(e) || ENTITY_GETCHAR_P(e))
      //The input is obtained from stdin
      unit = int_to_expression(STDIN_FILENO);
    else if (ENTITY_PERROR_P(e))
      unit = int_to_expression(STDERR_FILENO);
   
    else if(ENTITY_FOPEN_P(e))
      // the fopen function has the path's file as first argument.
      unit = make_unbounded_expression();
      
    else if(ENTITY_BUFFERIN_P(e) || ENTITY_BUFFEROUT_P(e))
      // the first argument is an integer specifying the logical unit
      // The expression should be evaluated and used if an integer is returned
      unit = make_unbounded_expression();
    
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
    
    private_io_entity = global_name_to_entity
      (IO_EFFECTS_PACKAGE_NAME, IO_EFFECTS_ARRAY_NAME);

    pips_assert("private_io_entity is defined", 
		private_io_entity != entity_undefined);

    ref1 = make_reference(private_io_entity, indices);
    ref2 = copy_reference(ref1);
    /* FI: I would like not to use "preference" isntead of
       "reference", but this causes a bug in cumulated effects and I
       do not have time to chase it. */
    eff1 = (*reference_to_effect_func)(ref1, is_action_read,false);
    eff2 = (*reference_to_effect_func)(ref2, is_action_write,false);

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
  return generic_io_effects(e, args, TRUE);
}

/* c_io_effects to handle the effects of functions of the "stdio.h" library. Amira Mensi*/
static list c_io_effects(entity e, list args)
{
  return generic_io_effects(e, args, FALSE);
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

  private_rgs_entity = global_name_to_entity
    (RAND_EFFECTS_PACKAGE_NAME,
     RAND_GEN_EFFECTS_NAME);

  pips_assert("gen_seed_effects", private_rgs_entity != entity_undefined);

  ref = make_reference(private_rgs_entity, indices);

  ifdebug(8) print_reference(ref);

  /* Read first. */
  if(init_p != TRUE){
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
  return any_rgs_effects( e, args, TRUE);
}

/* The seed is read and then written */
static list rgs_effects(entity e, list args)
{
  return any_rgs_effects( e, args, FALSE);
}

/* To handle the effects of heap related functions. */
static list any_heap_effects(entity e, list args)
{
  list le = NIL;
  list lep = NIL;
  entity malloc_entity = entity_undefined;
  reference ref;

  pips_debug(5, "begin for function \"%s\"\n", entity_user_name(e));

  MAP(EXPRESSION,exp,{
    lep = generic_proper_effects_of_expression(exp);
    le = gen_nconc(le, lep);
    //ifdebug(8) print_effects(le);
    //ifdebug(8) print_effects(lep);
  }, args);

  malloc_entity = global_name_to_entity
    (MALLOC_EFFECTS_PACKAGE_NAME,
     MALLOC_EFFECTS_NAME);

  pips_assert("malloc entity pre-exists", !entity_undefined_p(malloc_entity));

  ref = make_reference(malloc_entity, NIL);

  ifdebug(8) print_reference(ref);

  /* Read first. */
    le = gen_nconc(le, generic_proper_effects_of_read_reference(ref));

  /* Write back. */
  le = gen_nconc(le, generic_proper_effects_of_written_reference(ref));

  pips_debug(5, "end\n");

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
			       make_ram(global_name_to_entity(TOP_LEVEL_MODULE_NAME,
							      IO_EFFECTS_PACKAGE_NAME),
					global_name_to_entity(IO_EFFECTS_PACKAGE_NAME,
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
    else { /* C language */
      /* FI: we lack information about the number of elements written */
      /* This is not generic! */
      entity ioptr = make_dummy_io_ptr();
      reference r = make_reference(ioptr, CONS(EXPRESSION, make_unbounded_expression(), NIL));
      effect eff = (*reference_to_effect_func)(r, is_action_write,false);
      effect_approximation_tag(eff) = is_approximation_may;
      /* FI: this is really not generic! */
      extern list c_summary_effect_to_proper_effects(effect, expression);
      le = c_summary_effect_to_proper_effects(eff, exp);
      /* FI: We also need the read effects implied by the evaluation
	 of exp... but I do not know any function available to do
	 that. generic_proper_effects_of_expression() is ging to
	 return a bit too much. */
      
      
      if(FALSE) {
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
	    pips_internal_error("Operator \"\%s\" not handled\n", entity_name(op));
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
  return effects_of_any_ioelem(exp, act, TRUE);
}

/**
   Computes the effects of an io C intrinsic function on the actual
   argument arg according to the action tag act.
   proper effects on the evaluation of the argument are systematically added.

 @param arg is an actual argument of a C io intrinsic function
 @param act is an action tag as described in the  IoElementDescriptorTable.
        (its value can be either 'f', 's', 'r', 'w','x', 'v' or 'n').
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
    case 's':
      pips_debug(5, "stream or integer file descriptor case \n");

            /* We simulate actions on files by read/write actions
	       to a special static integer array.
	       GO:
	       It is necessary to do a read and write action to
	       the array, because it updates the file-pointer so
	       it reads it and then writes it ...
	    */
      
       indices = CONS(EXPRESSION, 
		      act == 'f'? unit : make_unbounded_expression(), 
		      NIL);
       must_p = act == 'f' ? true : false;
       private_io_entity = global_name_to_entity
	 (IO_EFFECTS_PACKAGE_NAME,
	  IO_EFFECTS_ARRAY_NAME);

       pips_assert("private_io_entity is defined\n", 
		   private_io_entity != entity_undefined);
       
       ref1 = make_reference(private_io_entity, indices);
       ref2 = copy_reference(ref1);
       /* FI: I would like not to use "preference" instead of
	  "reference", but this causes a bug in cumulated effects and I
	  do not have time to chase it. */
       eff1 = (*reference_to_effect_func)(ref1, is_action_read,false);
       eff2 = (*reference_to_effect_func)(ref2, is_action_write,false);
       
       if(!must_p) 
	 {
	   effect_approximation_tag(eff1) = is_approximation_may;
	   effect_approximation_tag(eff2) = is_approximation_may;
	 }
       le = gen_nconc(le, CONS(EFFECT, eff1, CONS(EFFECT, eff2, NIL)));
       /* and the effects on the file pointer */
       le = gen_nconc(le, c_actual_argument_to_may_summary_effects(arg, 'x'));

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
    case 'n':
      pips_debug(5, "only effects on actual argument evaluation\n");
      break;
    default :
      pips_internal_error("unknown tag\n");
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
        else if(act==is_action_write /* This is a read */
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
