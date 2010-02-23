/* Copyright 2007, 2008 Alain Muller, Frederique Silber-Chaussumier

This file is part of STEP.

The program is distributed under the terms of the GNU General Public
License.
*/


/*

Module STEP

regroupement de la majorité des includes

*/
// includes C standard
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <ctype.h>
#include <string.h>

// include C3&linear

#include "linear.h"

// include NEWGEN

#include "genC.h"
#include "text.h"

#include "ri.h"
#include "properties.h"

#include "step_private.h"
#include "outlining_private.h"

// PIPS

#include "misc.h"
#include "text-util.h"
#include "ri-util.h"
#include "pipsdbm.h"
#include "resources.h"
#include "step.h"
#include "control.h"
#include "parser_private.h"
#include "syntax.h"
#include "transformations.h"
#include "callgraph.h"

/* debug macro
 */
extern int the_current_debug_level;
#define GEN_DEBUG(D, W, P) ifdebug(D) { pips_debug(D, "%s:\n", W); P;}
#define STEP_DEBUG_STATEMENT(D, W, S) GEN_DEBUG(D, W, print_statement(S))
#define STEP_DEBUG_CODE(D, W, M, S) GEN_DEBUG(D, W, step_print_code(stderr, M, S))
#define STEP_DEBUG_DIRECTIVE(D, W, DRT) GEN_DEBUG(D, W, step_print_directive(DRT))

#define STEP_SF_SUFFIX ".step_sf"
#define STEP_MPI_SUFFIX "_MPI"
#define STEP_OMP_SUFFIX "_OMP"


#define STEP_DEFAULT_RT_H "src/Scripts/step/default/"

enum {DO_DIRECTIVE,
      SECTION_DIRECTIVE};

#define NAME_BUFFERSIZE 3


/* OMP directives text
*/
#define OMP_DIRECTIVE "!$OMP "
#define OMP_DIR_CONT  "!$OMP&"


#define PARALLEL_TEXT "parallel"
#define END_PARALLEL_TEXT "end parallel"

#define DO_TEXT "do"
#define END_DO_TEXT "end do"
#define PARALLEL_DO_TEXT "parallel do"
#define END_PARALLEL_DO_TEXT "end parallel do"

#define SECTIONS_TEXT "sections"
#define END_SECTIONS_TEXT "end sections"
#define PARALLEL_SECTIONS_TEXT "parallel sections"
#define END_PARALLEL_SECTIONS_TEXT "end parallel sections"
#define SECTION_TEXT "section"

#define MASTER_TEXT "master"
#define END_MASTER_TEXT "end master"

#define BARRIER_TEXT "barrier"

/* Suffix for generated OMP directives routines
*/
#define SUFFIX_OMP_PARALLEL "par"
#define SUFFIX_OMP_END_PARALLEL "par_"

#define SUFFIX_OMP_DO "do"
#define SUFFIX_OMP_END_DO "do_"
#define SUFFIX_OMP_PARALLEL_DO "pardo"
#define SUFFIX_OMP_END_PARALLEL_DO "pardo_"

#define SUFFIX_OMP_SECTIONS "sections"
#define SUFFIX_OMP_PARALLEL_SECTIONS "parsections"
#define SUFFIX_OMP_SECTION "sect"

#define SUFFIX_OMP_MASTER "master"
#define SUFFIX_OMP_END_MASTER "master_"

#define SUFFIX_OMP_BARRIER "barrier"
