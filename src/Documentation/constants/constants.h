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
/* this file is maintained under Documentation/constants.h
 */

/* the following use to be "constants.h" alone in Include.
 * I put it there not to lose it someday. FC.
 */

#include "specs.h"

/* Auxiliary data files
 */

#define PIPSMAKE_RC "pipsmake.rc"

#define WPIPS_RC "wpips.rc"

#define XV_HELP_RC "pips_help.txt"

/* Default name for a property file */
#define PROPERTIES_RC "properties.rc"
/* Name of the file that contains the name of an old property file,
   such as properties-2009-10-31.rc. This is used to keep
   non-regression tests alive. */
#define OLD_PROPERTIES_RC "old_properties"

/* filename extensions
 */
#define SEQUENTIAL_CODE_EXT ".code"
#define PARALLEL_CODE_EXT ".parcode"

#define SEQUENTIAL_FORTRAN_EXT ".f"
#define SEQUENTIAL_C_EXT ".c"
#define PARALLEL_FORTRAN_EXT ".par.f"
#define PARALLEL_C_EXT ".par.c"

/* Suffixes for code and parsed_code. No idea why the word PREDICAT
   is used for code. Old misusage? */
#define PREDICAT_FORTRAN_EXT ".pre.f"
#define PRETTYPRINT_FORTRAN_EXT ".pp.f"
#define PREDICAT_C_EXT ".pre.c"
#define PRETTYPRINT_C_EXT ".pp.c"

#define WP65_BANK_EXT ".bank"
#define WP65_COMPUTE_EXT ".wp65"

#define ENTITIES_EXT ".entities"

#define EMACS_FILE_EXT "-emacs"

#define GRAPH_FILE_EXT "-graph"


/* Some directory names... */

/* Where is the output of HPFC in the workspace: */
#define HPFC_COMPILED_FILE_DIR "hpfc"
#define COMPLEXITY_COST_TABLES "complexity_cost_tables"
   
/* say that's all
 */
