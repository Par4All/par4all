/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

  This file is part of NewGen.

  NewGen is free software: you can redistribute it and/or modify it under the
  terms of the GNU General Public License as published by the Free Software
  Foundation, either version 3 of the License, or any later version.

  NewGen is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
  License for more details.

  You should have received a copy of the GNU General Public License along with
  NewGen.  If not, see <http://www.gnu.org/licenses/>.

*/
#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif

#include <stdio.h>
#include "genC.h"
#include "newgen_include.h"

extern int build();
extern FILE *genspec_in, *genspec_out;

/* MAIN is the C entry (in fact a renaming for BUILD). */

#ifdef GENSPEC_DEBUG
extern int genspec_debug ;
#endif

int main(
    int argc,
    char *argv[])
{
#ifdef GENSPEC_DEBUG
    genspec_debug = 0 ; 
#endif
    Read_spec_mode = 0 ;

    /* explicit initialization (lex default not assumed)
     */
    genspec_in = stdin;
    genspec_out = stdout;
    return build(argc, argv);
}
