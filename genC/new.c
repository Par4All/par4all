/*

	-- NewGen Project

	The NewGen software has been designed by Remi Triolet and Pierre
	Jouvelot (Ecole des Mines de Paris). This prototype implementation
	has been written by Pierre Jouvelot.

	This software is provided as is, and no guarantee whatsoever is
	provided regarding its appropriate behavior. Any request or comment
	should be sent to newgen@cri.ensmp.fr.

	(C) Copyright Ecole des Mines de Paris, 1989

	$Id$
 */

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
