/*

	-- NewGen Project

	The NewGen software has been designed by Remi Triolet and Pierre
	Jouvelot (Ecole des Mines de Paris). This prototype implementation
	has been written by Pierre Jouvelot.

	This software is provided as is, and no guarantee whatsoever is
	provided regarding its appropriate behavior. Any request or comment
	should be sent to newgen@isatis.ensmp.fr.

	(C) Copyright Ecole des Mines de Paris, 1989

*/


/* new.c */

#include <stdio.h>
#include "newgen_include.h"

extern int build();

/* MAIN is the C entry (in fact a renaming for BUILD). */

#ifdef ZZDEBUG
extern int zzdebug ;
#endif

int main( argc, argv )
int argc;
char *argv[] ;
{
#ifdef ZZDEBUG
    zzdebug = 0 ; 
#endif
    Read_spec_mode = 0 ;
    return( build( argc, argv )) ;
}
