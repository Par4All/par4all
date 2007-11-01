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


#ifndef newgen_types_included
#define newgen_types_included

/*
  
  $Id$

  The implementation of the basic types UNIT, BOOL, TAG and STRING. The
  others CHAR, INT and FLOAT are provided by C. 

*/

/* same as linear boolean.h */
#ifndef BOOLEAN_INCLUDED
#define BOOLEAN_INCLUDED
typedef enum { false, true } boolean;
#define	TRUE     true
#define	FALSE    false
#endif /* BOOLEAN_INCLUDED */

/* newgen compatibility */
typedef boolean bool;

/* STRING
 */
#ifdef string
#undef string
#endif
typedef char *string ;

#define string_undefined ((string)-15)
#define string_undefined_p(s) ((s)==string_undefined)
#define copy_string(s) strdup(s)

typedef int tag;
#define tag_undefined (-3)

typedef int unit ;
#define UU ((void*)0)
#define UUINT(i) ((void*)(i))

#define array_undefined NULL
#define array_undefined_p(a) ((a)==NULL)

#ifdef __STRICT_ANSI__
#define GEN_PROTO(x) x
#else
#ifdef __STDC__
#define GEN_PROTO(x) x
#else
#define GEN_PROTO(x) ()
#endif /* __STDC__ */
#endif /* __STRICT_ANSI__ */

#endif /* newgen_types_included */
