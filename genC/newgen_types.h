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


#ifndef TYPES_INCLUDED
#define TYPES_INCLUDED

/* -- $RCSfile: newgen_types.h,v $ (version $Revision$)

   The implementation of the basic types UNIT, BOOL, TAG and STRING. The
   others CHAR, INT and FLOAT are provided by C. 

*/

typedef int bool;
#define	TRUE     1
#define	FALSE    0

/* STRING
 */
#ifdef string
#undef string
#endif
typedef char *string ;

/* this is the disk representation of an undefined string
 */
#define disk_string_undefined "## Newgen: shouldn't appear, indeed... ##\n"
#define disk_string_undefined_p(s) (strcmp(s,disk_string_undefined)==0)

/* to be used for core memory
 */
#define string_undefined ((string)-15)
#define string_undefined_p(s) ((s)==string_undefined)
#define copy_string(s) strdup(s)


typedef int tag;
#define tag_undefined (-3)

typedef int unit ;
#define UU 0

#define array_undefined NULL
#define array_undefined_p(a) ((a)==NULL)

union domain ;
struct inlinable ;
struct gen_binding {
  char *name ;
  int compiled ;
  int index ;
  int alloc ;
  union domain *domain ;
  struct inlinable *inlined ;
} ;

#ifdef __STRICT_ANSI__
#define GEN_PROTO(x) x
#else
#ifdef __STDC__
#define GEN_PROTO(x) x
#else
#define GEN_PROTO(x) ()
#endif /* __STDC__ */
#endif /* __STRICT_ANSI__ */

#endif /* file */

/*  end of $RCSfile: newgen_types.h,v $
 */
