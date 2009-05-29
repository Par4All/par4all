/*

  $Id$

  Copyright 1989-2009 MINES ParisTech

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

#ifndef newgen_types_included
#define newgen_types_included

/*

  The implementation of the basic types UNIT, BOOL, TAG and STRING. The
  others CHAR, INT and FLOAT are provided by C.

*/

/* same as linear boolean.h */
#ifndef BOOLEAN_INCLUDED
#define BOOLEAN_INCLUDED
typedef enum { false, true } boolean;
#ifndef TRUE /* defined by glib2.0 AS (!FALSE) */
#define	TRUE     true
#endif /* TRUE */
#ifndef FALSE /* idem AS (0) */
#define	FALSE    false
#endif /* FALSE */
#endif /* BOOLEAN_INCLUDED */

/* newgen compatibility */
typedef boolean bool;

/* STRING
 */
#ifdef string
#undef string
#endif
typedef char *string ;

#ifndef _INT_TYPE_DEFINED
/* RK wants an "int" which is the size of a "pointer".
 * However, the use of the relevant "intptr_t" type in the code
 * amounts to source file defacing, hence this definition. FC.
 * The ifndef macro is to avoid a double definition it the type
 * needs to be defined in the C3/linear library as well.
 */
typedef intptr_t _int;
typedef uintptr_t _uint;
#define _INT_TYPE_DEFINED
#endif /* _INT_TYPE_DEFINED */

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
