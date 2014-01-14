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

#ifndef newgen_types_included
#define newgen_types_included

/*

  The implementation of the basic types UNIT, BOOL, TAG and STRING. The
  others CHAR, INT and FLOAT are provided by C.

*/


/* STRING
 */
#ifdef string
#undef string
#endif
typedef char *string ;
#define string_undefined ((string)-15)
#define string_undefined_p(s) ((s)==string_undefined)
#define copy_string(s) strdup(s)

/* _INT
 */
#ifndef _int_undefined
/* RK wants an "int" which is the size of a "pointer".
 * However, the use of the relevant "intptr_t" type in the code
 * amounts to source file defacing, hence this definition. FC.
 * The ifndef macro is to avoid a double definition if the type
 * needs to be defined in the C3/linear library as well.
 */
typedef intptr_t _int;
typedef uintptr_t _uint;
// also add corresponding format string
#include <inttypes.h>
#define _intFMT PRIuPTR
#define _uintFMT "u" PRIuPTR
#define _int_undefined ((_int)-15) /* SG: this is dangerous: a valid value is used to state an invalid value */
#define _int_undefined_p(i) ((i)==_int_undefined)

#endif /* _int_undefined */

/* BOOL
 */

/* SG: _Bool is not compatible with newgen because it does not permit the definition of an `undefined' state
   we use the int type for compatible behavior */
#ifndef BOOLEAN_INCLUDED /* similar to linear ... */
#define BOOLEAN_INCLUDED

#ifdef bool
    #error newgen header not compatible with stdbool.h
#endif

/* we cannot use an enum or stdbool because we need to be compatible with newgen,
 * thus boolean need to handle 3 states : true, false, and undefined  */
typedef int bool;

#define false 0
#define true 1
#endif

#define bool_undefined ((bool)-15) /* SG: this is a dangerous semantic: bool_undefined evaluates to true ... */
#define bool_undefined_p(b) ((b)==bool_undefined)




/* TAG
 */
typedef int tag;
#define tag_undefined (-3)

/* UNIT
 */
typedef int unit ;
#define UU ((void*)0)
#define UUINT(i) ((void*)(i))

/* ARRAY
 */
#define array_undefined NULL
#define array_undefined_p(a) ((a)==NULL)

typedef struct cons * list;

// functional types
typedef bool (*gen_filter_func_t)(const void *);
typedef bool (*gen_filter2_func_t)(const void *, const void *);
typedef string (*gen_string_func_t)(const void *);
// for qsort: void * points to a pointer to the newgen type
// so it is really "gen_chunk**", i.e. "entity*" or "statement*"
typedef int (*gen_cmp_func_t)(const void *, const void *);
typedef bool (*gen_eq_func_t)(const void *, const void *);
typedef void (*gen_iter_func_t)(void *);

// obsolete?
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
