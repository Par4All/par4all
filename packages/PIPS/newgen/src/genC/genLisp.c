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
/*

  This file includes the function used to implement user types
  in (Common)Lisp.

  The implementation is based on CL structures. We could provide type
  informations in order to (potentially) improve the efficiency.
  The type information (i.e., the index in the Domains table) is put here
  for compatibility with C only.

  . An inlined value is the value,
  . A list is a Lisp list,
  . An array is a Lisp array,
  . An AND type is a DEFSTRUCT,
  . An OR type is an OR structure with a tag (a keyword) and a value.

  Note that bool (which doesn't exist in Lisp) uses true and false, and
  so (if (foo-bar x) ...) as to be written (if (true? (foo-bar x))
  ...)

 */

#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif
#include <stdio.h>
#include "newgen_include.h"

#define TYPE(bp) (bp-Domains-Number_imports-Current_start)
#define NEWGEN_IMPL "#+akcl lisp:vector #-akcl (lisp:vector lisp:t)"

/* The name of the variable from which to start counting domain numbers. */

static char start[ 1024 ] ;

/* The package name in which functions will be defined. */

static char *package ;

struct gen_binding *Tabulated_bp ;

int Read_spec_mode ;			/* Unused in Lisp */

#define OR_VALUE_INDEX 3

static int or_counter ;

/* INIT_MEMBER returns the initialization code for a value in the domain DP. */

char *
init_member( dp )
     union domain *dp ;
{
  static char buffer[ 1024 ] ;

  switch( dp->ba.type ) {
  case BASIS_DT: {
    struct gen_binding *bp = dp->ba.constructand ;

    if( IS_INLINABLE( bp ))
	    sprintf( buffer, "%s", bp->inlined->Lisp_value ) ;
    else if( IS_EXTERNAL( bp ))
	    sprintf( buffer, "`(:external ,(lisp:+ %s %d) %s)", 
		     start, TYPE( bp ), ":external-undefined" ) ;
    else sprintf( buffer, ":undefined", bp->name ) ;
    break ;
  }
  case LIST_DT:
    sprintf( buffer, ":list-undefined" ) ;
    break ;
  case ARRAY_DT: {
    struct intlist *ilp ;

    sprintf( buffer, "(lisp:make-array '(" ) ;

    for( ilp = dp->ar.dimensions ; ilp != NULL ; ilp = ilp->cdr ) {
      strcat( buffer, itoa( ilp->val )) ;
      strcat( buffer, " " ) ;
    }
    strcat( buffer, ") :initial-element '--no-value--)" ) ;
    break ;
  }
  case SET_DT:
      sprintf( buffer, "(set:set-make)" ) ;
      break ;
  default:
    fatal( "init_member: Unknown type %s\n", itoa( dp->ba.type )) ;
    /*NOTREACHED*/
  }
  return( buffer ) ;
}

/* GEN_EXTERNAL generates the type code for external type BP. */

void
gen_external( bp )
struct gen_binding *bp ;
{
    printf( "(lisp:defvar %s (lisp:+ %s %d))\n", bp->name, start, TYPE( bp )) ;
}

/* GEN_EXTERNAL_MEMBER generates the manipulation functions for a possible
   external  member (either and or or) in domain DP and OFFSET. It returns
   whether some code has been generated or not. 

   See INIT_MEMBER to understand mutation code. */

int
gen_external_member( dp, offset )
union domain *dp ;
int offset ;
{
    if( dp->ba.type == BASIS_DT ) {
	struct gen_binding *bp = dp->ba.constructand ;
	
	if( !IS_INLINABLE( bp ) && IS_EXTERNAL( bp )) {
	    printf( "(lisp:defun %s-%s (and)\n", bp->name, dp->ba.constructor ) ;
	    printf( "  (lisp:caddr (lisp:svref and %d)))\n", offset ) ;
	    
	    printf( "(lisp:defsetf %s-%s (and) (new-and)\n", 
		   bp->name, dp->ba.constructor ) ;
	    printf( "  `(lisp:setf (lisp:caddr (lisp:svref ,and %d)) ", offset ) ;
	    printf( ",new-and))\n" ) ;
	    return( 1 ) ;
	}
    }
    return( 0 ) ;
}

/* GEN_PRELUDE generates prelude declarations for potentially tabulated 
   domain BP. */

static void
gen_prelude( bp )
struct gen_binding *bp ;
{
    printf( "(lisp:setf (lisp:aref newgen::*gen-tabulated-alloc*" ) ;
    printf( " (lisp:+ %s %d)) %d)\n",
	   start, TYPE( bp ), (IS_TABULATED( bp )) ? 0 : -1 ) ;
    printf( "(lisp:defvar %s::%s-domain (lisp:+ %s %d))\n",
	   package, bp->name, start, TYPE( bp )) ;

    if( IS_TABULATED( bp )) {
	printf( "(lisp:setf (lisp:aref newgen::*gen-tabulated-index* " ) ;
	printf( "%s::%s-domain) %d)\n", package, bp->name, bp->index ) ;
    }
}

/* GEN_POSTLUDE generates tabulation table updates. */

static void
gen_postlude( bp )
struct gen_binding *bp ;
{
    if( IS_TABULATED( bp )) {
	printf( "(lisp:defvar old-make-%s)\n", bp->name ) ;
	printf( "(lisp:setf old-make-%s (lisp:symbol-function 'make-%s))\n",
	        bp->name, bp->name ) ;
	printf( "(lisp:fmakunbound 'make-%s)\n", bp->name) ;
	printf( "(lisp:setf (lisp:symbol-function 'make-%s)\n", bp->name ) ;
	printf( " #'(lisp:lambda (lisp:&rest args)\n", bp->name ) ;
 	printf( " (lisp:let ((node (lisp:apply old-make-%s args)))\n", bp->name ) ;
	printf( "  (newgen::enter-tabulated-def\n" ) ;
	printf( "   (lisp:aref newgen::*gen-tabulated-index* %s-domain)\n",
	        bp->name ) ;
	printf( "  %s-domain\n", bp->name ) ;
	printf( "  (lisp:aref node %d)\n", HASH_OFFSET ) ;
	printf( "  node\n" ) ;
	printf( "  :allow-ref-p lisp:nil)\n" ) ;
	printf( "  node)))\n" ) ;
    }
}

/* GEN_TYPE generates the type member for potentially tabulated BP domain. 
 */
static generate_type_member( bp )
struct gen_binding *bp ;
{
    printf( " (-type- `(:newgen ,(lisp:+ %s %d)))\n", start, TYPE( bp ) ) ;

    if( IS_TABULATED( bp )) {
	printf( " (-tabular- (newgen::find-free-tabulated %d))\n", 
	        TYPE( bp )) ;
    }
}

/* GEN_AND generates the manipulation functions for an AND type BP. */

void
gen_and( bp )
struct gen_binding *bp ;
{
    union domain *dom = bp->domain ;
    struct domainlist *dlp = dom->co.components ;
    int size ;

    gen_prelude( bp ) ;
    printf( "(lisp:defstruct (%s (:type %s))\n", bp->name, NEWGEN_IMPL ) ;
    generate_type_member( bp ) ;

    for( ; dlp != NULL ; dlp=dlp->cdr ) {
	union domain *dp = dlp->domain ;

	printf( " (%s %s)\n", dp->ba.constructor, init_member( dp )) ;
    }
    printf( ")\n" ) ;

    for( size=2, dlp=dom->co.components; dlp != NULL ; dlp=dlp->cdr, size++ ) {
	gen_external_member( dlp->domain, size ) ;
    }
    gen_postlude( bp ) ;
}

/* GEN_OR generates the manipulation function for an OR_OP type BP. Note
   that for a UNIT_TYPE_NAME, no access function is defined since the value is
   meaningless. */

void
gen_or( bp )
     struct gen_binding *bp ;
{
    char *name = bp->name ;
    union domain *dom = bp->domain ;
    struct domainlist *dlp ;
    char *or_impl = (IS_TABULATED( bp )) ? "tabular-or" : "or" ;

    gen_prelude( bp ) ;
    printf( "(lisp:defun make-%s ", name ) ;
    printf( "(tag lisp:&optional (val :unit))\n" );
    printf( " (lisp:let ((node (newgen::make-%s)))\n", or_impl) ;
    printf( "   (lisp:setf (newgen::%s-type node) `(:newgen ,(lisp:+ %s %d)))\n", 
	   or_impl, start, TYPE( bp ) ) ;

    if( IS_TABULATED( bp )) {
	printf( "   (lisp:setf (newgen::%s-tabular node) ", or_impl ) ;
	printf( "(newgen::find-free-tabulated (lisp:+ %s %d)))\n", 
	        start, TYPE( bp )) ;
    }
    printf( "   (lisp:setf (newgen::%s-tag node) tag)\n", or_impl ) ;
    printf( "   (lisp:setf (newgen::%s-val node) val)\n", or_impl ) ;
    printf( "   node))\n" ) ;
    printf( "(lisp:defmacro %s-tag(node) `(newgen::%s-tag ,node))\n", 
	    name, or_impl ) ;

    for( dlp=dom->co.components ; dlp != NULL ; dlp=dlp->cdr, or_counter++ ){
	union domain *dp = dlp->domain ;

	printf( "(lisp:defconstant is-%s-%s %d)\n",
	       name, dp->ba.constructor, or_counter ) ;
	printf( "(lisp:setf newgen::*tag-names* " ) ;
        printf( "(lisp:acons %d 'is-%s-%s newgen::*tag-names*))\n",
	       or_counter, name, dp->ba.constructor ) ;
	printf( "(lisp:defmacro %s-%s-p (or) ", name, dp->ba.constructor ) ;
	printf( "`(lisp:= (%s-tag ,or) is-%s-%s))\n", 
	       name, name, dp->ba.constructor ) ;

	if( dp->ba.type == BASIS && 
	   strcmp( dp->ba.constructand->name, UNIT_TYPE_NAME ) == 0 ||
	   gen_external_member( dp, OR_VALUE_INDEX )) 
		continue ;

	printf( "(lisp:defmacro %s-%s (or) `(newgen::%s-val ,or))\n", 
	       name, dp->ba.constructor, or_impl ) ;
    }
    gen_postlude( bp ) ;
}

/* GEN_LIST defines the manipulation functions for a list type BP. */

void
gen_list( bp )
     struct gen_binding *bp ;
{
    gen_prelude( bp ) ;
    printf( "(lisp:defstruct (%s (:type %s))\n", bp->name, NEWGEN_IMPL ) ;
    generate_type_member( bp ) ;
    printf( " (%s '()))\n", bp->domain->li.constructor ) ;
    gen_postlude( bp ) ;
}

/* GEN_ARRAY defines the manipulation functions for an array type BP. */

void
gen_array( bp )
     struct gen_binding *bp ;
{
    union domain *dom = bp->domain ;

    gen_prelude( bp ) ;
    printf( "(lisp:defstruct (%s (:type %s))\n", bp->name, NEWGEN_IMPL ) ;
    generate_type_member( bp ) ;
    printf( " (%s %))\n", 
	   dom->ar.constructor, init_member( dom->ar.element->domain )) ;
    gen_postlude( bp ) ;
}

/* GEN_SET defines the manipulation functions for a set type BP. */

void
gen_set( bp )
     struct gen_binding *bp ;
{
    union domain *dom = bp->domain ;

    gen_prelude( bp ) ;
    printf( "(lisp:defstruct (%s (:type %s))\n", bp->name, NEWGEN_IMPL ) ;
    generate_type_member( bp ) ;
    printf( " (%s %))\n", 
	   dom->se.constructor, init_member( dom->se.element->domain )) ;
    gen_postlude( bp ) ;
}

/* GEN_DOMAIN generates the manipulation functions for a type BP. This is
   manily a dispatching function. */

void
gen_domain( bp )
     struct gen_binding *bp ;
{
  union domain *dp = bp->domain ;

  if( !IS_INLINABLE( bp ) && !IS_EXTERNAL( bp )) {
      printf( "(lisp:defmacro write-%s (fd obj) `(gen-write ,fd ,obj))\n",
	        bp->name, bp->name ) ;
      printf("(lisp:defmacro read-%s (lisp:&optional (fd *standard-input*))\n",
	     bp->name ) ;
      printf("  `(gen-read ,fd))\n");
  }
  switch( dp->ba.type ) {
  case CONSTRUCTED_DT:
    if( dp->co.op == AND_OP ) gen_and( bp ) ;
    else if( dp->co.op == OR_OP ) gen_or( bp ) ;
    else fatal( "gen_domain: Unknown constructed %s\n", itoa( dp->co.op )) ;
    break ;
  case LIST_DT:
    gen_list( bp ) ;
    break ;
  case ARRAY_DT:
    gen_array( bp ) ;
    break ;
 case SET_DT:
    gen_set( bp ) ;
    break ;
  case EXTERNAL_DT:
    gen_external( bp ) ;
    break ;
  default:
    fatal( "gen_domain: Unknown type %s\n", itoa( dp->ba.type )) ;
  }
}

/* GENCODE generates the code necessary to manipulate every non-inlinable
   type in the Domains table. */

void
gencode( file )
char *file ;
{
    struct gen_binding *bp ;
    int domain_count = 0 ;
  
    or_counter = 0 ;
    package = file ;
    sprintf( start, "newgen::*gen-%s-start*", file ) ;

    for( bp = Domains ; bp < &Domains[ MAX_DOMAIN ] ; bp++ ) {
	if( bp->name == NULL ||
	    IS_INLINABLE( bp ) || IS_IMPORT( bp ) || bp == Tabulated_bp ) 
		continue ;

	gen_domain( bp ) ;	
    }
}
	
