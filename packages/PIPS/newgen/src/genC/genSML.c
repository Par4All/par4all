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
#include <ctype.h>
#include <string.h>
#include "newgen_include.h"
#include "genC.h"

#undef GEN_HEADER
/* For simplicity, the tabulation slot is always here */
#define GEN_HEADER 2

#define IS_NON_INLINABLE_BASIS(f) (strcmp(f,"chunk")==0)
#define UPPER(c) ((islower( c )) ? toupper( c ) : c )
#define TYPE(bp) (bp-Domains-Number_imports-Current_start)

#define OR_TAG_OFFSET 2

static char start[ 1024 ] ;

int Read_spec_mode ;

struct gen_binding *Tabulated_bp ;

/* GEN_SIZE returns the size (in gen_chunks) of an object of type defined by
   the BP type. */

int
gen_size( bp )
struct gen_binding *bp ;
{
    int overhead = GEN_HEADER ;

    switch( bp->domain->ba.type ) {
    case BASIS_DT:
    case ARRAY_DT:
    case LIST_DT:
    case SET_DT:
	return( overhead + 1 ) ;
    case CONSTRUCTED_DT:
	if( bp->domain->co.op == OR_OP )
		return( overhead + 2 ) ;
	else {
	    int size ;
	    struct domainlist *dlp = bp->domain->co.components ;

	    for( size=0 ; dlp != NULL ; dlp=dlp->cdr, size++ )
		    ;
	    return( overhead + size ) ;
	}
    default:
	fatal( "gen_size: Unknown type %s\n", itoa( bp->domain->ba.type )) ;
	/*NOTREACHED*/
    }
}

/* PRIMITIVE_FIELD returns the appropriate field to acces an object in BP.
   Note how inlined types are managed (see genC.h comments). */

static char *
primitive_field( dp )
     union domain *dp ;
{
    static char buffer[ 1024 ];

    switch( dp->ba.type ) {
    case BASIS_DT: {
	struct gen_binding *bp = dp->ba.constructand ;

	if( IS_INLINABLE( bp )) {
	    sprintf( buffer, "%s", bp->name ) ;
	}
	else if( IS_EXTERNAL( bp )) {
	    fprintf(stderr, "primitive_field not implemented\n");
	}
	else sprintf( buffer, "chunk" ) ;
	break ;
    }
    case LIST_DT:
	sprintf( buffer, "list" ) ;
	break ;
    case SET_DT:
	sprintf( buffer, "set" ) ;
	break ;
    case ARRAY_DT: 
	sprintf( buffer, "vector" ) ;
	break ;
    default:
	fatal( "primitive_field: Unknown type %s\n", itoa( dp->ba.type )) ;
	/*NOTREACHED*/
    }
    return( buffer ) ;
}

/* GEN_MEMBER generates a member access functions for domain DP and
   OFFSET. NAME is the domain of the defined domain. */

static void
gen_member( name, dp, offset)
char *name ;
union domain *dp ;
int offset ;
{
    char *field = primitive_field( dp ) ;

    if( dp->ba.type == BASIS_DT && 
       strcmp( dp->ba.constructand->name, UNIT_TYPE_NAME ) == 0 )
	    return ;

    (void) printf( "fun %s_%s (vector node) = ", name, dp->ba.constructor ) ;
    (void) printf("case (sub (node,%d)) of (%s x) => x;\n",
		  offset, field) ;
}

/* GEN_ARG returns the constructor name of domain DP. */

static char *
gen_arg( dp )
union domain *dp ;
{
    return( (dp->ba.type == BASIS_DT) ? dp->ba.constructor :
	    (dp->ba.type == LIST_DT) ? dp->li.constructor :
	    (dp->ba.type == SET_DT) ? dp->se.constructor :
	    (dp->ba.type == ARRAY_DT) ? dp->ar.constructor :
	    (fatal( "gen_arg: Unknown type %s\n", itoa( dp->ba.type )), 
	     (char *)NULL) ) ;
}

/* GEN_ARGS returns a comma-separated list of constructor names for the list
   of domains DLP. */

static char *
gen_args( dlp )
struct domainlist *dlp ;
{
    static char buffer[ 1024 ] ;

    for( sprintf( buffer, "" ) ; dlp->cdr != NULL ; dlp = dlp->cdr ) {
	strcat( buffer, gen_arg( dlp->domain )) ;
	strcat( buffer, ", " )  ;
    }
    strcat( buffer, gen_arg( dlp->domain )) ;
    return( buffer ) ;
}

static char *
gen_update_arg( dp, index  )
struct domain *dp ;
int index ;
{
    static char arg[ 1024 ] ;

    sprintf(arg, "update(gen_v, %d, %s %s);", index, 
	    primitive_field(dp), gen_arg(dp));
    return(arg) ;
}

/* GEN_UPDATE_ARGS returns a comma-separated list of constructor names
for the list of domains DLP. */

static char *
gen_update_args( dlp )
struct domainlist *dlp ;
{
    static char buffer[ 1024 ] ;
    int index = GEN_HEADER ;

    for( sprintf(buffer, "") ; dlp->cdr!=NULL ; dlp=dlp->cdr, index++ ) {
	strcat(buffer, gen_update_arg(dlp->domain, index)) ;
    }
    strcat(buffer, gen_update_arg(dlp->domain, index)) ;
    return( buffer ) ;
}

/* GEN_MAKE generates the gen_alloc call for gen_bindings BD with SIZE user
   members and ARGS as list of arguments. */

static void
gen_make( bp, size, args, updated_args, option_name )
struct gen_binding *bp ;
int size ;
char *args, *updated_args, *option_name ;
{
    extern int printf();

    (void) printf("fun make_%s%s%s (%s) =", bp->name, 
		  (strcmp(option_name, "") == 0 ? "" : "_"), option_name,
		  args) ;
    (void) printf(" let val gen_v = array(%d+%d, undefined) in\n",
		  GEN_HEADER, size);
    (void) printf(" update(gen_v, 0, int (%s+%d));\n", start, TYPE(bp)) ;
    (void) printf(" %s\n", updated_args);

    if( IS_TABULATED( bp )) {
	fprintf(stderr, "enter_tabulated not called\n");
    }
    (void) printf(" vector gen_v end:%s;\n", bp->name) ;

    if( IS_TABULATED( bp )) {
	printf( "val %s_domain = %s+%d;\n", bp->name, start, TYPE( bp )) ;
    }
}

/* GEN_AND generates the manipulation functions for an AND type BP. */

void
gen_and( bp )
     struct gen_binding *bp ;
{
    union domain *dom = bp->domain ;
    struct domainlist *dlp ;
    int size ;

    gen_make(bp, gen_size(bp)-GEN_HEADER,
	     gen_args(dom->co.components), 
	     gen_update_args(dom->co.components),
	     "") ;
    size = GEN_HEADER ;

    for( dlp=dom->co.components ; dlp != NULL ; dlp=dlp->cdr )
	    gen_member( bp->name, dlp->domain, size++ ) ;
}

/* GEN_OR generates the manipulation function for an OR_OP type BP. Note
   that for a UNIT_TYPE_NAME, no access function is defined since the value is
   meaningless. */

void
gen_or( bp )
struct gen_binding *bp ;
{
    extern int printf();
    char *name = bp->name ;
    union domain *dom = bp->domain ;
    struct domainlist *dlp ;
    int offset ;
  
    (void) printf("fun %s_tag (vector or) = ", name ) ;
    (void) printf("case (sub (or,%d)) of (int x) => x;\n", 
		  GEN_HEADER) ;

    for( dlp=dom->co.components,offset=dom->co.first ;
	dlp != NULL ; 
	dlp=dlp->cdr, offset++ ) {
	union domain *dp = dlp->domain ;
	static char args[ 1024 ] ;
	static char updated_args[ 1024 ] ;

	sprintf(args, "tag, %s", dp->ba.constructor ) ;
	strcpy(updated_args, "update(gen_v, 2, int tag);") ;
	gen_make(bp, 2,args , 
		 strcat(updated_args, gen_update_arg(dp, 3)),
		 dp->ba.constructor);
	(void) printf("val is_%s_%s = %d;\n",
		      name, dp->ba.constructor, offset ) ;
	(void) printf("fun %s_%s_p or = ((%s_tag or)=is_%s_%s);\n",
		      name, dp->ba.constructor, name, 
		      name, dp->ba.constructor ) ;
	gen_member( name, dp, OR_TAG_OFFSET ) ;
    }
}

/* GEN_LIST defines the manipulation functions for a list type BP. */

void
gen_list( bp )
struct gen_binding *bp ;
{
    extern int printf();
    char *name = bp->name ;
    union domain *dom = bp->domain ;
    int data = GEN_HEADER ;

    gen_make(bp, 1, dom->li.constructor, gen_update_arg(dom, 2), "") ;
    (void) printf("fun %s_%s (vector li) = ", name, dom->li.constructor ) ;
    (void) printf("(sub (li,%d));\n", data) ;
}

/* GEN_SET defines the manipulation functions for a set type BP. */

void
gen_set( bp )
struct gen_binding *bp ;
{
    fprintf( stderr, "Set: too be implemented\n" ) ;
}

/* GEN_ARRAY defines the manipulation functions for an array type BP. */

void
gen_array( bp )
     struct gen_binding *bp ;
{
    extern int printf();
    char *name = bp->name ;
    union domain *dom = bp->domain ;
    int data = GEN_HEADER ;

    gen_make(bp, 1, dom->ar.constructor, gen_update_arg(dom, 2), "");
    (void) printf("fun %s_%s (vector ar) = ", name, dom->ar.constructor ) ;
    (void) printf("(sub (ar,%d));\n", data) ;
}

/* GEN_EXTERNAL defines the acces functions for an external type BP. 
   The TYPEDEF has to be added by the user, but should be castable to a
   string (char *). */

void
gen_external( bp )
struct gen_binding *bp ;
{
    fprintf( stderr, "External: too be implemented\n" ) ;
}

/* GEN_DOMAIN generates the manipulation functions for a type BP. This is
   manily a dispatching function. */

void
gen_domain( bp )
struct gen_binding *bp ;
{
    extern int printf();
    union domain *dp = bp->domain ;
    char *s = bp->name ;

    if( !IS_EXTERNAL( bp )) {
	(void) printf( "type %s = chunk;\n", bp->name ) ;
	(void) printf( "val %s_undefined = (undefined:%s);\n", 
		       bp->name, bp->name ) ;
	(void) printf( "fun write_%s fd obj = gen_write fd obj;\n",
	        bp->name ) ;
	(void) printf( "fun read_%s fd = (gen_read fd):%s;\n", 
	        bp->name, bp->name ) ;
    }
    switch( dp->ba.type ) {
    case CONSTRUCTED_DT:
	switch( dp->co.op ) {
	case AND_OP: 
	    gen_and( bp ) ;
	    break ;
	case OR_OP:
	    gen_or( bp ) ;
	    break ;
	default:
	    fatal( "gen_domain: Unknown constructed %s\n", itoa( dp->co.op )) ;
	}
	break ;
    case LIST_DT:
	gen_list( bp ) ;
	break ;
    case SET_DT:
	gen_set( bp ) ;
	break ;
    case ARRAY_DT:
	gen_array( bp ) ;
	break ;
    case EXTERNAL_DT:
	gen_external( bp ) ;
	break ;
    default:
	fatal( "gen_domain: Unknown type %s\n", itoa( dp->ba.type )) ;
    }
}

/* GENCODE generates the code necessary to manipulate every internal and 
   non-inlinable type in the Domains table. */

void
gencode( file )
char *file ;
{
    struct gen_binding *bp = Domains ;
    
    sprintf( start, "gen_%s_start", file ) ;

    for( ; bp < &Domains[ MAX_DOMAIN ] ; bp++ ) {
	if(bp->name == NULL || IS_INLINABLE( bp ) || IS_IMPORT( bp ) ||
	   bp == Tabulated_bp ) 
		continue ;

	gen_domain( bp ) ;
    }
}
	
