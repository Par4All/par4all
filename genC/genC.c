/*
 * $Id$
 *
 * $Log: genC.c,v $
 * Revision 1.31  1998/04/09 15:19:18  coelho
 * RCS headers.
 *
 */
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

/* genC.c

   This file includes the function used to implement user types in C.

   The implementation is based on vectors of gen_chunks. The first one always
   holds, when considered as an integer, the index in the Domains table of the
   type of the object.

   . An inlined value is simply stored inside one gen_chunk,
   . A list is a (CONS *),
   . A sey is a SET,
   . An array is a (CHUNK *),
   . Components values of an AND_OR value are stored in the following gen_chunks.
   . An OR_OP value has 2 more gen_chunks. The second on is the OR_TAG
     (an integer). The third is the component value.
*/

#include <stdio.h>
#include <ctype.h>
#include <string.h>

#include "genC.h"
#include "newgen_include.h"

#define IS_NON_INLINABLE_BASIS(c,f) (strcmp(c,"")==0&&strcmp(f,"p")==0)
#define UPPER(c) ((islower( c )) ? toupper( c ) : c )
#define TYPE(bp) (bp-Domains-Number_imports-Current_start)

#define OR_TAG_OFFSET 2

static char start[ 1024 ] ;


static void 
fprint_upper(
    FILE *file,
    char *s)
{
    for(; *s; s++) fprintf(file, "%c", UPPER(*s));
}


/* GEN_SIZE returns the size (in gen_chunks) of an object of type defined by
   the BP type. */

int
gen_size( bp )
struct gen_binding *bp ;
{
    int overhead = GEN_HEADER+IS_TABULATED( bp ) ;

    switch( bp->domain->ba.type ) {
    case BASIS_DT:
    case ARRAY_DT:
    case LIST_DT:
    case SET_DT:
	return( overhead + 1 ) ;
    case CONSTRUCTED_DT:
	if( bp->domain->co.op == OR_OP ) {
	    return( overhead + 2 ) ;
	}
	else if( bp->domain->co.op == AND_OP ) {
	    int size ;
	    struct domainlist *dlp = bp->domain->co.components ;
      
	    for( size=0 ; dlp != NULL ; dlp=dlp->cdr, size++ ) {
		;
	    }
	    return( overhead + size ) ;
	}
	else if( bp->domain->co.op == ARROW_OP ) {
	    return( overhead+1 ) ;
	}
    default:
	fatal( "gen_size: Unknown type %s\n", itoa( bp->domain->ba.type )) ;
	return(-1); /* to avoid a gcc warning */
	/*NOTREACHED*/
    }
}

/* PRIMITIVE_FIELD returns the appropriate field to acces an object in DP.
   Note how inlined types are managed (see genC.h comments).
   ??? static data returned, cannot be called twice in the same expression
       (FC)
 */

static char *
primitive_field( dp )
union domain *dp ;
{
    static char buffer[ 1024 ];

    switch( dp->ba.type ) {
    case BASIS_DT: {
	struct gen_binding *bp = dp->ba.constructand ;
      
	if( IS_INLINABLE( bp )) {
	    sprintf( buffer, "%c", *bp->name ) ;
	}
	else if( IS_EXTERNAL( bp )) {
	    sprintf( buffer, "e" ) ;
	}
	else {
	    sprintf( buffer, "p" ) ;
	}
	break ;
    }
    case LIST_DT:
	sprintf( buffer, "l" ) ;
	break ;
    case SET_DT:
	sprintf( buffer, "t" ) ;
	break ;
    case ARRAY_DT: 
	sprintf( buffer, "p" ) ;
	break ;
    default:
	fatal( "primitive_field: Unknown type %s\n", itoa( dp->ba.type )) ;
	/*NOTREACHED*/
    }
    return( buffer ) ;
}

static char *
primitive_type(dp)
union domain *dp;
{
    static char buffer[1024];
    
    switch( dp->ba.type )
    {
    case BASIS_DT: 
	sprintf(buffer, dp->ba.constructand->name);
	break;
    case ARRAY_DT: 
    case EXTERNAL_DT:
    case IMPORT_DT:
	sprintf(buffer, "(I don't know what to generate - FC:-)");
	break;
    case LIST_DT:
	sprintf(buffer, "list");
	break;
    case SET_DT:
	sprintf(buffer, "set");
	break;
    default:
	break;
    }
    
    return(buffer);
}

/* PRIMITIVE_CAST returns the appropriate casting to access an object in BP.
   This is mainly for multidimensional arrays and external (thus the 
   NAME). */
static char *
primitive_cast( dp )
union domain *dp ;
{
  static char buffer[ 1024 ];

  switch( dp->ba.type ) {
  case BASIS_DT: 
  {
      struct gen_binding *bp = dp->ba.constructand ;
      
      if (IS_INLINABLE(bp))     /*sprintf( buffer, "")*/ *buffer='\0';
      else if (IS_EXTERNAL(bp)) sprintf( buffer, "(%s)", bp->name ) ;
      else                      /*sprintf( buffer, "")*/ *buffer='\0';

      break ;
  }
  case EXTERNAL_DT:
  case IMPORT_DT:
  case LIST_DT:
  case SET_DT:
	  sprintf(buffer, " ");
	  break ;
  case ARRAY_DT: 
    if( dp->ar.dimensions->cdr != NULL ) {
	    struct intlist *dim = dp->ar.dimensions ;
     
	    sprintf( buffer, "(gen_chunk (*)" ) ;

	    for( dim = dim->cdr ; dim != NULL ; dim = dim->cdr ) {
		    strcat( buffer, "[" ) ;
		    strcat( buffer, itoa( dim->val )) ;
		    strcat( buffer, "]" ) ;
	    }
	    strcat( buffer, ")" ) ;
    }
    else {
	sprintf( buffer, "(gen_chunk *)" ) ;
    }
    break ; 
  default:
    fatal( "primitive_cast: Unknown type %s\n", itoa( dp->ba.type )) ;
    /*NOTREACHED*/
  }
  return( buffer ) ;
}

/* GEN_MEMBER generates a member access functions for domain DP and
   OFFSET. NAME is the domain of the defined domain. */

/* this non casted version is ok for setting a field
 */
static void
gen_member(
    char *name,
    union domain *dp,
    int offset)
{
    char *cast = primitive_cast( dp ) ;
    char *field = primitive_field( dp ) ;

    if (dp->ba.type == BASIS_DT && 
	strcmp( dp->ba.constructand->name, UNIT_TYPE_NAME ) == 0)
	return;
    
    /* non casted version
     */
    printf( "#define %s_%s_(node) ", name, dp->ba.constructor ) ;

    if( IS_NON_INLINABLE_BASIS( cast, field ))
	printf("GEN_CHECK((%s(((node)+%d)->%s)),%s_domain)\n", 
	       cast, offset, field, dp->ba.constructand->name ) ;
    else
	printf("(((node)+%d)->%s)\n", offset, field) ;

    /* casted version 
     */
    printf("#define %s_%s(node) (%s%s_%s_(node))\n", 
	   name, dp->ba.constructor, cast, name, dp->ba.constructor);

    return;
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

    for( buffer[0]='\0'; dlp->cdr != NULL ; dlp = dlp->cdr ) {
	strcat( buffer, gen_arg( dlp->domain )) ;
	strcat( buffer, "," )  ;
    }
    strcat( buffer, gen_arg( dlp->domain )) ;
    return( buffer ) ;
}

/* GEN_MAKE generates the gen_alloc call for gen_bindings BD with SIZE user
   members and ARGS as list of arguments. */

static void
gen_make( bp, size, args )
struct gen_binding *bp ;
int size ;
char *args ;
{
    printf("#define %s_domain (%s+%d)\n",
		  bp->name, start, TYPE( bp )) ;
    printf("#define make_%s(%s) ", bp->name, args ) ;
    printf("(%s)gen_alloc(%ld+%ld*sizeof(gen_chunk),%s,%s_domain%s%s)\n", 
	   bp->name, (long) GEN_HEADER_SIZE, (long) (size+IS_TABULATED(bp)), 
	   "GEN_CHECK_ALLOC", bp->name, (strlen(args)==0)? "": ",", args);
}

/* GEN_AND generates the manipulation functions for an AND type BP. */

void
gen_and( bp )
     struct gen_binding *bp ;
{
    union domain *dom = bp->domain ;
    struct domainlist *dlp ;
    int size = 0 ;

    for( dlp = dom->co.components ; dlp != NULL ; dlp=dlp->cdr, size++ ) ;

    gen_make( bp, size, gen_args( dom->co.components )) ;

    size = GEN_HEADER + IS_TABULATED( bp ) ;

    for( dlp=dom->co.components ; dlp != NULL ; dlp=dlp->cdr )
	gen_member( bp->name, dlp->domain, size++);
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
    int offset ;
  
    gen_make( bp, 2, "tag,val" ) ;
    (void) printf( "#define %s_tag(or) (((or)+%d)->i)\n", 
	    name, 1+IS_TABULATED( bp )) ;

    for( dlp=dom->co.components,offset=dom->co.first ;
	dlp != NULL ; 
	dlp=dlp->cdr, offset++ ) {
	union domain *dp = dlp->domain ;

	(void) printf( "#define is_%s_%s %d\n",
	        name, dp->ba.constructor, offset ) ;
	(void) printf( "#define %s_%s_p(or) ((%s_tag (or))==is_%s_%s)\n",
	        name, dp->ba.constructor, name, name, dp->ba.constructor ) ;
	
	gen_member( name, dp, OR_TAG_OFFSET ) ;
    }
}

/* GEN_ARROW generates the manipulation function for an ARROW_OP type BP. */

void
gen_arrow( bp )
struct gen_binding *bp ;
{
    char *name = bp->name ;
    union domain *dom = bp->domain ;
    union domain *image, *start ;
    int data = GEN_HEADER + IS_TABULATED( bp ) ;

    gen_make( bp, 1, "" ) ;

    start = dom->co.components->domain ;
    image = dom->co.components->cdr->domain ;

    (void) printf("#define apply_%s(hash, var) ", name ) ;
    (void) printf("(%sHASH_GET(%s,", 
		  primitive_cast(image), primitive_field(start));
    (void) printf("%s,(hash+%d)->h, (var)))\n",  primitive_field(image), data);

    (void) printf("#define update_%s(hash, var, val) ", name ) ;
    (void) printf("(HASH_UPDATE(%s", primitive_field(start)) ;
    (void) printf(",%s,((hash)+%d)->h,(var),(val)))\n", 
		  primitive_field(image), data ) ;

    (void) printf("#define extend_%s(hash, var, val) ", name ) ;
    (void) printf("(HASH_EXTEND(%s", primitive_field(start));
    (void) printf(",%s,((hash)+%d)->h,(var),(val)))\n",
		  primitive_field(image), data ) ;

    (void) printf("#define delete_%s(hash, var) ", name ) ;
    (void) printf("(HASH_DELETE(%s", primitive_field(start));
    (void) printf(",%s,((hash)+%d)->h,(var)))\n",
		  primitive_field(image), data ) ;

    /* key and value types
     */
    (void) printf("#define %s_key_type %s\n", name, primitive_type(start));
    (void) printf("#define %s_value_type %s\n", name, primitive_type(image));

    /* bound_XX_p
     */
    (void) printf("#define bound_%s_p(hash, var) (HASH_BOUND_P(%s,",
		  name, primitive_field(start));
    (void) printf("%s,(hash+%d)->h, (var)))\n", primitive_field(image), data);
    
    /* XX_MAP
     */
    (void) printf("#define ");
    fprint_upper(stdout, name);
    (void) printf("_MAP(k, v, code, fun) FUNCTION_MAP(%s, ", name);
    (void) printf("%s, ", primitive_field(start));
    (void) printf("%s, k, v, code, fun)\n", primitive_field(image));
}


/* GEN_LIST defines the manipulation functions for a list type BP. */

void
gen_list( bp )
struct gen_binding *bp ;
{
    char *name = bp->name ;
    union domain *dom = bp->domain ;
    int data = GEN_HEADER + IS_TABULATED( bp ) ;

    gen_make( bp, 1, "ar" ) ;
    (void) printf("#define %s_%s(li) (%s(((li)+%d)->%s))\n", 
		  name, dom->li.constructor, primitive_cast(dom), 
		  data, primitive_field(dom));
}

/* GEN_SET defines the manipulation functions for a set type BP. */

void
gen_set( bp )
struct gen_binding *bp ;
{
    char *name = bp->name ;
    union domain *dom = bp->domain ;
    int data = GEN_HEADER + IS_TABULATED( bp ) ;

    gen_make( bp, 1, "ar" ) ;
    (void) printf( "#define %s_%s(se) ", name, dom->se.constructor ) ;
    (void) printf( "(%s(((se)+%d)->%s))\n",
	   primitive_cast( dom ), data, primitive_field( dom )) ;
}

/* GEN_ARRAY defines the manipulation functions for an array type BP. */

void
gen_array( bp )
     struct gen_binding *bp ;
{
    char *name = bp->name ;
    union domain *dom = bp->domain ;
    int data = GEN_HEADER + IS_TABULATED( bp ) ;

    gen_make( bp, 1, "ar" ) ;
    (void) printf( "#define %s_%s(ar) ", name, dom->ar.constructor ) ;
    (void) printf( "(%s(((ar)+%d)->%s))\n",
		  primitive_cast( dom ), data, primitive_field( dom )) ;
}

/* GEN_EXTERNAL defines the acces functions for an external type BP. 
   The TYPEDEF has to be added by the user, but should be castable to a
   string (char *). */

void
gen_external( bp )
struct gen_binding *bp ;
{
    char *s = bp->name ;

    (void) printf( "#ifndef _newgen_%s_defined\n", s ) ;
    (void) printf( "#define _newgen_%s_defined\n", s ) ;

    (void) printf("#define ");
    fprint_upper(stdout, s);
    (void) printf("_NEWGEN_EXTERNAL (%s+%d)\n", start, TYPE( bp )) ;

    /* for externals, a cast macro is added to char* 
     */
    printf("#define newgen_%s(e) ((char*)e)\n", s);

    (void) printf( "#endif /* _newgen_%s_defined */\n", s) ;
}

/* GEN_DOMAIN generates the manipulation functions for a type BP. This is
   manily a dispatching function. */

void
gen_domain( bp )
struct gen_binding *bp ;
{
    union domain *dp = bp->domain ;

    if( !IS_EXTERNAL( bp )) 
    {
	(void) printf("#define ");
	fprint_upper(stdout, bp->name);
	(void) printf( "(x) ((x).p)\n" );

	(void) printf("#define ");
	fprint_upper(stdout, bp->name);
	(void) printf("_TYPE gen_chunkp\n");
	
	(void) printf( "typedef gen_chunk *%s ;\n", bp->name ) ;
	(void) printf("#define %s_undefined ((%s)gen_chunk_undefined)\n", 
		      bp->name, bp->name ) ;
	(void) printf("#define %s_undefined_p(x) ((x)==%s_undefined)\n", 
		      bp->name, bp->name ) ;
	(void) printf("#define copy_%s(x) ((%s)gen_copy_tree((gen_chunk *)x))\n",
		      bp->name, bp->name ) ;
	(void) printf("#define write_%s(fd,obj) %s\n",
		      bp->name, "(gen_write(fd,(gen_chunk *)obj))") ;
	(void) printf("#define read_%s(fd) ((%s)gen_read(fd))\n", 
		      bp->name, bp->name ) ;
	(void) printf("#define free_%s(o) (gen_free((gen_chunk *)o))\n", 
		      bp->name ) ;
	(void) printf("#define check_%s(o) ((void) gen_check((gen_chunk *)o, %s_domain))\n",
		      bp->name, bp->name);
	(void) printf("#define %s_consistent_p(o) (check_%s(o), gen_consistent_p((gen_chunk *)o))\n",
		      bp->name, bp->name);
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
	case ARROW_OP:
	    gen_arrow( bp ) ;
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
gencode(char * file)
{
    struct gen_binding *bp = Domains ;

    sprintf( start, "_gen_%s_start", file ) ;

    for( ; bp < &Domains[ MAX_DOMAIN ] ; bp++ ) {
	if( bp->name == NULL || 
	    IS_INLINABLE( bp ) || IS_IMPORT( bp ) || bp == Tabulated_bp ) {
	    continue ;
	}
	gen_domain( bp ) ;
    }
}
