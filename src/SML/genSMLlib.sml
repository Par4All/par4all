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


/* genClib.c

   The file has all the generic functions to manipulate C objects implemented
   by chunks (see genC.c). */

/*LINTLIBRARY*/

#include <stdio.h>
extern int fprintf();
extern int fscanf();
extern int fclose();
#include <varargs.h>
#include <stdlib.h>
#include <assert.h>
#include "include.h"
#include "genC.h"
#include "x.tab.h"

cons *Gen_cp_[ MAX_NESTED_CONS ] ;

/* GEN_TABULATED maps any bp->index to the tabulation table. TABULATED_BP is
   the fake domain that helps writing tabulation tables. GEN_TABULATED_NAMES
   maps any name <domain_number|name> to its offset in the Gen_tabulated
   of the domain number. */

chunk *Gen_tabulated_[ MAX_TABULATED ] ;
struct binding *Tabulated_bp ;
hash_table Gen_tabulated_names ;

int Read_spec_mode ;
bool Read_spec_performed = FALSE ;

/* The debug flag can be changed by the user to check genClib code. */

int gen_debug = 0 ;
static int gen_debug_indent = 0 ;

/* Default option in GEN_WRITE. */

static int disallow_undefined_tabulated = TRUE ;

/* DOMAIN_INDEX returns the index in the Domain table for object OBJ. */

static int domain_index( obj )
chunk *obj ;
{
    int i ;

    if( obj == NULL ) {
	user( "domain_index: Trying to use a NULL object\n", "" ) ;
	exit( 1 ) ;
    }
    if( obj == chunk_undefined ) {
	user( "domain_index: Trying to use an undefined object\n", "" ) ;
	exit( 1 ) ;
    }
    if( (i=obj->i) < 0 || i >= MAX_DOMAIN ) {
	user( "domain_index: Inconsistant domain number %s\n", itoa( i )) ;
	abort() ;
    }
    return( i ) ;
}

/* FPRINTF_SPACES prints NUMBER spaces on the FD file descriptor. */

static void
fprintf_spaces( fd, number )
FILE *fd ;
int number ;
{
    for( ; number ; number-- )
	    (void) fprintf( fd, " " ) ;
}

#ifdef DBG_READ
/* WRITE_CHUNK prints on the FILE stream a succession of L chunks (beginning
   at OBJ). This is used for debugging purposes. */

void
write_chunk( file, obj, l )
     FILE *file ;
     chunk *obj ;
     int l ;
{
  int i ;

  (void) fprintf( file, "Chunk %x on %d:", obj, l ) ;
  
  for( i=0 ; i<l ; i++ )
    (void) fprintf( file, "%d ", *(obj+i)) ;

  (void) fprintf( file, "\n" ) ;
}
#endif

/* ARRAY_SIZE returns the number of elements in the array whose dimension
   list is DIM. */

static int
array_size( dim )
     struct intlist *dim ;
{
  int sz = 1 ;

  for( ; dim != NULL ; dim = dim->cdr )
    sz *= dim->val ;

  return( sz ) ;
}

/* INIT_ARRAY returns a freshly allocated array initialized according to
   the information in its domain DP. */

static chunk *
init_array( dp )
     union domain *dp ;
{
  int sizarray = array_size( dp->ar.dimensions ) ;
  /*NOSTRICT*/
  chunk *ar = (chunk *)alloc( sizeof( chunk )*sizarray ) ;

  for( ; sizarray ; sizarray-- )
    ar[ sizarray-1 ].p = chunk_undefined ;

  return( ar ) ;
}

/* FIND_FREE_TABULATED finds a free slot for the tabulated domain BP.
   The slot 0 is unused (see write_tabulated_leaf_in) */

int
find_free_tabulated( bp )
struct binding *bp ;
{
    int i ;

    if( Gen_tabulated_[ bp->index ] == chunk_undefined ) {
	fatal( "find_free_tabulated: Uninitialized %s\n", bp->name ) ;
    }
    i = ((bp->alloc == MAX_TABULATED_ELEMENTS-1) ? 1 : bp->alloc)+1 ;

    for( ; ; i = (i == MAX_TABULATED_ELEMENTS-1) ? 1 : i+1 ) {
	if( i == bp->alloc ) {
	    user( "Too many elements in tabulated domain %s\n", bp->name ) ;
	    /* FI: return code not tested in read.y return( -1 ) ; */
	    abort();
	}
	if( (Gen_tabulated_[ bp->index ]+i)->p == chunk_undefined ) {
	    return( bp->alloc = i ) ;
	}
    }
}

/* GEN_ALLOC_COMPONENT updates the chunk CP from the arg list AP according
   to the domain DP. */

/*VARARGS2*/
void
gen_alloc_component( dp, cp, ap )
union domain *dp ;
chunk *cp ;
va_list *ap ;
{
    switch( dp->ba.type ) {
    case ARRAY :
	if( (cp->p = va_arg( *ap, chunk * )) == NULL )
	    cp->p =  init_array( dp ) ;
	break ;
    case LIST:
	cp->l = va_arg( *ap, cons * ) ; 
	break ;
    case SET:
	cp->t = va_arg( *ap, set ) ; 
	break ;
    case BASIS:
	if( IS_INLINABLE( dp->ba.constructand )) {
	    switch( *dp->ba.constructand->name ) {
	    case 'u': cp->u = va_arg( *ap, unit ) ; break ;
	    case 'b': cp->b = va_arg( *ap, bool ) ; break ;
	    case 'c': cp->c = va_arg( *ap, int ) ; break ;
	    case 'i': cp->i = va_arg( *ap, int ) ; break ;
	    case 'f': cp->f = va_arg( *ap, double ) ; break ;
	    case 's': cp->s = va_arg( *ap, string ) ; break ;
	    default:
		fatal( "gen_alloc: unknown inlinable %s\n",
		       dp->ba.constructand->name ) ;
	    }
	}
	else if( IS_EXTERNAL( dp->ba.constructand )) {
	    cp->s = va_arg( *ap, char * ) ;
	}
	else {
	    cp->p = va_arg( *ap, chunk * ) ;

	    if( gen_debug & GEN_DBG_CHECK ) {
		(void) gen_check( cp->p, dp->ba.constructand-Domains ) ;
	    }
	}
	break ;
    default:
	fatal( "gen_alloc_component: unknown type %s\n", itoa( dp->ba.type )) ;
    }
}

/* GEN_ALLOC allocates SIZE bytes to implement an object whose TYPE is
   the index in the Domains table. A fairly sophisticated initialization
   process is run, namely arrays are filled with undefineds. */

/*VARARGS0*/
chunk *
gen_alloc( va_alist )
va_dcl
{
    va_list ap ;
    /* extern char *malloc() ; bb, 92.06.24 */
    union domain *dp ;
    struct binding *bp ;
    struct domainlist *dlp ;
    chunk *cp ;
    int data ;

    va_start( ap ) ;
    /*NOSTRICT*/
    if( !Read_spec_performed ) {
	user( "gen_read_spec not performed prior to allocation\n", "" ) ;
	exit( 1 ) ;
    }
    cp = (chunk *)alloc( va_arg( ap, int )) ;
    bp = &Domains[ cp->i = va_arg( ap, int ) ] ;
    data = 1 + IS_TABULATED( bp );

    switch( (dp = bp->domain)->ba.type ) {
    case LIST: 
	(cp+data)->l = va_arg( ap, cons *) ;
	break ;
    case SET: 
	(cp+data)->t = va_arg( ap, set) ;
	break ;
    case ARRAY: 
	if( ((cp+data)->p = va_arg( ap, chunk *)) == NULL )
	    (cp+data)->p = init_array( dp ) ;
	break ;
    case CONSTRUCTED:
	if( dp->co.op == AND_OP ) {
	    chunk *cpp ;

	    for( dlp=dp->co.components, cpp=cp+data ;
		 dlp != NULL ; 
		 dlp=dlp->cdr, cpp++ ) {
		gen_alloc_component( dlp->domain, cpp, &ap ) ;
	    }
	}
	else if( dp->co.op == OR_OP ) {
	    int which ;

	    (cp+data)->i = va_arg( ap, int ) ;
	    which = (cp+data)->i - dp->co.first ;

	    for( dlp=dp->co.components; dlp!=NULL && which ;dlp=dlp->cdr ){
		which-- ;
	    }
	    if( dlp == NULL ) {
		user( "gen_alloc: unknown tag for type %s\n", bp->name ) ;
	    }
	    gen_alloc_component( dlp->domain, cp+data+1, &ap ) ;
	}
	else fatal( "gen_alloc: Unknown op %s\n", itoa( dp->co.op )) ;
	break ;
    default:
	fatal( "gen_alloc: Unknown type %s\n", itoa( dp->ba.type )) ;
    }
    if( IS_TABULATED( bp )) {
	enter_tabulated_def( bp->index, bp-Domains, (cp+HASH_OFFSET)->s, cp, 0 ) ;
    }
    va_end( ap ) ;

    return( cp ) ;
}

/* The DRIVER structure is used to monitor the general function which
   traverses objects. NULL is called whenver an undefined pointer is found.
   <sort>_IN is called whenever an object of <sort> is entered. If the
   returned value is TRUE, then recursive calls are made and, at the end,
   the <sort>_OUT function is called. */

struct driver {
  void (*null)() ;
  int (*leaf_in)() ;
  void (*leaf_out)() ;
  int (*simple_in)() ;
  void (*simple_out)() ;
  int (*obj_in)() ;
  void (*obj_out)() ;
} ;

/* To be called on any object pointer. */

#define CHECK_NULL(obj,bp,dr) \
  if( (obj) == chunk_undefined ) {(*(dr)->null)(bp) ; return ;}

static void gen_trav_obj() ;

/* GEN_TRAV_LEAF manages an OBJ value of type BP according to the current
   driver DR. A leaf is an object (inlined or not). */

static void
gen_trav_leaf( bp, obj, dr )
struct binding *bp ;
chunk *obj ;
struct driver *dr ;
{
    CHECK_NULL( obj, bp, dr ) ;

    if( gen_debug & GEN_DBG_TRAV_LEAF ) {
	fprintf_spaces( stderr, gen_debug_indent++ ) ;
	(void) fprintf( stderr, "trav_leaf dealing with " ) ;

	if( IS_INLINABLE( bp ))
		(void) fprintf( stderr, "inlined %s\n", bp->name ) ;
	else if( IS_EXTERNAL( bp ))
		(void) fprintf( stderr, "external %s\n", bp->name ) ;
	else if( IS_TABULATED( bp ))
		(void) fprintf( stderr, "tabulated %s\n", bp->name ) ;
	else (void) fprintf( stderr, "constructed %s\n", bp->name ) ;
    }
    if( (*dr->leaf_in)( obj, bp )) {
	if( !IS_INLINABLE( bp ) && !IS_EXTERNAL( bp )) {
	    if( gen_debug & GEN_DBG_CHECK ) {
		(void) gen_check( obj->p, bp-Domains ) ;
	    }
	    gen_trav_obj( obj->p, dr ) ;
	}
	(*dr->leaf_out)( obj, bp ) ;
    }
    if( gen_debug & GEN_DBG_TRAV_LEAF ) 
	    gen_debug_indent-- ;
}

/* GEN_TRAV_SIMPLE traverses a simple OBJ (which is a (CONS *) for a list
   or points to the first element of an array) of type DP according to the
   driver DR. */

static void
gen_trav_simple( dp, obj, dr )
union domain *dp ;
chunk *obj ;
struct driver *dr ;
{
    CHECK_NULL( obj, (struct binding *)NULL, dr ) ;

    if( gen_debug & GEN_DBG_TRAV_SIMPLE ) {
	fprintf_spaces( stderr, gen_debug_indent++ ) ;
	(void) fprintf( stderr, "trav_simple dealing with " ) ;
	print_domain( stderr, dp ) ;
	(void) fprintf( stderr, "\n" ) ;
    }
    if( (*dr->simple_in)( obj, dp )) {
	switch( dp->ba.type ) {
	case BASIS: 
	    gen_trav_leaf( dp->ba.constructand, obj, dr ) ;
	    break ;
	case LIST: {
	    cons *p ;

	    for( p = obj->l ; p != NULL ; p = p->cdr ) {
		gen_trav_leaf( dp->li.element, &p->car, dr ) ;
	    }
	    break ;
	}
	case SET: {
	    SET_MAP( elt, {
		gen_trav_leaf( dp->se.element, (chunk *)&elt, dr );
	    }, obj->t ) ;
	    break ;
	}
	case ARRAY: {
	    int i ;
	    int size = array_size( dp->ar.dimensions ) ;

	    for( i=0 ; i<size ; i++ )
		    gen_trav_leaf( dp->ar.element, obj->p+i, dr ) ;

	    break ;
	}
	default:
	    fatal( "gen_trav_simple: Unknown type %s\n", itoa( dp->ba.type )) ;
	}
	(*dr->simple_out)( obj, dp ) ;
    }
    if( gen_debug & GEN_DBG_TRAV_SIMPLE ) 
	    gen_debug_indent-- ;
}

/* GEN_TRAV_OBJ (the root function) traverses the object OBJ according to
   the driver DR. */

static void
gen_trav_obj( obj, dr )
     chunk *obj ;
     struct driver *dr ;
{
    CHECK_NULL( obj, (struct binding *)NULL, dr ) ;

    if( !Read_spec_performed ) {
	user( "gen_read_spec not performed prior to use\n", "" ) ;
	exit( 1 ) ;
    }
{
    if( (*dr->obj_in)( obj )) {
	struct binding *bp = &Domains[ domain_index( obj ) ] ;
	union domain *dp = bp->domain ;
	int data = 1+IS_TABULATED( bp ) ;

	if( gen_debug & GEN_DBG_TRAV_OBJECT ) {
	    fprintf_spaces( stderr, gen_debug_indent++ ) ;
	    (void) fprintf( stderr, "trav_obj dealing with " ) ;
	    print_domain( stderr, dp ) ;
	    (void) fprintf( stderr, "\n" ) ;
	}
	switch( dp->ba.type ) {
	case LIST: 
	case SET:
	case ARRAY:
	    gen_trav_simple( dp, obj+data, dr ) ;
	    break ;
	case CONSTRUCTED: {
	    if( dp->co.op == AND_OP ) {
		chunk *cp ;
		struct domainlist *dlp = dp->co.components ;

		for( cp = obj+data ; dlp != NULL ; cp++, dlp = dlp->cdr )
		    gen_trav_simple( dlp->domain, cp, dr ) ;
	    }
	    else if( dp->co.op == OR_OP ) {
		struct domainlist *dlp = dp->co.components ;
		int which = (obj+data)->i - dp->co.first ;

		for( ; dlp!=NULL && which ; which--,dlp=dlp->cdr )
		    ;
		if( dlp == NULL )
		    fatal( "gen_trav_obj: Unknown tag %s\n", 
			  itoa( (obj+data)->i )) ;

		gen_trav_simple( dlp->domain, obj+data+1, dr ) ;
	    }
	    else
		fatal( "gen_trav_obj: Unknown op %s\n", 
		      itoa( dp->co.op )) ;
	    break ;
	}
	default:
	    fatal( "gen_trav_obj: Unknown type %s\n", itoa( dp->ba.type )) ;
	}
	(*dr->obj_out)( obj, bp ) ;
    }
}
    if( gen_debug & GEN_DBG_TRAV_OBJECT ) 
	gen_debug_indent-- ;
}

/* Useful functions */

static void
null()
{
}

static int
go()
{
  return( 1 ) ;
}



/* These functions computes an hash table of object pointers (to be used
   to manage sharing when dealing with objects). */

#define MAX_SHARED_OBJECTS 10000

static char first_seen[ MAX_SHARED_OBJECTS ] ;
static char seen_once[ MAX_SHARED_OBJECTS ] ;

#define FIRST_SEEN(s) (s>=first_seen && s<first_seen+MAX_SHARED_OBJECTS)

/* The OBJ_TABLE maps objects to addresses within the arrays FIRST_SEEN
   and SEEN_ONCE. In the first case, if the address is FIRST_SEEN, then
   this is the first occurence of the object; if it has a non-zero
   offset i, then it is the i-th shared object seen so far. This offset
   is used in SHARED_OBJ to decide which number to print and update the
   OBJ_TABLE to associate the object to SEEN_ONCE+i so that latter
   references can be correctly generated. */

static hash_table obj_table = (hash_table)NULL ;

/* The running counter of shared objects number. */

static int shared_number = 0 ;

static int shared_obj_in() ;

/* SHARED_OBJ_IN introduces an object OBJ in the OBJ_TABLE. If it is
   already in the table, don't recurse (at least, if you want to avoid an
   infinite loop) and give it a number. Else recurse. */

static int
shared_obj_in( obj )
chunk *obj ;
{
    char *seen ;

    if( (seen=hash_get( obj_table, (char *)obj )) != HASH_UNDEFINED_VALUE ) {
	if( seen == first_seen ) {
	    hash_del( obj_table, (char *)obj ) ;

	    if( ++shared_number >= MAX_SHARED_OBJECTS ) {
		fatal( "shared_obj_in: Too many shared objects\n", "" ) ;
	    }
	    hash_put( obj_table, (char *)obj, first_seen+shared_number ) ;
	}
	return( 0 ) ;
    }
    hash_put( obj_table, (char *)obj, first_seen ) ;
    return( 1 ) ;
}

static int
shared_simple_in( obj, dp )
chunk *obj ;
union domain *dp ;
{
    switch( dp->ba.type ) {
    case BASIS:
	return( !dp->ba.persistant ) ;
    case LIST: {
	cons *p ;

	if( obj->l == list_undefined ) {
	    return( 0 ) ;
	}
	for( p=obj->l ; p!=NIL ; p=p->cdr ) {
	    if( hash_get( obj_table, (char *)p ) != HASH_UNDEFINED_VALUE ) {
		user( "shared_simpl_in: Sharing of cons\n" ) ;
		abort () ;
	    }
	    else {
		hash_put( obj_table, (char *)p, (char *)p ) ;
	    }
	}
        return( !dp->li.persistant ) ;
    }
    case SET:
	return( !dp->se.persistant ) ;
    case ARRAY:
	return( !dp->ar.persistant ) ;
    }
    fatal( "shared_simple_in: unknown type %s\n", itoa( dp->ba.type )) ;
}

static int
shared_leaf_in( obj, bp )
chunk *obj ;
struct binding *bp ;
{
    return( !IS_TABULATED( bp )) ;
}

/* SHARED_POINTERS creates (in OBJ_TABLE) the association between objects
   and their numbers (!= 0 if sharing). Inlined values are NOT considered
   to be shared (neither list and arrays), just objects (i.e. objects which
   have a user name, a spec in Domains). KEEP says whether the previous
   sharing table is preserved. */

void
shared_pointers( obj, keep )
chunk *obj ;
bool keep ;
{
  struct driver dr ;

  dr.null = dr.leaf_out = dr.simple_out = dr.obj_out = null ;
  dr.obj_in = shared_obj_in ;
  dr.simple_in = shared_simple_in ;
  dr.leaf_in = shared_leaf_in ;

  if( obj_table == (hash_table)NULL ) {
      obj_table  = hash_table_make( hash_pointer, 0 ) ;
  }
  else if( !keep ) {
      hash_table_free( obj_table ) ;
      obj_table  = hash_table_make( hash_pointer, 0 ) ;	
      shared_number = 0 ;
  }
  gen_trav_obj( obj, &dr ) ;
}

/* SHARED_OBJ manages the OBJect modulo sharing (the OBJ_TABLE has to be
   set first, see above). If the object isn't shared, don't do nothing.
   else, if that's the first appearance, call FIRST and go on, else 
   call OTHERS. If the obj_table isn't defined, recurse. */

static int
shared_obj( obj, first, others )
chunk *obj ;
void (*first)() ;
void (*others)() ;
{
    char *shared ;
    int shared_number ;

    if( obj_table == (hash_table)NULL ) {
	return( 0 ) ;
    }
    if( ((shared=hash_get( obj_table, (char *)obj)) == HASH_UNDEFINED_VALUE)
	|| (shared == first_seen )) {
	return( 0 ) ;
    }
    else if( FIRST_SEEN( shared )) {
	(*first)( shared_number = shared-first_seen ) ;
	hash_del( obj_table, (char *)obj ) ;
	hash_put( obj_table, (char *)obj, seen_once+shared_number ) ;
	return( 0 ) ;
    }
    else {
	(*others)( shared - seen_once ) ;
	return( 1 ) ;
    }
}

/* SHARED_GO is like GO except that it doesn't ask for continuation if
   the node OBJ has already been seen. */

static int shared_go( obj )
chunk *obj ;
{
    return( !shared_obj( obj, null, null )) ;
}



/* These functions are used to implement the freeing of objects. A
   tabulated constructor has to stop recursive freeing. */

/* A tabulated domain BP prohibits its OBJ to be recursively freed. */

static int
free_leaf_in( obj, bp )
chunk *obj ;
struct binding *bp ;
{
    return( !IS_TABULATED( bp ) && !shared_obj( obj, null, null )) ;
}

/* FREE_LEAF_OUT manages external types */

static void
free_leaf_out( obj, bp ) 
chunk *obj ;
struct binding *bp ;
{
    if( IS_INLINABLE( bp )) return ;

    if( IS_EXTERNAL( bp )) {
	if( bp->domain->ex.free == NULL ) {
	    user( "gen_free: uninitialized external type %s\n",
		 bp->name ) ;
	    return ;
	}
	(*(bp->domain->ex.free))( obj->s ) ;
    }
}

/* FREE_SIMPLE_IN checks for defined domains and persistancy. */

static int
free_simple_in( obj, dp )
chunk *obj ;
union domain *dp ;
{
    switch( dp->ba.type ) {
    case BASIS:
	return( !dp->ba.persistant ) ;
    case LIST:
	return( !dp->li.persistant && obj->l != list_undefined ) ;
    case SET:
	return( !dp->se.persistant && obj->t != set_undefined ) ;
    case ARRAY:
	return( !dp->ar.persistant && obj->p != array_undefined ) ;
    }
    fatal( "free_simple_in: unknown type %s\n", itoa( dp->ba.type )) ;
}



/* FREE_SIMPLE_OUT frees the spine of the list OBJ or the whole array
   (according to the type DP). The components are (obviously ?) freed by the
   recursive traversal functions (I said it once ... and for all). */

static void
free_simple_out( obj, dp )
     chunk *obj ;
     union domain *dp ;
{
    switch( dp->ba.type ) {
    case LIST:
	gen_free_list( obj->l ) ;
	break ;
    case SET:
	set_free( obj->t ) ;
	break ;
    case ARRAY:
	free( (char *) obj->p ) ;
	break ;
    }
}

/* FREE_OBJ_OUT just frees the object OBJ. */

static chunk freed_chunk ;

/*ARGSUSED*/
static void
free_obj_out( obj, bp )
chunk *obj ;
struct binding *bp ;
{
    if( IS_TABULATED( bp )) {
	static char local[ 1024 ] ;
	    
	(void) sprintf( local, "%d%c%s", 
		 domain_index( obj ), HASH_SEPAR, (obj+HASH_OFFSET)->s ) ;

	if( Gen_tabulated_names == (hash_table)NULL ) {
	    fatal( "free_obj_out: Null tabulated names for %s\n", bp->name ) ;
	}
	if( hash_del( Gen_tabulated_names, local ) == HASH_UNDEFINED_VALUE ) {
	    user( "free_tabulated: clearing unexisting %s\n", local ) ;
	}
	(Gen_tabulated_[ bp->index ]+abs( (obj+1)->i ))->p = chunk_undefined; 
    }
    obj->p = (chunk *)0 ;
    free((void *) obj ) ;
}

/* GEN_LOCAL_FREE frees the object OBJ with or withou KEEPing the sharing. */ 

static void
gen_local_free( obj, keep )
chunk *obj ;
bool keep ;
{
    struct driver dr ;

    dr.null = null ;
    dr.leaf_out = free_leaf_out ;
    dr.leaf_in = free_leaf_in ;
    dr.obj_in = shared_go ;
    dr.simple_in = free_simple_in ;
    dr.simple_out = free_simple_out ;
    dr.obj_out = free_obj_out ;
    if( !keep ) {
	shared_pointers( obj, FALSE ) ;
    }
    gen_trav_obj( obj, &dr ) ;
}

/* GEN_FREE frees the object OBJ. */ 

void
gen_free( obj )
chunk *obj ;
{
    gen_local_free( obj, FALSE ) ;
}

/* GEN_FREE_WITH_SHARING frees the object OBJ. */ 

void
gen_free_with_sharing( obj )
chunk *obj ;
{
    gen_free( obj, TRUE ) ;
}



/* These functions are used to implement the copying of objects. A
   tabulated constructor has to stop recursive duplication. */

static hash_table copy_table;		/* maps an object on its copy */

chunk *copy_hsearch(key)
chunk *key;
{
    chunk *p ;

    if( key == (chunk *)NULL || key == (chunk *)HASH_UNDEFINED_VALUE) {
	return( key ) ;
    }
    if ((p=(chunk *)hash_get( copy_table, (char *)key ))==
	(chunk *)HASH_UNDEFINED_VALUE) {
	fatal( "[copy_hsearch] bad key: %s\n", itoa( (int) key ));
    }
    return(p);
}
    
void copy_hput( t, k, v )
hash_table t ;
char *k, *v ;
{
    if( k != (char *) HASH_UNDEFINED_VALUE && k != (char *) NULL)
	hash_put( t, k, v ) ;
}


/* COPY_OBJ_IN duplicates an object if it has not already been seen
   (this migth happen with shared objects). inlined sub-domains are copied
   by the call to memcpy. remaining sub-domains require further processing
*/

static int copy_obj_in(obj)
chunk *obj ;
{
    int size;
    chunk *new_obj;
    struct binding *bp = &Domains[ domain_index( obj ) ] ;

    if (shared_obj( obj, null, null ))
	    return 0;

    /* memory is allocated to duplicate the object referenced by obj */
    size = gen_size(bp)*sizeof(chunk);
    new_obj = (chunk *)alloc(size);

    /* the object obj is copied into the new one */
    (void) memcpy((char *) new_obj, (char *) obj, size);

    /* hash table copy_table is updated */
    copy_hput(copy_table, (char *)obj, (char *)new_obj);
    
    return 1;
}

/* Just check for defined simple domains. */

static int copy_simple_in( obj, dp )
chunk *obj ;
union domain *dp ;
{
    switch( dp->ba.type ) {
    case BASIS:
	return( 1 ) ;
    case LIST:
	return( obj->l != list_undefined ) ;
    case SET:
	return( obj->t != set_undefined ) ;
    case ARRAY:
	return( obj->p != array_undefined ) ;
    }
    fatal( "copy_simple_in: unknown type %s\n", itoa( dp->ba.type )) ;
}

/* Recursive copying is allowed for sub-domains except in case of
   tabulated ones. tabulated domain BP prohibits its OBJ to be recursively
   copied. */

static int copy_leaf_in(obj,bp)
chunk *obj ;
struct binding *bp ;
{
    return(!IS_TABULATED(bp)) ;
}

/* COPY_LEAF_OUT manages external sub-domains. warning: the test
   IS_EXTERNAL cannot be applied on an inlined sub-domain */

static void copy_leaf_out(obj,bp) 
chunk *obj ;
struct binding *bp ;
{
    if (IS_INLINABLE(bp))
	    return;
		
    if (IS_EXTERNAL(bp)) {
	if (bp->domain->ex.copy == NULL) {
	    user("gen_copy_tree: uninitialized external type %s\n",
		 bp->name) ;
	    return ;
	}
	copy_hput(copy_table, obj->s, (*(bp->domain->ex.copy))(obj->s)) ;
    }
}

/* GEN_COPY_LIST duplicates cons cells. if list elements are inlinable,
   the old cell CARs are copied into the new ones. if not, the new cells
   must contain the objects that copy_table provides for the old objects
   contained in old cells. the second argument is the domain pointer of old
   list */

cons *gen_copy_list(old_l, dp)
cons *old_l;
union domain *dp ;
{
    cons *old_p, *new_p, *new_l, *pc;
    int inlinable;

    inlinable = IS_INLINABLE(dp->li.element);
    new_l = NIL;

    for (old_p = old_l ; old_p != NIL ; old_p = old_p->cdr) {
	pc = (cons *)alloc( sizeof(struct cons) ) ;

	/* the cons cell is updated */
	if (inlinable)
		pc->car = old_p->car;
	else {
	    pc->car.p = copy_hsearch( old_p->car.p ) ;
	}
	pc->cdr = NIL;
	
	/* pc is linked to the new list */
	if (new_l == NIL)
		new_l = pc;
	else
		new_p->cdr = pc;
	new_p = pc;
    }
    return(new_l);
}

/* GEN_COPY_ARRAY duplicates an array. if array elements are inlinable,
   the old array is copied into the new one. if not, the new array must
   contain the objects that copy_table provides for the old objects
   contained in the old array. the second argument is the domain pointer of
   the old array */

chunk *gen_copy_array(old_a, dp)
chunk *old_a;
union domain *dp ;
{
    int i, size, inlinable;
    chunk *new_a;

    size = array_size(dp->ar.dimensions);
    inlinable = IS_INLINABLE(dp->ar.element);
    new_a = (chunk *) alloc( sizeof(chunk)*size ) ;

    if (inlinable)
	    (void) memcpy((char *) new_a, (char *) old_a, size*sizeof(chunk));
    else {
	for (i = 0; i < size; i++) {
	    new_a[i].p = copy_hsearch( old_a[i].p ) ;
	}
    }
    return(new_a);
}	

/* GEN_COPY_SET duplicates a set. */

set
gen_copy_set( old_s, dp )
set old_s;
union domain *dp ;
{
    set new_s = set_make( dp->se.what ) ;

    if( IS_INLINABLE( dp->se.element )) {
	set_assign( new_s, old_s ) ;
    }
    else {
	SET_MAP( elt, {chunk *new = copy_hsearch( (chunk *)elt );

		       set_add_element( new_s, new_s, new ) ;},
		 old_s ) ;
    }
    return( new_s );
}	

/* COPY_SIMPLE_OUT copies the spine of the list OBJ or the whole array
   (according to the type DP). The components are copied by the recursive
   traversal functions */

static void copy_simple_out(obj,dp)
chunk *obj ;
union domain *dp ;
{
    switch (dp->ba.type) {
    case LIST:
	/* spine of the list is duplicated and  hash table copy_table
	   is updated */
	copy_hput(copy_table, (char *) (obj->l), 
		 (char *) gen_copy_list(obj->l, dp));
	break ;
    case SET:
	copy_hput(copy_table, (char *) (obj->t), 
		 (char *) gen_copy_set(obj->t, dp));
	break ;
    case ARRAY:
	/* array  is duplicated and  hash table copy_table is updated */
	copy_hput(copy_table, (char *)obj->p,
		 (char *)gen_copy_array(obj->p, dp));
	break ;
    }
}

/* COPY_OBJ_OUT achieves to update the new object (copy of the old one)
   once all sub-domains have been recursively copied */

static void copy_obj_out(obj,bp)
chunk *obj ;
struct binding *bp ;
{
    union domain *dp = bp->domain ;
    int data = 1+IS_TABULATED( bp ) ;
    chunk *new_obj = copy_hsearch(obj);;

    switch( dp->ba.type ) {
    case LIST: 
    case SET:
    case ARRAY:
	(new_obj+data)->p = copy_hsearch((obj+data)->p);
	break ;
    case CONSTRUCTED:
	if( dp->co.op == AND_OP ) {
	    chunk *cp ;
	    struct domainlist *dlp = dp->co.components ;

	    for( cp = obj+data ; dlp != NULL ; cp++, dlp = dlp->cdr ) {
		if( dlp->domain->ba.type != BASIS ||
		   !IS_INLINABLE( dlp->domain->ba.constructand ) &&
		   !IS_EXTERNAL( dlp->domain->ba.constructand ) &&
		   !IS_TABULATED(dlp->domain->ba.constructand )) {
		    (new_obj+(cp-obj))->p = copy_hsearch(cp->p);
		}
	    }
	}
	else if( dp->co.op == OR_OP ) {
	    struct domainlist *dlp = dp->co.components ;
	    int which = (obj+data)->i - dp->co.first ;

	    for( ; dlp!=NULL && which ; which--,dlp=dlp->cdr )
		    ;
	    if( dlp == NULL )
		    fatal( "[copy_obj_out] Unknown tag %s\n", 
			  itoa( (obj+data)->i )) ;

	    if( dlp->domain->ba.type != BASIS ||
	       !IS_INLINABLE( dlp->domain->ba.constructand ) &&
	       !IS_EXTERNAL( dlp->domain->ba.constructand )) {
		(new_obj+data+1)->p = copy_hsearch((obj+data+1)->p);
	    }
	}
	else
		fatal( "[copy_obj_out] Unknown op %s\n", itoa( dp->co.op )) ;
	break ;
    default:
	fatal( "[copy_obj_out] Unknown type %s\n", itoa( dp->ba.type )) ;
    }
}

/* GEN_COPY_TREE makes a copy of the object OBJ */ 

chunk *gen_copy_tree( obj )
chunk *obj ;
{
    chunk *copy;
    struct driver dr ;

    dr.null = null ;
    dr.leaf_out = copy_leaf_out ;
    dr.leaf_in = copy_leaf_in ;
    dr.obj_in = copy_obj_in ;
    dr.simple_in = copy_simple_in ;
    dr.simple_out = copy_simple_out ;
    dr.obj_out = copy_obj_out;

    /* sharing is computed */
    shared_pointers( obj, FALSE ) ;

    /* the copy_table is initialized */
    if (copy_table == (hash_table) NULL) {
	copy_table = hash_table_make( hash_pointer, 0 ) ;
    }

    /* recursive travel thru data structures begins ... */
    gen_trav_obj(obj,&dr) ;

    /* the result is extracted from the copy_table */
    copy = copy_hsearch( (char *)obj ) ;

    /* the copy_table is cleared */
    hash_table_clear(copy_table);	

    return(copy); 
}


/* FREE_TABULATED_LEAF_IN frees tabulated leaf OBJ of domain BP only once. */

static int
free_tabulated_leaf_in( obj, bp )
chunk *obj ;
struct binding *bp ;
{
    if ( IS_TABULATED( bp )) {
	if ( obj->p == chunk_undefined ) {
	    return( 0 ) ;
	}
	free_obj_out( obj->p, bp ) ;
	obj->p = chunk_undefined ;
	return( 0 ) ;
    }
    return( free_leaf_in( obj, bp )) ;
}

/* GEN_FREE_TABULATED frees all the elements of the tabulated table of
   BINDING. */

int
gen_free_tabulated( domain )
int domain ;
{
    struct binding *bp = &Domains[ domain ] ;
    int index = bp->index ;
    chunk *fake_obj = gen_alloc( HEADER_SIZE+sizeof( chunk ),
			         Tabulated_bp-Domains,
			         Gen_tabulated_[ index ] ) ;
    struct driver dr ;
    int i ;

    Tabulated_bp->domain->ar.element = bp ;
    dr.null = null ;
    dr.leaf_out = free_leaf_out ;
    dr.leaf_in = free_tabulated_leaf_in ;
    dr.obj_in = shared_go ;
    dr.simple_in = go ;
    dr.simple_out = free_simple_out ;
    dr.obj_out = free_obj_out ;
    shared_pointers( fake_obj, FALSE ) ;
#ifdef DBG_HASH
    (void) fprintf( stderr, "Gen_freeing_tabulated\n" ) ;
    hwrite( Gen_tabulated_names ) ;
#endif
    gen_trav_obj( fake_obj, &dr ) ;
#ifdef DBG_HASH
    (void) fprintf( stderr, "After gen_free_tabulated\n" ) ;
    hwrite( Gen_tabulated_names ) ;
#endif

    bp->alloc = 1 ;
    Gen_tabulated_[ bp->index ] = 
	    (chunk *)alloc( MAX_TABULATED_ELEMENTS*sizeof( chunk )) ;
    
    for( i=0 ; i<MAX_TABULATED_ELEMENTS ; i++ ) {
	(Gen_tabulated_[ bp->index ]+i)->p = chunk_undefined ;
    }
    return( domain ) ;
}

/* GEN_CLEAR_TABULATED_ELEMENT only clears the entry for object OBJ in the
   Gen_tabulated_ and Gen_tabulated_names tables. */

void
gen_clear_tabulated_element( obj )
chunk *obj  ;
{
    struct binding *bp = &Domains[ domain_index( obj ) ] ;

    if( IS_TABULATED( bp )) {
	static char local[ 1024 ] ;
	    
	(void) sprintf( local, "%d%c%s", 
		 domain_index( obj ), HASH_SEPAR, (obj+HASH_OFFSET)->s ) ;

	if( Gen_tabulated_names == (hash_table)NULL ) {
	    fatal( "clear_tabulated: Null tabulated names for %s\n", 
		   bp->name ) ;
	}
	if( hash_del( Gen_tabulated_names, local ) == HASH_UNDEFINED_VALUE ) {
	    user( "clear_tabulated: clearing unexisting %s\n", local ) ;
	}
	(Gen_tabulated_[ bp->index ]+abs( (obj+1)->i ))->p = chunk_undefined ;
    }
    else {
	user( "clear_tabulated: not a tabulated element\n" ) ;
    }
}

/* These functions implements the writing of objects. */

/* USER_FILE is used by driver functions (sorry, no closure in C). */

static FILE *user_file ;

/* WRITE_DEFINE_SHARED_NODE defines the node whose number is N. */

void
write_define_shared_node( n )
     int n ;
{
  (void) fprintf( user_file, "[%d ", n ) ;
}

/* WRITE_SHARED_NODE references a shared node N. */

void
write_shared_node( n ) 
     int n ;
{
  (void) fprintf( user_file, "#]shared %d ", n ) ;
}

static void
write_null( bp )
struct binding *bp ;
{
    (void) fprintf( user_file, "#]null\n" ) ;
}

/* WRITE_OBJ_IN writes the OBJect of type BP. We first prints its type
   (its index in the Domains table), its tag (for OR_OP types) and then
   ... let's do the recursion. */

static int
write_obj_in( obj ) 
     chunk *obj ;
{
    struct binding *bp = &Domains[ domain_index( obj ) ] ;
    union domain *dp = bp->domain ;

    if( shared_obj( obj, write_define_shared_node, write_shared_node ))
	    return( 0 ) ;

    (void) fprintf( user_file, "#(#]type %d ", bp-Domains ) ;

    if( IS_TABULATED( bp )) {
	(void) fprintf( user_file, "%d ", abs( (obj+1)->i )) ;
    }
    switch( dp->ba.type ) {
    case EXTERNAL:
	fatal( "write_obj_in: Don't know how to write an EXTERNAL: %s\n", 
	      bp->name ) ;
	break ;
    case CONSTRUCTED:
	if( dp->co.op == OR_OP )
		(void) fprintf( user_file, "%d ", (obj+1+IS_TABULATED( bp ))->i ) ;
	break ;
    }
    return( 1 ) ;
}

/* WRITE_OBJ_OUT is done when the OBJect (of type BP) has been printed. Just
   close the opening parenthese. */

/*ARGSUSED*/
static void
write_obj_out( obj, bp )
     chunk *obj ;
     struct binding *bp ;
{
  (void) fprintf( user_file, ")\n" ) ;
}
  
static void
write_string( init, s, end )
string init, s, end ;
{
    assert(s!=NULL);
    for( (void) fprintf( user_file, init ) ; *s != '\0' ; s++ ) {
	(void) fprintf( user_file, (*s=='"' || *s=='\\') ? "\\%c" : "%c", *s ) ;
    }
    (void) fprintf( user_file, end ) ;
}

/* WRITE_LEAF_IN prints the OBJect of type BP. If it is inlined, prints it
   according to the format, else recurse. */

static int
write_leaf_in( obj, bp )
chunk *obj ;
struct binding *bp ;
{
    if( IS_TABULATED( bp )) {
	if( obj->p == chunk_undefined ) {
	    if( disallow_undefined_tabulated ) {
		user("gen_write: writing undefined tabulated object\n",
		     NULL) ;
	    }
	    else {
		(void) fprintf( user_file, "#]null " ) ;
	    }
	}
	else {
	    (void) fprintf( user_file ,"#]ref %d \"%d%c", 
		     bp->index, bp-Domains, HASH_SEPAR ) ;
	    write_string( "", (obj->p+HASH_OFFSET)->s, "\" " ) ;
	}
	return( 0 ) ;
    }
    else if( IS_INLINABLE( bp )) {
	char *format = bp->inlined->C_format ;

	if( strcmp( bp->name, UNIT_TYPE ) == 0 ) 
		(void) fprintf( user_file, format ) ;
	else if( strcmp( bp->name, "bool" ) == 0 )
		(void) fprintf( user_file, format, obj->b ) ;
	else if( strcmp( bp->name, "int" ) == 0 ) 
		(void) fprintf( user_file, format, obj->i ) ;
	else if( strcmp( bp->name, "float" ) == 0 )
		(void) fprintf( user_file, format, obj->f ) ;
	else if( strcmp( bp->name, "string" ) == 0 )
		write_string( "\"", obj->s, "\"" ) ;
	else fatal( "write_leaf_in: Don't know how to print %s\n", bp->name ) ;
	(void) fprintf( user_file, " " ) ;
    }
    else if( IS_EXTERNAL( bp )) {
	if( bp->domain->ex.write == NULL ) {
	    user( "gen_write: uninitialized external type %s\n",
		 bp->name ) ;
	    return( 0 ) ;
	}
	(void) fprintf( user_file, "#]external %d ", bp-Domains ) ;
	(*(bp->domain->ex.write))( user_file, obj->s ) ;
    }
    return( 1 ) ;
}

/* WRITE_SIMPLE_IN is done before printing a simple OBJect of type DP. The
   sharing of basis objects will be done later. */

static int
write_simple_in( obj, dp )
chunk *obj ;
union domain *dp ;
{
    switch( dp->ba.type ) {
    case LIST:
	if( obj->l == list_undefined ) {
	    (void) fprintf( user_file, "#]list " ) ;
	    return( 0 ) ;
	}
	(void) fprintf( user_file, "(" ) ;
	break ;
    case SET:
	if( obj->t == set_undefined ) {
	    (void) fprintf( user_file, "#]set " ) ;
	    return( 0 ) ;
	}
	(void) fprintf( user_file, "{ %d ", dp->se.what ) ;
	break ;
    case ARRAY:
	if( obj->p == array_undefined ) {
	    (void) fprintf( user_file, "#]array " ) ;
	    return( 0 ) ;
	}
	(void) fprintf( user_file, "#(" ) ;
	break ;
    }
    return( 1 ) ;
}

/* WRITE_LEAF_OUT prints the closing parenthesis of (non-basis) simple OBJect
   of type DP. */

/*ARGSUSED*/
static void
write_simple_out( obj, dp )
chunk *obj ;
union domain *dp ;
{
    switch( dp->ba.type ) {
    case SET:
	(void) fprintf( user_file, "}" ) ;
	break ;
    case LIST:
    case ARRAY:
	(void) fprintf( user_file, ")" ) ;
	break ;
    }
}

/* GEN_WRITE writes the OBJect on the stream FD. Sharing is managed (the 
   number of which is printed before the object.) */

void
gen_write( fd, obj )
FILE *fd ;
chunk *obj ;
{
    struct driver dr ;

    dr.null = write_null ;
    dr.leaf_out = null ;
    dr.leaf_in = write_leaf_in ;
    dr.simple_in = write_simple_in ;
    dr.simple_out = write_simple_out ;
    dr.obj_in = write_obj_in ;
    dr.obj_out = write_obj_out ;
    user_file = fd ;
    shared_pointers( obj, FALSE ) ;
    (void) fprintf( fd, "%d ", shared_number ) ;
    gen_trav_obj( obj, &dr ) ;
}

/* GEN_WRITE_WITHOUT_SHARING writes the OBJect on the stream FD. Sharing
   is NOT managed.*/

void
gen_write_without_sharing( fd, obj )
FILE *fd ;
chunk *obj ;
{
    struct driver dr ;

    dr.null = write_null ;
    dr.leaf_out = null ;
    dr.leaf_in = write_leaf_in ;
    dr.simple_in = write_simple_in ;
    dr.simple_out = write_simple_out ;
    dr.obj_in = write_obj_in ;
    dr.obj_out = write_obj_out ;
    user_file = fd ;
    if( obj_table != (hash_table)NULL ) {
	hash_table_free( obj_table ) ;
    }
    obj_table = (hash_table)NULL ;
    (void) fprintf( fd, "0 " ) ;
    gen_trav_obj( obj, &dr ) ;
}

/* WRITE_TABULATED_LEAF_IN prints the OBJect of type BP. If it is tabulated,
   then recurse. */

static int
write_tabulated_leaf_in( obj, bp )
chunk *obj ;
struct binding *bp ;
{
    if( IS_TABULATED( bp )) {
	int number ;

	if( obj->p == chunk_undefined ) {
    	    write_null( bp ) ;
	    return( 0 ) ;
	}
	if( (number = (obj->p+1)->i) == 0 ) {
	    fatal( "write_tabulated_leaf_in: Zero index in domain %s\n", 
		   bp->name ) ;
	}
	if( number >= 0 ) {
	    (void) fprintf( user_file ,"#]def %d \"%d%c", 
		     bp->index, bp-Domains, HASH_SEPAR ) ;
	    write_string( "", (obj->p+HASH_OFFSET)->s, "\" " ) ;
	    (obj->p+1)->i = - (obj->p+1)->i ;
	    return( 1 ) ;
	}
    }
    return( write_leaf_in( obj, bp )) ;
}

/* GEN_WRITE_TABULATED writes the tabulated object TABLE on FD. Sharing is 
   managed */

int
gen_write_tabulated( fd, domain )
FILE *fd ;
int domain ;
{
    int index =  Domains[ domain ].index ;
    chunk *fake_obj = gen_alloc( HEADER_SIZE+sizeof( chunk ),
			         Tabulated_bp-Domains,
			         Gen_tabulated_[ index ] ) ;
    struct driver dr ;

    Tabulated_bp->domain->ar.element = &Domains[ domain ] ;
    dr.null = write_null ;
    dr.leaf_out = null ;
    dr.leaf_in = write_tabulated_leaf_in ;
    dr.simple_in = write_simple_in ;
    dr.simple_out = write_simple_out ;
    dr.obj_in = write_obj_in ;
    dr.obj_out = write_obj_out ;
    user_file = fd ;
    shared_pointers( fake_obj, FALSE ) ;
    (void) fprintf( fd, "%d %d ", domain, shared_number ) ;
    gen_trav_obj( fake_obj, &dr ) ;
    free((char *) fake_obj ) ;
    return( domain ) ;
}

#ifdef BSD
static char *strdup( s )
char *s ;
{
    char *new = alloc( strlen( s )+1 ) ;

    strcpy( new, s ) ;
    return( new ) ;
}
#endif

/* GEN_READ_SPEC reads the specifications. This has to be used
   -- before -- any utilization of manipulation functions. */

/*VARARGS0*/
void
gen_read_spec( va_alist )
va_dcl
{
    va_list ap ;
    extern FILE *zzin ;
    char *spec ;
    chunk **cpp ;
    struct binding *bp ;
    char *mktemp(), *tmp ;

    va_start( ap ) ;
    init() ;
    Read_spec_mode = 1 ;
    tmp = mktemp(strdup("/tmp/newgen.XXXXXX")) ;

    while( spec = va_arg( ap, char * )) {
	if( (zzin = fopen( tmp, "w" )) == NULL ) {
	    user( "Cannot open temp spec file in write mode\n" ) ;
	    return ;
	}
	fprintf( zzin, "%s", spec ) ;
	fclose( zzin ) ;

	if( (zzin = fopen( tmp, "r" )) == NULL ) {
	    user( "Cannot open temp spec file in read mode\n" ) ;
	    return ;
	}
	zzparse() ;
	fclose( zzin ) ;
    }
    if( unlink( tmp )) {
	fatal( "Cannot unlink tmp file %s\n", tmp ) ;
    }
    compile() ;

    for( cpp= &Gen_tabulated_[0] ; 
	 cpp<&Gen_tabulated_[MAX_TABULATED] ; 
	 cpp++ ) {
	*cpp = chunk_undefined ;
    }
    for( bp = Domains ; bp < &Domains[ MAX_DOMAIN ] ; bp++ ) {
	if( bp->name != NULL &&
	   !IS_INLINABLE( bp ) && !IS_EXTERNAL( bp ) &&
	   bp->domain->ba.type == IMPORT ) {
	    user( "Cannot run with imported domains: %s\n", bp->name ) ;
	    return ;
	}
	if( IS_TABULATED( bp )) {
	    int i ;

	    bp->alloc = 1 ;
	    Gen_tabulated_[ bp->index ] = 
		    (chunk *)alloc( MAX_TABULATED_ELEMENTS*sizeof( chunk )) ;
	    
	    for( i=0 ; i<MAX_TABULATED_ELEMENTS ; i++ ) {
		(Gen_tabulated_[ bp->index ]+i)->p = chunk_undefined ;
	    }
	    if( Gen_tabulated_names == NULL ) {
		Gen_tabulated_names = 
			hash_table_make( hash_string,
					 MAX_TABULATED*MAX_TABULATED_ELEMENTS ) ;
	    }

	}
    }
    gen_cp_ = &Gen_cp_[ 0 ] ;
    Read_spec_mode = 0 ;
    Read_spec_performed = TRUE ;
    va_end( ap ) ;
}

/* GEN_INIT_EXTERNAL defines entry points for free, read and write functions 
   of external types */

void
gen_init_external( which, read, write, free, copy )
int which ;
char *(*read)() ;
void (*write)() ;
void (*free)() ;
char *(*copy)() ;
{
	struct binding *bp = &Domains[ which ] ;
	union domain *dp = bp->domain ;

	if( dp->ba.type != EXTERNAL ) {
		user( "gen_init_external: %s isn't external\n", bp->name ) ;
		return ;
	}
	if( dp->ex.read != NULL ) {
		user( "gen_init_external: redefinition of %s skipped\n",
		      bp->name ) ;
		return ;
	}
	dp->ex.read = read ;
	dp->ex.write = write ;
	dp->ex.free = free ;
	dp->ex.copy = copy ;
}

/* GEN_MAKE_ARRAY allocates an initialized array of NUM chunks. */

chunk *
gen_make_array( num )
     int num ;
{
  int i ;
  /*NOSTRICT*/
  chunk *ar = (chunk *)alloc( sizeof( chunk )) ;

  for( i=0 ; i<num ; i++ ) 
    ar[ i ].p = chunk_undefined ;

  return( ar ) ;
}

/* GEN_READ reads any object from the FILE stream. Sharing is restored. */

chunk *
gen_read( file )
     FILE *file ;
{
  extern FILE *xxin ;
  
  xxin = file ;
  xxparse() ;
  return( Read_chunk ) ;
}

/* GEN_READ_TABULATED reads FILE to update the Gen_tabulated_ table. Creates
   if CREATE_P is true. */

int
gen_read_tabulated( file, create_p )
FILE *file ;
int create_p ;
{
    extern FILE *xxin ;
    chunk *cp ;
    int domain, index ;
    int i ;
    extern int allow_forward_ref ;

    xxin = file ;
#ifdef flex_scanner
    if( (i=xxlex()) != READ_INT ) {
	char buffer[ 1024 ] ;

	(void) sprintf( buffer, "%d", i ) ;
	user( "Incorrect data for gen_read_tabulated: %s\n", buffer ) ;
	exit( 1 ) ;
    }
    domain = atoi( xxtext ) ;
#else
    (void) fscanf( file, "%d", &domain ) ;
#endif

    if( create_p ) {
	if( Gen_tabulated_[ index = Domains[ domain ].index ] == NULL ) {
	    user( "gen_read_tabulated: Trying to read untabulated domain %s\n",
		  Domains[ domain ].name ) ;
	}
	Domains[ domain ].alloc = 1 ;

	for( i = 0 ; i < MAX_TABULATED_ELEMENTS ; i++ ) {
	    (Gen_tabulated_[ index ]+i)->p = chunk_undefined ;
	}
    }
    allow_forward_ref = TRUE ;
    xxparse() ;
    allow_forward_ref = FALSE ;

    free((char *) ((Read_chunk+1)->p) ) ;
    free((char *) Read_chunk ) ;
    return( domain ) ;
}

int
gen_read_and_check_tabulated( file, create_p )
FILE *file ;
int create_p ;
{
    int i ;
    extern hash_table Gen_tabulated_names ;
    int domain ;

    domain = gen_read_tabulated( file, create_p ) ;

    HASH_MAP( k, v, {
	chunk *hash = (chunk *)v ;

        if( hash->i < 0 ) {
            user( "Tabulated element not defined: %s\n", k ) ;
        }
    }, Gen_tabulated_names ) ;
    return( domain ) ;
}

/* GEN_CHECK checks that the chunk received OBJ is of the appropriate TYPE. */ 

chunk *
gen_check( obj, t )
chunk *obj ;
int t ;
{
    char buffer[ 1024 ] ;

    if( obj != chunk_undefined && t != obj->i ) {
	(void) sprintf( buffer, 
		"gen_check: Type clash (expecting %s, getting %s)\n",
		Domains[ t ].name, Domains[ obj->i ].name ) ;
	user( buffer, (char *)NULL ) ;
    }
    return( obj ) ;
}

int
gen_consistent_p( obj )
chunk *obj ;
{
    static FILE *black_hole = NULL ;
    int old_gen_debug = gen_debug ;
    extern int error_seen ;

    if( black_hole == NULL ) {
	if( (black_hole=fopen( "/dev/null", "r")) == NULL ) {
	    fatal( "Cannot open /dev/null !", "" ) ;
	    /*NOTREACHED*/
	}
    }
    error_seen = 0 ;
    gen_debug = GEN_DBG_CHECK ;
    gen_write( black_hole, obj ) ;
    gen_debug = old_gen_debug ;
    return( error_seen  == 0 ) ;
}

	    
	  
