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


/* $RCSfile: genClib.c,v $ ($Date: 1997/07/22 15:44:58 $, )
 * version $Revision$
 * got on %D%, %T%
 *
 * The file has all the generic functions to manipulate C objects
 * implemented by gen_chunks (see genC.c).
 */

/*LINTLIBRARY*/

#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <stdlib.h>
#include <setjmp.h>

#include "genC.h"
#include "newgen_include.h"
#include "genread.h"

/* lex files
 */
extern FILE *genread_in, *genread_out;
extern FILE *genspec_in, *genspec_out;

#define GO (1)

extern int max_tabulated_elements();
extern gen_chunk *enter_tabulated_def();

cons *Gen_cp_[ MAX_NESTED_CONS ] ;
cons **gen_cp_ = (cons**) NULL ; 

gen_chunk Gen_hash_[ MAX_NESTED_HASH ] ;
gen_chunk *gen_hash_ = (gen_chunk*) NULL ; 

/* GEN_TABULATED maps any bp->index to the tabulation table. TABULATED_BP is
 * the fake domain that helps writing tabulation tables. GEN_TABULATED_NAMES
 * maps any name <domain_number|name> to its offset in the Gen_tabulated
 * of the domain number.
 */

gen_chunk *Gen_tabulated_[ MAX_TABULATED ] ;
struct gen_binding *Tabulated_bp ;

int Read_spec_mode ;
static bool Read_spec_performed = FALSE ;

/* The debug flag can be changed by the user to check genClib code. */
/* If you set gen_debug dynamically with gdb, do not forget to set
 * gen_debug_indent to a positive enough value to avoid problems
 * when gen_trav_obj() moves upwards the point it was when gen_debug
 * was set
 */

int gen_debug = 0 ;
static int gen_debug_indent = 0 ;

/* Default option in GEN_WRITE. */

static int disallow_undefined_tabulated = TRUE ;

/*
 * MACROS added for code checking
 *
 * Fabien COELHO 10/06/94
 */
#define check_domain(dom) \
  message_assert("Consistant domain number",\
		 (dom)>=0 && (dom)<MAX_DOMAIN)

#define check_read_spec_performed() \
  message_assert("gen_read_spec not performed prior to use", \
		 Read_spec_performed);

#define newgen_free(p) (*(char*)(p)='\0', free(p))

/* DOMAIN_INDEX returns the index in the Domain table for object OBJ.
 */
static int domain_index( obj )
gen_chunk *obj ;
{
    message_assert("No NULL object", obj!=NULL);
    message_assert("No undefined object", obj!=gen_chunk_undefined);
    if ((obj->i)<0 || (obj->i)>=MAX_DOMAIN) 
	fatal("Inconsistent domain number %d (%p) found\n", 
	      obj->i, obj);
    return obj->i;
}

/* inlined version of domain_index. what is done by optimizing compilers?
 */
#define quick_domain_index(obj) \
  (((! (obj)) || ((obj)==gen_chunk_undefined) || \
    ((obj)->i<0) || ((obj)->i>MAX_DOMAIN)) ? \
   domain_index(obj) : (obj)->i) /* prints the error message or returns */


/***************************************************** GEN_TABULATED_NAMES */

static hash_table Gen_tabulated_names = 0;

void gen_init_Gen_tabulated_names(void)
{
    message_assert("NULL table", !Gen_tabulated_names)
    Gen_tabulated_names = hash_table_make(hash_string, 1000);
}

void gen_close_Gen_tabulated_names(void)
{
    message_assert("defined table", Gen_tabulated_names);
    hash_table_free(Gen_tabulated_names);
    Gen_tabulated_names = 0;
}

#define check_Gen_tabulated_names() \
  message_assert("Gen_tabulated_names defined", Gen_tabulated_names)

static char * build_unique_tabulated_name(int domain, char * name)
{
    int len = strlen(name);

    /* permanent buffer */
    static int size = 0;
    static char * buffer = 0;
    
    if (len+30>size) {
	size = len+30;
	if (buffer) free(buffer);
	buffer = (char*) malloc(sizeof(char)*size);
	if (!buffer) fatal("build_unique_tabulated_name: memory exhausted\n");
    }

    sprintf(buffer, "%d%c%s", domain, HASH_SEPAR, name);
    return buffer;
}

static char * build_unique_tabulated_name_for_obj(gen_chunk * obj)
{
    char * name = (obj+HASH_OFFSET)->s;
    int domain = quick_domain_index(obj);
    return build_unique_tabulated_name(domain, name);
}	

/* deletes obj from the tabulated names...
 */
void gen_delete_tabulated_name(gen_chunk * obj)
{
    char * key = build_unique_tabulated_name_for_obj(obj);
    char * okey, * val;
    check_Gen_tabulated_names();
    
    if ((val=hash_delget(Gen_tabulated_names, key, &okey)) == 
	HASH_UNDEFINED_VALUE)
	fatal("gen_delete_tabulated_name: clearing unexisting (%s)\n", key);

    free(okey); free(val);
}

char * gen_get_tabulated_name_basic(int domain, char * id)
{
    char * key = build_unique_tabulated_name(domain, id);
    check_Gen_tabulated_names();
    return hash_get(Gen_tabulated_names, key);
}

char * gen_get_tabulated_name(gen_chunk * obj)
{
    char * key = build_unique_tabulated_name_for_obj(obj);
    check_Gen_tabulated_names();
    return hash_get(Gen_tabulated_names, key);
}

char * gen_get_tabulated_name_direct(char * key)
{
    check_Gen_tabulated_names();
    return hash_get(Gen_tabulated_names, key);
}

void gen_put_tabulated_name(int domain, char * name, char * val)
{
    char * key = build_unique_tabulated_name(domain, name);
    check_Gen_tabulated_names();
    hash_put(Gen_tabulated_names, strdup(key), val);
}

void gen_put_tabulated_name_direct(char * key, char * val)
{
    check_Gen_tabulated_names();
    hash_put(Gen_tabulated_names, key, val);
}




/* FPRINTF_SPACES prints NUMBER spaces on the FD file descriptor.`
 */
static void
fprintf_spaces( fd, number )
FILE *fd ;
int number ;
{
    number = number<0 ? 0 : number;
    number = number<40 ? number : 40; /* limited indentation */
    for(; number ; number-- )
	(void) fprintf( fd, " " ) ;
}

#ifdef DBG_READ
/* WRITE_CHUNK prints on the FILE stream a succession of L gen_chunks
 * (beginning at OBJ). This is used for debugging purposes. 
 */
void
write_gen_chunk( file, obj, l )
     FILE *file ;
     gen_chunk *obj ;
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
 * list is DIM. 
 */
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
 * the information in its domain DP.
 */
static gen_chunk *
init_array( dp )
     union domain *dp ;
{
  int sizarray = array_size( dp->ar.dimensions ) ;
  /*NOSTRICT*/
  gen_chunk *ar = (gen_chunk *)alloc( sizeof( gen_chunk )*sizarray ) ;

  for( ; sizarray ; sizarray-- )
    ar[ sizarray-1 ].p = gen_chunk_undefined ;

  return( ar ) ;
}

static int 
array_own_allocated_memory(
    union domain *dp)
{
    return sizeof(gen_chunk)*array_size(dp->ar.dimensions);
}

/* FIND_FREE_TABULATED finds a free slot for the tabulated domain BP.
   The slot 0 is unused (see write_tabulated_leaf_in) */

int
gen_find_free_tabulated( bp )
struct gen_binding *bp ;
{
    int i ;

    if( Gen_tabulated_[ bp->index ] == gen_chunk_undefined ) {
	fatal( "find_free_tabulated: Uninitialized %s\n", bp->name ) ;
    }
    i = ((bp->alloc == max_tabulated_elements()-1) ? 1 : bp->alloc)+1 ;

    for( ; ; i = (i == max_tabulated_elements()-1) ? 1 : i+1 ) {
	if( i == bp->alloc ) {
	    user( "Too many elements in tabulated domain %s\n", bp->name ) ;
	    user("Current limit (%d) can be redefined by setting environment "
		 "variable NEWGEN_MAX_TABULATED_ELEMENTS\n",
		 max_tabulated_elements());
	    exit(1);
	}
	if( (Gen_tabulated_[ bp->index ]+i)->p == gen_chunk_undefined ) {
	    return( bp->alloc = i ) ;
	}
    }
}

/* GEN_ALLOC_COMPONENT updates the gen_chunk CP from the arg list AP according
   to the domain DP. */

/*VARARGS2*/
void
gen_alloc_component( dp, cp, ap, gen_check_p )
union domain *dp ;
gen_chunk *cp ;
va_list *ap ;
int gen_check_p ;
{
    switch( dp->ba.type ) {
    case ARRAY_DT :
	if( (cp->p = va_arg( *ap, gen_chunk * )) == NULL )
	    cp->p =  init_array( dp ) ;
	break ;
    case LIST_DT:
	cp->l = va_arg( *ap, cons * ) ; 
	break ;
    case SET_DT:
	cp->t = va_arg( *ap, set ) ; 
	break ;
    case BASIS_DT:
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
	    cp->p = va_arg( *ap, gen_chunk * ) ;

	    if( gen_debug & GEN_DBG_CHECK || gen_check_p ) {
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

void
gen_alloc_constructed( ap, bp, dp, cp, data, gen_check_p )
va_list ap ;
struct gen_binding *bp ;
union domain *dp ;
gen_chunk *cp ;
int data ;
int gen_check_p ;
{
    struct domainlist *dlp ;

    switch( dp->co.op ) {
    case AND_OP : {
	gen_chunk *cpp ;	

	for( dlp=dp->co.components, cpp=cp+data ;
	    dlp != NULL ; 
	    dlp=dlp->cdr, cpp++ ) {
	    gen_alloc_component( dlp->domain, cpp, &ap, gen_check ) ;
	}
	break ;
    }
    case OR_OP: {
	int which ;

	(cp+data)->i = va_arg( ap, int ) ;
	which = (cp+data)->i - dp->co.first ;

	for( dlp=dp->co.components; dlp!=NULL && which ;dlp=dlp->cdr ){
	    which-- ;
	}
	if( dlp == NULL ) {
	    user( "gen_alloc: unknown tag for type %s\n", bp->name ) ;
	}
	gen_alloc_component( dlp->domain, cp+data+1, &ap, gen_check_p ) ;
	break ;
    }
    case ARROW_OP: {
	(cp+data)->h = hash_table_make( hash_chunk, 0 ) ;
	break ;
    }
    default:
	fatal( "gen_alloc: Unknown op %s\n", itoa( dp->co.op )) ;
    }
}

gen_chunk *
gen_alloc(int size, int gen_check_p, int dom, ...)
{
    va_list ap ;
    union domain *dp ;
    struct gen_binding *bp ;
    gen_chunk *cp ;
    int data ;

    check_read_spec_performed();

    va_start(ap, dom);

    cp = (gen_chunk *)alloc(size) ;
    cp->i = dom;

    bp = &Domains[dom];
    data = 1 + IS_TABULATED( bp );

    switch( (dp = bp->domain)->ba.type ) {
    case LIST_DT: 
	(cp+data)->l = va_arg( ap, cons *) ;
	break ;
    case SET_DT: 
	(cp+data)->t = va_arg( ap, set) ;
	break ;
    case ARRAY_DT: 
	if( ((cp+data)->p = va_arg( ap, gen_chunk *)) == NULL ) {
	    (cp+data)->p = init_array( dp ) ;
	}
	break ;
    case CONSTRUCTED_DT:
	gen_alloc_constructed( ap, bp, dp, cp, data, gen_check_p ) ;
	break ;
    default:
	fatal( "gen_alloc: Unknown type %s\n", itoa( dp->ba.type )) ;
    }

    if (IS_TABULATED(bp)) 
	enter_tabulated_def(bp->index, dom, (cp+HASH_OFFSET)->s, cp, FALSE);

    va_end( ap ) ;

    return( cp ) ;
}

/* The DRIVER structure is used to monitor the general function which
 * traverses objects. NULL is called whenver an undefined pointer is found.
 * <sort>_IN is called whenever an object of <sort> is entered. If the
 * returned value is TRUE, then recursive calls are made and, at the end,
 * the <sort>_OUT function is called. 
 */

struct driver {
  void (*null)() ;
  int (*leaf_in)() ;
  void (*leaf_out)() ;
  int (*simple_in)() ;
  void (*array_leaf)() ;
  void (*simple_out)() ;
  int (*obj_in)() ;
  void (*obj_out)() ;
} ;

/* To be called on any object pointer.
 */

#define CHECK_NULL(obj,bp,dr) \
  if((obj)==gen_chunk_undefined) {(*(dr)->null)(bp) ; return ;}

static void gen_trav_obj() ;
static bool gen_trav_stop_recursion = FALSE; /* set to TRUE to stop... */

/* GEN_TRAV_LEAF manages an OBJ value of type BP according to the current
   driver DR. A leaf is an object (inlined or not). */

static void
gen_trav_leaf( bp, obj, dr )
struct gen_binding *bp ;
gen_chunk *obj ;
struct driver *dr ;
{
    if (gen_trav_stop_recursion) return;

    CHECK_NULL(obj, bp, dr) ;

    if (gen_debug & GEN_DBG_TRAV_LEAF)
    {
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

    if( (*dr->leaf_in)(obj, bp))
    {
	if (gen_trav_stop_recursion) return;

	if( !IS_INLINABLE( bp ) && !IS_EXTERNAL( bp ))
	{
	    if (gen_debug & GEN_DBG_CHECK)
		(void) gen_check( obj->p, bp-Domains ) ;
	    
	    CHECK_NULL(obj->p, bp, dr) ;
	    gen_trav_obj( obj->p, dr ) ;
	}

	(*dr->leaf_out)(obj, bp) ;
    }

    if (gen_debug & GEN_DBG_TRAV_LEAF) gen_debug_indent-- ;
}

/* GEN_TRAV_SIMPLE traverses a simple OBJ (which is a (CONS *) for a list
   or points to the first element of an array) of type DP according to the
   driver DR. */

static void
gen_trav_simple( dp, obj, dr )
union domain *dp ;
gen_chunk *obj ;
struct driver *dr ;
{
    if (gen_trav_stop_recursion) return;

    CHECK_NULL(obj, (struct gen_binding *)NULL, dr);

    if( gen_debug & GEN_DBG_TRAV_SIMPLE ) 
    {
	fprintf_spaces( stderr, gen_debug_indent++ ) ;
	(void) fprintf( stderr, "trav_simple dealing with " ) ;
	print_domain( stderr, dp ) ;
	(void) fprintf( stderr, "\n" ) ;
    }

    if( (*dr->simple_in)(obj, dp))
    {
	switch( dp->ba.type ) 
	{
	case BASIS_DT: 
	    gen_trav_leaf( dp->ba.constructand, obj, dr ) ;
	    break ;
	case LIST_DT: 
	{
	    cons *p ;

	    for (p = obj->l ; p != NULL ; p = p->cdr)
		gen_trav_leaf( dp->li.element, &p->car, dr ) ;
	    break ;
	}
	case SET_DT:
	    SET_MAP(elt,
		{
		    gen_trav_leaf(dp->se.element, (gen_chunk *)&elt, dr);
		}, 
		    obj->t) ;
	    break ;
	case ARRAY_DT: 
	{
	    int i ;
	    int size = array_size(dp->ar.dimensions) ;

	    for (i=0 ; i<size; i++)
		(*dr->array_leaf)(dp->ar.element, i, obj->p+i, dr);

	    break ;
	}
	default:
	    fatal( "gen_trav_simple: Unknown type %s\n", itoa( dp->ba.type )) ;
	}

	if (gen_trav_stop_recursion) {
	    if (gen_debug & GEN_DBG_TRAV_SIMPLE) gen_debug_indent-- ;
	    return;
	}
	(*dr->simple_out)( obj, dp ) ;
    }

    if (gen_debug & GEN_DBG_TRAV_SIMPLE) gen_debug_indent-- ;
}

/* GEN_ARRAY_LEAF is the default recursive call to gen_trav_leaf.
 */

static void
gen_array_leaf(bp, i, obj, dr)
struct gen_binding *bp ;
int i ;
gen_chunk *obj ;
struct driver *dr ;
{
    if (gen_trav_stop_recursion) return;

    gen_trav_leaf( bp, obj, dr ) ;
}


/* GEN_TRAV_OBJ (the root function) traverses the object OBJ according to
   the driver DR. */

static void 
gen_trav_obj_constructed(obj, bp, dp, data, dr)
gen_chunk *obj ;
struct driver *dr ;
struct gen_binding *bp ;
union domain *dp ;
int data ;
{
    struct domainlist *dlp;

    if (gen_trav_stop_recursion) return;

    dlp = dp->co.components ;

    switch(dp->co.op)
    {
    case AND_OP: 
    {
	gen_chunk *cp ;

	for(cp = obj+data ; dlp != NULL ; cp++, dlp = dlp->cdr)
	{
	    gen_trav_simple(dlp->domain, cp, dr) ;
	    if (gen_trav_stop_recursion) return;
	}
	break ;
    }
    case OR_OP: 
    {
	int which = (obj+data)->i - dp->co.first ;

	for(; dlp!=NULL && which; which--,dlp=dlp->cdr)
	    ;

	if(dlp == NULL)
	    fatal( "gen_trav_obj: Unknown tag %s\n", itoa( (obj+data)->i )) ;
	
	gen_trav_simple(dlp->domain, obj+data+1, dr);
	break ;
    }
    case ARROW_OP: 
    {
	union domain 
	    *dkeyp=dlp->domain, 
	    *dvalp=dlp->cdr->domain;

	HASH_MAP(k, v, 
	     {
		 gen_trav_simple(dkeyp, (gen_chunk *) k, dr) ;
		 if (gen_trav_stop_recursion) return;
		 gen_trav_simple(dvalp, (gen_chunk *) v, dr) ;
		 if (gen_trav_stop_recursion) return;		 
	     }, 
		 (obj+data)->h ) ;
	break ;
    }
    default:
	fatal( "gen_trav_obj: Unknown op %s\n", itoa(dp->co.op)) ;
    }
}

static void
gen_trav_obj( obj, dr )
     gen_chunk *obj ;
     struct driver *dr ;
{
    if (gen_trav_stop_recursion) return;

    CHECK_NULL(obj, (struct gen_binding *)NULL, dr);

    if ((*dr->obj_in)(obj, dr))
    {
	struct gen_binding *bp = &Domains[quick_domain_index(obj)] ;
	union domain *dp = bp->domain ;
	int data = 1+IS_TABULATED( bp ) ;

	if (gen_trav_stop_recursion) return;

	if( gen_debug & GEN_DBG_TRAV_OBJECT )
	{
	    fprintf_spaces( stderr, gen_debug_indent++ ) ;
	    (void) fprintf( stderr, "trav_obj (%p) ", obj) ;
	    print_domain( stderr, dp ) ;
	    (void) fprintf( stderr, "\n" ) ;
	}

	switch( dp->ba.type ) 
	{
	case LIST_DT: 
	case SET_DT:
	case ARRAY_DT:
	    gen_trav_simple(dp, obj+data, dr);
	    break ;
	case CONSTRUCTED_DT: 
	    gen_trav_obj_constructed(obj, bp, dp, data, dr);
	    break ;
	default:
	    fatal( "gen_trav_obj: Unknown type %s\n", itoa(dp->ba.type));
	}

	if (gen_trav_stop_recursion) return;

	(*dr->obj_out)(obj, bp, dr);
    }
    if( gen_debug & GEN_DBG_TRAV_OBJECT ) gen_debug_indent--;
}

static int
tabulated_leaf_in( obj, bp )
gen_chunk *obj ;
struct gen_binding *bp ;
{
    return(!IS_TABULATED(bp));
}

/* These functions computes an hash table of object pointers
 * (to be used to manage sharing when dealing with objects). 
 */

#define MAX_SHARED_OBJECTS 10000

static char *first_seen = (char *)NULL ;
static char *seen_once = (char *)NULL ;

#define FIRST_SEEN(s) ((s)>=first_seen && (s)<first_seen+MAX_SHARED_OBJECTS)

/* The OBJ_TABLE maps objects to addresses within the arrays FIRST_SEEN
 * and SEEN_ONCE. In the first case, if the address is FIRST_SEEN, then
 * this is the first occurence of the object; if it has a non-zero
 * offset i, then it is the i-th shared object seen so far. This offset
 * is used in SHARED_OBJ to decide which number to print and update the
 * OBJ_TABLE to associate the object to SEEN_ONCE+i so that latter
 * references can be correctly generated. 
 */

static hash_table obj_table = (hash_table)NULL ;

/* returns the number of byte allocated for obj_table.
 * for FI and debugging purposes...
 * FC
 */
int current_shared_obj_table_size()
{
    return hash_table_own_allocated_memory(obj_table);
}

/* The running counter of shared objects number.
 */

static int shared_number = 0 ;

/* GEN_TRAV_ENVS are stacked to allow recursive calls to GEN_TRAV_OBJ 
 * (cf. GEN_RECURSE)
 */

#define MAX_GEN_TRAV_ENV 100

static int gen_trav_env_top = 0 ;

struct gen_trav_env {
    char *first_seen ;
    char *seen_once ;
    hash_table obj_table ;
    int shared_number ;
} gen_trav_envs[ MAX_GEN_TRAV_ENV ] ;

static void push_gen_trav_env() 
{
    struct gen_trav_env *env ;

    message_assert("Too many recursive gen_trav",
		   gen_trav_env_top < MAX_GEN_TRAV_ENV);

    env = &gen_trav_envs[gen_trav_env_top++] ;
    env->first_seen = first_seen ;
    env->seen_once = seen_once ;
    env->obj_table = obj_table ;
    env->shared_number = shared_number ;

    first_seen = (char *)alloc( MAX_SHARED_OBJECTS ) ;
    seen_once = (char *)alloc( MAX_SHARED_OBJECTS ) ;
    obj_table = hash_table_make( hash_pointer, 0 ) ;
    shared_number = 0 ;
}

static void pop_gen_trav_env() 
{
    struct gen_trav_env *env ;

    message_assert("Too many pops", gen_trav_env_top >= 0);

    newgen_free( first_seen ) ;
    newgen_free( seen_once ) ;
    hash_table_free( obj_table ) ;

    first_seen = (env = &gen_trav_envs[--gen_trav_env_top])->first_seen ;
    seen_once = env->seen_once ;
    obj_table = env->obj_table ;
    shared_number = env->shared_number ;
}

/* SHARED_OBJ_IN introduces an object OBJ in the OBJ_TABLE. If it is
   already in the table, don't recurse (at least, if you want to avoid an
   infinite loop) and give it a number. Else recurse.
 */

static int
shared_obj_in( obj, dr )
gen_chunk *obj ;
struct driver *dr ;
{
    char *seen = hash_get( obj_table, (char *)obj );

    if(seen!=HASH_UNDEFINED_VALUE)
    {
	if(seen == first_seen) 
	{
	    shared_number++;
	    message_assert("shared table not full",
			   shared_number<MAX_SHARED_OBJECTS);
	    
	    hash_update( obj_table, (char *)obj, first_seen+shared_number ) ;
	}
	return(!GO) ;
    }

    hash_put(obj_table, (char *)obj, first_seen ) ;
    return(GO) ;
}

static int
shared_simple_in( obj, dp )
gen_chunk *obj ;
union domain *dp ;
{
    switch( dp->ba.type ) {
    case BASIS_DT:
	return( !dp->ba.persistant ) ;
    case LIST_DT: {
	cons *p ;

	if( obj->l == list_undefined ) {
	    return( !GO) ;
	}
	for( p=obj->l ; p!=NIL ; p=p->cdr ) 
	{
	    message_assert("Sharing of cons",
		  hash_get(obj_table, (char *)p) == HASH_UNDEFINED_VALUE);

	    hash_put(obj_table, (char *)p, (char *)p);
	}
        return( !dp->li.persistant ) ;
    }
    case SET_DT:
	return( !dp->se.persistant ) ;
    case ARRAY_DT:
	return( !dp->ar.persistant ) ;
    }
    fatal( "shared_simple_in: unknown type %s\n", itoa( dp->ba.type )) ;

    return(-1); /* just to avoid a gcc warning */
}

/* SHARED_POINTERS creates (in OBJ_TABLE) the association between objects
   and their numbers (!= 0 if sharing). Inlined values are NOT considered
   to be shared (neither list and arrays), just objects (i.e. objects which
   have a user name, a spec in Domains). KEEP says whether the previous
   sharing table is preserved. */

static void
shared_pointers( obj, keep )
gen_chunk *obj ;
bool keep ;
{
  struct driver dr ;

  dr.null = dr.leaf_out = dr.simple_out = dr.obj_out = gen_null ;
  dr.obj_in = shared_obj_in ;
  dr.simple_in = shared_simple_in ;
  dr.array_leaf = gen_array_leaf ;
  dr.leaf_in = tabulated_leaf_in ;

  message_assert("obj_table not null", obj_table!=(hash_table)NULL);

  if(!keep) {
      hash_table_clear(obj_table) ;
      shared_number = 0 ;
  } 
  /* else the obj_table is kept as it is.
   */

  gen_trav_obj( obj, &dr ) ;
}

/* SHARED_OBJ manages the OBJect modulo sharing (the OBJ_TABLE has to be
   set first, see above). If the object isn't shared, don't do nothing.
   else, if that's the first appearance, call FIRST and go on, else 
   call OTHERS. If the obj_table isn't defined, recurse. */

static int
shared_obj( obj, first, others )
gen_chunk *obj ;
void (*first)() ;
void (*others)() ;
{
    char *shared ;
    int shared_number ;

    message_assert("Defined obj_table", obj_table!=(hash_table)NULL);

    shared = hash_get( obj_table, (char *)obj);

    if(shared==HASH_UNDEFINED_VALUE || shared == first_seen ) 
	return(!GO) ;
    else 
    if( FIRST_SEEN( shared )) 
    {
	(*first)( shared_number = shared-first_seen ) ;
	hash_update( obj_table, (char *)obj, seen_once+shared_number ) ;
	return( !GO) ;
    }
    else 
    {
	(*others)( shared - seen_once ) ;
	return( GO) ;
    }
}



/********************************************************************* FREE */

/* These functions are used to implement the freeing of objects. A
   tabulated constructor has to stop recursive freeing. */

static hash_table free_already_seen = (hash_table) NULL;

static bool 
free_already_seen_p(
    gen_chunk *obj)
{
    message_assert("hash_table defined", free_already_seen);
    
    if (hash_get(free_already_seen, (char *)obj)==(char*)TRUE)
	return TRUE;
    /* else seen for next time !
     */
    hash_put(free_already_seen, (char *)obj, (char *) TRUE);
    return FALSE;
}

/* A tabulated domain BP prohibits its OBJ to be recursively freed. */

static int
free_leaf_in( obj, bp )
gen_chunk *obj ;
struct gen_binding *bp ;
{
    return !IS_TABULATED(bp)/* && !free_already_seen_p(obj) */; /* ??? */
}

/* FREE_LEAF_OUT manages external types. */

static void
free_leaf_out( obj, bp ) 
gen_chunk *obj ;
struct gen_binding *bp ;
{
    if( IS_INLINABLE(bp )) {
	if( *bp->name == 's' && obj->s && !string_undefined_p(obj->s))
		newgen_free(obj->s); 
	return ;
    }
    else
    if( IS_EXTERNAL( bp )) {
	if( bp->domain->ex.free == NULL ) {
	    user( "gen_free: uninitialized external type %s\n",
		 bp->name ) ;
	    return ;
	}
	(*(bp->domain->ex.free))( obj->s ) ;
    }
}

/* FREE_SIMPLE_OUT frees the spine of the list OBJ or the whole array
   (according to the type DP). The components are (obviously ?) freed by the
   recursive traversal functions (I said it once ... and for all). */

static void
free_simple_out( obj, dp )
     gen_chunk *obj ;
     union domain *dp ;
{
    switch( dp->ba.type ) {
    case LIST_DT:
	gen_free_list( obj->l ) ;
	break ;
    case SET_DT:
	set_free( obj->t ) ;
	break ;
    case ARRAY_DT:
	newgen_free( (char *) obj->p ) ;
	break ;
    }
}

/* FREE_OBJ_OUT just frees the object OBJ. */
/* static gen_chunk freed_gen_chunk ; */

static void
free_obj_out( obj, bp, dr )
gen_chunk *obj ;
struct gen_binding *bp ;
struct driver *dr ;
{
    union domain *dp ;

    if((dp=bp->domain)->ba.type == CONSTRUCTED_DT && dp->co.op == ARROW_OP) {
	hash_table h = (obj+1 + IS_TABULATED( bp ))->h ;

	HASH_MAP( k, v, {
	    newgen_free( (void *)k ) ;
	    newgen_free( (void *)v ) ;
	}, h ) ;
	hash_table_free( h ) ;
    }
    obj->p = (gen_chunk *)0 ;
    newgen_free((void *) obj) ;
}

/* GEN_LOCAL_FREE frees the object OBJ with or withou KEEPing the sharing. */ 

static int
persistant_simple_in( obj, dp )
gen_chunk *obj ;
union domain *dp ;
{
    switch( dp->ba.type ) {
    case BASIS_DT:
	return( !dp->ba.persistant && obj->i) ; /* ??? */
    case LIST_DT:
	return( !dp->li.persistant && obj->l && obj->l != list_undefined ) ;
    case SET_DT:
	return( !dp->se.persistant && obj->t && obj->t != set_undefined ) ;
    case ARRAY_DT:
	return( !dp->ar.persistant && obj->p && obj->p != array_undefined ) ;
    }
    fatal( "persistant_simple_in: unknown type %s\n", itoa( dp->ba.type )) ;

    return(-1); /* just to avoid a gcc warning */
}

static int
free_obj_in(
    gen_chunk *obj,
    struct driver *dr)
{
    int notseen = !free_already_seen_p(obj);

    if (notseen) {
	int dom = quick_domain_index(obj);
	struct gen_binding *bp = &Domains[dom] ;

	if( IS_TABULATED( bp )) 
	{
	    gen_delete_tabulated_name(obj);
	    (Gen_tabulated_[bp->index]+abs((obj+1)->i))->p=gen_chunk_undefined;
	}
    }
    
    return notseen;
}

/* version withouy shared_pointers.
 * automatic re-entry allowed. FC.
 */
void
gen_free(
    gen_chunk *obj)
{
    /* reentry or not: whether the already_seen table is initialized or not...
     */
    bool first_in_stack = (free_already_seen==(hash_table)NULL);
    struct driver dr ;
    
    check_read_spec_performed();
    
    dr.null = gen_null ;
    dr.leaf_out = free_leaf_out ;
    dr.leaf_in = free_leaf_in ;
    dr.obj_in = free_obj_in ;
    dr.simple_in = persistant_simple_in ;
    dr.array_leaf = gen_array_leaf ;
    dr.simple_out = free_simple_out ;
    dr.obj_out = free_obj_out ;
    
    if (first_in_stack)
      free_already_seen = hash_table_make(hash_pointer, 0);
    
    gen_trav_obj( obj, &dr ) ;
    
    if (first_in_stack)
    {
	hash_table_free(free_already_seen);
	free_already_seen = NULL;
    }
}

void 
gen_full_free_list(
		   list l)
{
    list p, nextp ;
    bool first_in_stack = (free_already_seen==(hash_table)NULL);
    
    if (first_in_stack)
	free_already_seen = hash_table_make(hash_pointer, 0);
    
    for( p = l ; p != NIL ; p = nextp ) {
	nextp = p->cdr ;
	gen_free( CAR(p).p ) ;
	newgen_free( p ) ;
    }
    
    if (first_in_stack)
    {
	hash_table_free(free_already_seen);
	free_already_seen = NULL;
    }
}

/********************************************************************* COPY */

/* These functions are used to implement the copying of objects. A
   tabulated constructor has to stop recursive duplication. */

static hash_table copy_table = NULL;/* maps an object on its copy */

static gen_chunk *
copy_hsearch(gen_chunk *key)
{
    gen_chunk *p ;

    /* special cases... */
    if(!key || string_undefined_p((char*)key) ||
       key == (gen_chunk *)HASH_UNDEFINED_VALUE)
	return key;

    if ((p=(gen_chunk *)hash_get( copy_table, (char *)key ))==
	(gen_chunk *)HASH_UNDEFINED_VALUE) {
	fatal( "[copy_hsearch] bad key: %p\n", key );
    }
    return(p);
}
    
static void 
copy_hput(
    hash_table t,
    char *k, 
    char *v)
{
    if( k != (char *) HASH_UNDEFINED_VALUE && k != (char *) NULL)
	hash_put(t, k, v) ;
}


/* COPY_OBJ_IN duplicates an object if it has not already been seen
   (this migth happen with shared objects). inlined sub-domains are copied
   by the call to memcpy. remaining sub-domains require further processing
*/

static int 
copy_obj_in(obj, dr)
gen_chunk *obj ;
struct driver *dr ;
{
    struct gen_binding *bp = &Domains[quick_domain_index( obj ) ] ;

    /* if (shared_obj( obj, gen_null, gen_null )) return 0;*/

    if (!hash_defined_p(copy_table, (char*) obj))
    {
	/* memory is allocated to duplicate the object referenced by obj 
	 */
	gen_chunk *new_obj;
	int size = gen_size(bp)*sizeof(gen_chunk);
	new_obj = (gen_chunk *)alloc(size);

	/* thus content is copied, thus no probleme with inlined and so
	 * and newgen domain number.
	 */
	(void) memcpy((char *) new_obj, (char *) obj, size);

	/* hash table copy_table is updated 
	 */
	copy_hput(copy_table, (char *)obj, (char *)new_obj);
	return TRUE;
    }

    return FALSE;
}

/* Just check for defined simple domains. */

static int 
copy_simple_in(
    gen_chunk *obj,
    union domain *dp)
{
    bool persistence = dp->ba.persistant;

    /* persistent arcs are put as copy of themself... 
     */
    if (persistence)
	copy_hput(copy_table, (char *) obj->p, (char *) obj->p);

    switch(dp->ba.type) {
    case BASIS_DT:
	return(!persistence);
    case LIST_DT:
	return(!persistence && obj->l!=list_undefined);
    case SET_DT:
	return(!persistence && obj->t!=set_undefined);
    case ARRAY_DT:
	return(!persistence && obj->p!=array_undefined);
    }
    fatal("copy_simple_in: unknown type %s\n", itoa(dp->ba.type));

    return(-1); /* just to avoid a gcc warning */
}

/* COPY_LEAF_OUT manages external sub-domains. warning: the test
   IS_EXTERNAL cannot be applied on an inlined sub-domain */

static void 
copy_leaf_out(obj,bp) 
gen_chunk *obj ;
struct gen_binding *bp ;
{
    if (IS_INLINABLE(bp)) 
    {
	if (*bp->name=='s' && obj->s && !string_undefined_p(obj->s) &&
	    !hash_defined_p(copy_table, obj->s))
	    copy_hput(copy_table, obj->s, strdup(obj->s));

	return;
    }
		
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

static list 
gen_copy_list(
    list old_l,
    union domain *dp)
{
    list old_p, new_p = NIL, new_l, pc;
    bool inlinable, persistant, tabulated;

    inlinable = IS_INLINABLE(dp->li.element);
    tabulated = IS_TABULATED(dp->li.element);
    persistant = dp->li.persistant;
    new_l = NIL;

    if (inlinable || persistant || tabulated)
	return gen_copy_seq(old_l);

    /* else the items must also be copied
     */
    for (old_p = old_l ; old_p != NIL ; old_p = old_p->cdr) 
    {
	pc = (list)alloc(sizeof(struct cons)) ;

	pc->car.p = copy_hsearch(old_p->car.p) ;
	pc->cdr = NIL;
	
	/* pc is linked to the new list 
	 */
	if (new_l == NIL)
	    new_l = pc;
	else
	    new_p->cdr = pc;
	new_p = pc;
    }

    return new_l;
}

/* GEN_COPY_ARRAY duplicates an array. if array elements are inlinable,
   the old array is copied into the new one. if not, the new array must
   contain the objects that copy_table provides for the old objects
   contained in the old array. the second argument is the domain pointer of
   the old array */

static gen_chunk *
gen_copy_array(old_a, dp)
gen_chunk *old_a;
union domain *dp ;
{
    int i, size, inlinable;
    gen_chunk *new_a;

    size = array_size(dp->ar.dimensions);
    inlinable = IS_INLINABLE(dp->ar.element);
    new_a = (gen_chunk *) alloc( sizeof(gen_chunk)*size ) ;

    if (inlinable) {
	(void) memcpy((char *) new_a, (char *) old_a, size*sizeof(gen_chunk));
    }
    else {
	for (i = 0; i < size; i++) {
	    new_a[i].p = copy_hsearch( old_a[i].p ) ;
	}
    }
    return(new_a);
}	

/* GEN_COPY_SET duplicates a set. */

static set
gen_copy_set( old_s, dp )
set old_s;
union domain *dp ;
{
    set new_s = set_make( dp->se.what ) ;

    if( IS_INLINABLE( dp->se.element )) {
	set_assign( new_s, old_s ) ;
    }
    else {
	SET_MAP( elt, {
	  gen_chunk *new = copy_hsearch( (gen_chunk *)elt );

	  set_add_element( new_s, new_s, (char *)new ) ;
	}, old_s ) ;
    }
    return( new_s );
}	

/* COPY_SIMPLE_OUT copies the spine of the list OBJ or the whole array
   (according to the type DP). The components are copied by the recursive
   traversal functions */

static void 
copy_simple_out(obj,dp)
gen_chunk *obj ;
union domain *dp ;
{
    switch (dp->ba.type) {
    case LIST_DT:
	/* spine of the list is duplicated and  hash table copy_table
	   is updated */
	copy_hput(copy_table, (char *) (obj->l), 
		 (char *) gen_copy_list(obj->l, dp));
	break ;
    case SET_DT:
	copy_hput(copy_table, (char *) (obj->t), 
		 (char *) gen_copy_set(obj->t, dp));
	break ;
    case ARRAY_DT:
	/* array  is duplicated and  hash table copy_table is updated */
	copy_hput(copy_table, (char *)obj->p,
		 (char *)gen_copy_array(obj->p, dp));
	break ;
    }
}

/* COPY_OBJ_OUT achieves to update the new object (copy of the old one)
   once all sub-domains have been recursively copied */

#define COPYABLE_DOMAIN(d) \
( d->ba.type != BASIS_DT || \
 (!(IS_INLINABLE(d->ba.constructand) && (*d->ba.constructand->name!='s')) && \
  !IS_TABULATED( d->ba.constructand )))

static void
copy_obj_out_constructed( obj, bp, dp, data, new_obj, dr ) 
gen_chunk *obj, *new_obj ;
struct gen_binding *bp ;
union domain *dp ;
int data ;
struct driver *dr ;
{
    struct domainlist *dlp = dp->co.components ;

    switch( dp->co.op ) {
    case AND_OP: {
	gen_chunk *cp ;

	for( cp = obj+data ; dlp != NULL ; cp++, dlp = dlp->cdr ) {
	    if(COPYABLE_DOMAIN( dlp->domain)) {
		(new_obj+(cp-obj))->p = copy_hsearch(cp->p);
	    }
	}
	break ;
    }
    case OR_OP: {
	int which = (obj+data)->i - dp->co.first ;

	for( ; dlp!=NULL && which ; which--,dlp=dlp->cdr ) {
	    ;
	}
	if( dlp == NULL ){
	    fatal("[copy_obj_out] Unknown tag %s\n", itoa( (obj+data)->i )) ;
	}
	if( COPYABLE_DOMAIN( dlp->domain )) {
	    (new_obj+data+1)->p = copy_hsearch((obj+data+1)->p);
	}
	break ;
    }
    case ARROW_OP: {
	bool cp_domain = (COPYABLE_DOMAIN( dlp->domain )) ;
	bool cp_codomain = (COPYABLE_DOMAIN( dlp->cdr->domain )) ;
	
	(new_obj+data)->h = hash_table_make(hash_table_type((obj+data)->h), 0);

	HASH_MAP( k, v, {
	    k =  (cp_domain ? (char *)copy_hsearch( (gen_chunk *)k ) : k) ;
	    v =  (cp_codomain ? (char *)copy_hsearch( (gen_chunk *)v ) : v) ;
	    hash_put((new_obj+data)->h, k, v ) ;
	}, (obj+data)->h ) ;
	break ;
    }
    default:
	fatal( "[copy_obj_out] Unknown op %s\n", itoa( dp->co.op )) ;
    }
}

static void 
copy_obj_out(obj,bp,dr)
gen_chunk *obj ;
struct gen_binding *bp ;
struct driver *dr ;
{
    union domain *dp = bp->domain ;
    int data = 1+IS_TABULATED( bp ) ;
    gen_chunk *new_obj = copy_hsearch(obj) ;

    switch( dp->ba.type ) {
    case LIST_DT: 
    case SET_DT:
    case ARRAY_DT:
	(new_obj+data)->p = copy_hsearch((obj+data)->p);
	break ;
    case CONSTRUCTED_DT:
	copy_obj_out_constructed( obj, bp, dp, data, new_obj, dr ) ;
	break ;
    default:
	fatal( "[copy_obj_out] Unknown type %s\n", itoa( dp->ba.type )) ;
    }
}

/* GEN_COPY_TREE makes a copy of the object OBJ */ 

static gen_chunk *
gen_local_copy_tree(
    gen_chunk *obj, 
    bool keep) /* whether to keep the copy tables... */
{
    gen_chunk *copy;
    struct driver dr ;
    hash_table old_copy_table = hash_table_undefined;

    check_read_spec_performed();

    dr.null = gen_null ;
    dr.leaf_out = copy_leaf_out ;
    dr.leaf_in = tabulated_leaf_in ;
    dr.obj_in = copy_obj_in ;
    dr.simple_in = copy_simple_in ;
    dr.array_leaf = gen_array_leaf ;
    dr.simple_out = copy_simple_out ;
    dr.obj_out = copy_obj_out;

    /* Save the old copy_table if required...
     */
    if (!keep)
    {
	old_copy_table = copy_table;
	copy_table = hash_table_make( hash_pointer, 0 ) ;
    }

    gen_trav_obj(obj, &dr) ;
    copy = copy_hsearch(obj) ;

    /* restore copy environment if needed
     */
    if (!keep)
    {	
	hash_table_free(copy_table);
	copy_table = old_copy_table;
    }

    return copy; 
}

gen_chunk *
gen_copy_tree(
    gen_chunk *obj)
{
    return gen_local_copy_tree(obj, FALSE);
}

/* for re-entry only in gen_copy_tree... 
 * ??? because externals are internals... FC.
 */
gen_chunk *
gen_copy_tree_with_sharing(
    gen_chunk *obj)
{
    return gen_local_copy_tree(obj, TRUE);
}


/*********************************************************** FREE_TABULATED */

/* GEN_FREE_TABULATED frees all the elements of the tabulated table of
   BINDING. */

int 
gen_free_tabulated(int domain)
{
    struct gen_binding *bp = &Domains[ domain ];
    gen_chunk * t = Gen_tabulated_[bp->index];
    int i, size = max_tabulated_elements();
    
    check_read_spec_performed();

    /* since gen_free is reentrant and manages sharing globally
     * with the following table, we just call it for each object
     * and everything is fine. Well, I hope so. FC
     */
    message_assert("not initialized", !free_already_seen);
    free_already_seen = hash_table_make(hash_pointer, 0);

    /* fprintf(stderr, " -- domain %d (%d allocated)\n", domain, bp->alloc); */

    for (i=0; i<size; i++)
    {
	if (t[i].p && !gen_chunk_undefined_p(t[i].p)) {
	    /*fprintf(stderr, "freeing [%d] (0x%x) %s \n", i,
		    (unsigned int) t[i].p, ((t[i].p)+2)->s); */

	    gen_free(t[i].p), t[i].p = gen_chunk_undefined;
	}
    }

    hash_table_free(free_already_seen);
    free_already_seen = NULL;

    bp->alloc = 1; /* ??? */
    return domain;
}

/* GEN_CLEAR_TABULATED_ELEMENT only clears the entry for object OBJ in the
   Gen_tabulated_ and Gen_tabulated_names tables. */

void
gen_clear_tabulated_element( obj )
gen_chunk *obj  ;
{
    struct gen_binding *bp = &Domains[ quick_domain_index( obj ) ] ;

    if( IS_TABULATED( bp )) {
	gen_delete_tabulated_name(obj);
	(Gen_tabulated_[ bp->index ]+abs( (obj+1)->i ))->p = 
	    gen_chunk_undefined ;
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
struct gen_binding *bp ;
{
    (void) fprintf( user_file, "#]null\n" ) ;
}

/* WRITE_OBJ_IN writes the OBJect of type BP. We first prints its type
   (its index in the Domains table), its tag (for OR_OP types) and then
   ... let's do the recursion. */

static int
write_obj_in( obj, dr ) 
gen_chunk *obj ;
struct driver *dr ;
{
    struct gen_binding *bp = &Domains[ quick_domain_index( obj ) ] ;
    union domain *dp = bp->domain ;
    int data = 1+IS_TABULATED( bp ) ;

    /* fprintf(stderr, "object 0x%x type %s\n", (unsigned int) obj, bp->name); 
     */

    if( shared_obj( obj, write_define_shared_node, write_shared_node ))
	    return( !GO) ;

    (void) fprintf( user_file, "#(#]type %d ", bp-Domains ) ;

    if( IS_TABULATED( bp )) {
	(void) fprintf( user_file, "%d ", abs( (obj+1)->i )) ;
    }
    switch( dp->ba.type ) {
    case EXTERNAL_DT:
	fatal( "write_obj_in: Don't know how to write an EXTERNAL: %s\n", 
	      bp->name ) ;
	break ;
    case CONSTRUCTED_DT:
	if( dp->co.op == OR_OP ) {
	    (void) fprintf( user_file, "%d ", (obj+data)->i ) ;
	} 
	else if( dp->co.op == ARROW_OP ) {
	    (void) fprintf(user_file, "#]arrow ") ;
	}
	break ;
    }
    return( GO) ;
}

/* WRITE_OBJ_OUT is done when the OBJect (of type BP) has been printed. Just
   close the opening parenthese. */

/*ARGSUSED*/
static void
write_obj_out( obj, bp, dr )
gen_chunk *obj ;
struct gen_binding *bp ;
struct driver *dr ;
{
    union domain *dp = bp->domain ;

    switch( dp->ba.type ) {
    case CONSTRUCTED_DT:
	if( dp->co.op == ARROW_OP ) {
	    (void) fprintf(user_file, ")") ;
	}
	break ;
    }
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
gen_chunk *obj ;
struct gen_binding *bp ;
{
    if( IS_TABULATED( bp )) {
	if( obj->p == gen_chunk_undefined ) {
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
	return( !GO) ;
    }
    else if( IS_INLINABLE( bp )) {
	char *format = bp->inlined->C_format ;

	if( strcmp( bp->name, UNIT_TYPE_NAME ) == 0 ) 
	    (void) fprintf( user_file, format ) ;
	else if( strcmp( bp->name, "bool" ) == 0 )
	    (void) fprintf( user_file, format, obj->b ) ;
	else if( strcmp( bp->name, "int" ) == 0 ) 
	    (void) fprintf( user_file, format, obj->i ) ;
	else if( strcmp( bp->name, "float" ) == 0 )
	    (void) fprintf( user_file, format, obj->f ) ;
	else if( strcmp( bp->name, "string" ) == 0 )
	{
	    /* special management of string_undefined... FC.\
	     */
	    string s = obj->s;
	    write_string( "\"", 
			 string_undefined_p(s) ? disk_string_undefined : s,
			 "\"" ) ;
	}
	else fatal( "write_leaf_in: Don't know how to print %s\n", bp->name ) ;
	(void) fprintf( user_file, " " ) ;
    }
    else if( IS_EXTERNAL( bp )) {
	if( bp->domain->ex.write == NULL ) {
	    user( "gen_write: uninitialized external type %s\n",
		 bp->name ) ;
	    return( !GO) ;
	}
	(void) fprintf( user_file, "#]external %d ", bp-Domains ) ;
	(*(bp->domain->ex.write))( user_file, obj->s ) ;
    }
    return( GO) ;
}

/* WRITE_SIMPLE_IN is done before printing a simple OBJect of type DP. The
   sharing of basis objects will be done later. */

static int
write_simple_in( obj, dp )
gen_chunk *obj ;
union domain *dp ;
{
    switch( dp->ba.type ) {
    case LIST_DT:
	if( obj->l == list_undefined ) {
	    (void) fprintf( user_file, "#]list " ) ;
	    return( !GO) ;
	}
	(void) fprintf( user_file, "(" ) ;
	break ;
    case SET_DT:
	if( obj->t == set_undefined ) {
	    (void) fprintf( user_file, "#]set " ) ;
	    return( !GO) ;
	}
	(void) fprintf( user_file, "{ %d ", dp->se.what ) ;
	break ;
    case ARRAY_DT:
	if( obj->p == array_undefined ) {
	    (void) fprintf( user_file, "#]array " ) ;
	    return( !GO) ;
	}
	(void) fprintf(user_file, "#( %d ", 
		       array_size( dp->ar.dimensions )) ;
	break ;
    }
    return( GO) ;
}

/* WRITE_ARRAY_LEAF only writes non-null elements, in a sparse way. */

static void
write_array_leaf( bp, i, obj, dr )
struct gen_binding *bp ;
int i ;
gen_chunk *obj ;
struct driver *dr ;
{
    if( IS_INLINABLE( bp ) || IS_EXTERNAL( bp )) {
	gen_trav_leaf( bp, obj, dr ) ;
    }
    else if( obj->p != gen_chunk_undefined ) {
	fprintf( user_file, "%d ", i ) ;
	    
	gen_trav_leaf( bp, obj, dr ) ;
    }
}

/* WRITE_LEAF_OUT prints the closing parenthesis of (non-basis) simple OBJect
   of type DP. */

/*ARGSUSED*/
static void
write_simple_out( obj, dp )
gen_chunk *obj ;
union domain *dp ;
{
    switch( dp->ba.type ) {
    case SET_DT:
	(void) fprintf( user_file, "}" ) ;
	break ;
    case LIST_DT:
    case ARRAY_DT:
	(void) fprintf( user_file, ")" ) ;
	break ;
    }
}

/* GEN_WRITE writes the OBJect on the stream FD. Sharing is managed (the 
   number of which is printed before the object.) */

void
gen_write(fd, obj)
FILE *fd ;
gen_chunk *obj ;
{
    struct driver dr ;

    check_read_spec_performed();

    dr.null = write_null ;
    dr.leaf_out = gen_null ;
    dr.leaf_in = write_leaf_in ;
    dr.simple_in = write_simple_in ;
    dr.array_leaf = write_array_leaf ;
    dr.simple_out = write_simple_out ;
    dr.obj_in = write_obj_in ;
    dr.obj_out = write_obj_out ;

    user_file = fd ;

    push_gen_trav_env();

    shared_pointers(obj, FALSE) ;
    (void) fprintf( fd, "%d ", shared_number ) ;
    gen_trav_obj( obj, &dr ) ;

    pop_gen_trav_env() ;

}

/* GEN_WRITE_WITHOUT_SHARING writes the OBJect on the stream FD. Sharing
   is NOT managed.
*/

void
gen_write_without_sharing( fd, obj )
FILE *fd ;
gen_chunk *obj ;
{
    struct driver dr ;

    check_read_spec_performed();

    dr.null = write_null ;
    dr.leaf_out = gen_null ;
    dr.leaf_in = write_leaf_in ;
    dr.simple_in = write_simple_in ;
    dr.array_leaf = gen_array_leaf ;
    dr.simple_out = write_simple_out ;
    dr.obj_in = write_obj_in ;
    dr.obj_out = write_obj_out ;

    user_file = fd ;

    if (obj_table != (hash_table)NULL)
    {
	hash_table_free(obj_table) ;
	obj_table = (hash_table)NULL ;
    }

    (void) fprintf( fd, "0 " ) ;
    gen_trav_obj( obj, &dr ) ;
}

/* WRITE_TABULATED_LEAF_IN prints the OBJect of type BP. If it is tabulated,
   then recurse. */

static int
write_tabulated_leaf_in( obj, bp )
gen_chunk *obj ;
struct gen_binding *bp ;
{
    if( IS_TABULATED( bp )) {
	int number ;

	if( obj->p == gen_chunk_undefined ) {
    	    write_null( bp ) ;
	    return( !GO) ;
	}
	if( (number = (obj->p+1)->i) == 0 ) {
	    fatal( "write_tabulated_leaf_in: Zero index in domain %s\n", 
		   bp->name ) ;
	}

	/* fprintf(stderr, "writing %d %s\n", number, (obj->p+HASH_OFFSET)->s);
	 */
	if( number >= 0 ) {
	    (void) fprintf( user_file ,"#]def %d \"%d%c", 
		     bp->index, bp-Domains, HASH_SEPAR ) ;
	    write_string( "", (obj->p+HASH_OFFSET)->s, "\" " ) ;
	    
	    /* once written the domain number sign is inverted... */ 
	    (obj->p+1)->i = - (obj->p+1)->i ; 
	    return( GO) ;
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
    gen_chunk *fake_obj = gen_alloc(GEN_HEADER_SIZE+sizeof( gen_chunk ),
				0,
				Tabulated_bp-Domains,
			        Gen_tabulated_[ index ] ) ;
    struct driver dr ;

    check_read_spec_performed();

    Tabulated_bp->domain->ar.element = &Domains[ domain ] ;
    dr.null = write_null ;
    dr.leaf_out = gen_null ;
    dr.leaf_in = write_tabulated_leaf_in ;
    dr.simple_in = write_simple_in ;
    dr.array_leaf = write_array_leaf ;
    dr.simple_out = write_simple_out ;
    dr.obj_in = write_obj_in ;
    dr.obj_out = write_obj_out ;
    user_file = fd ;

    push_gen_trav_env() ;

    shared_pointers( fake_obj, FALSE ) ;
    (void) fprintf(fd, "%d %d %d ", 
		   domain, max_tabulated_elements(), shared_number ) ;
    gen_trav_obj( fake_obj, &dr ) ;

    pop_gen_trav_env() ;

    newgen_free((char *) fake_obj ) ;
    return( domain ) ;
}



#ifdef BSD
static char *strdup( s )
char *s ;
{
    char *new = (char*) malloc( strlen( s )+1 ) ;
    strcpy( new, s ) ;
    return new;
}
#endif

/* GEN_READ_SPEC reads the specifications. This has to be used
   -- before -- any utilization of manipulation functions. */

static void init_gen_quick_recurse_tables(void);

extern void genspec_set_string_to_parse(char*);
extern void genspec_reset_string_to_parse(void);

void
gen_read_spec(char * spec, ...)
{
    va_list ap ;
    gen_chunk **cpp ;
    struct gen_binding *bp ;
    extern int unlink();

    /* default initialization of newgen lexers files:
     */
    genread_in = stdin;
    genspec_in = stdin;
    genread_out = stdout;
    genspec_out = stdout;

    /* now let's read the spec strings...
     */
    va_start(ap, spec) ;

    init() ;
    Read_spec_mode = 1 ;

    while(spec)
    {
	genspec_set_string_to_parse(spec);
	genspec_parse() ;
	genspec_reset_string_to_parse();

	spec = va_arg( ap, char *);
    }

    compile() ;

    for( cpp= &Gen_tabulated_[0] ; 
	 cpp<&Gen_tabulated_[MAX_TABULATED] ; 
	 cpp++ ) {
	*cpp = gen_chunk_undefined ;
    }
    for( bp = Domains ; bp < &Domains[ MAX_DOMAIN ] ; bp++ ) {
	if( bp->name != NULL &&
	   !IS_INLINABLE( bp ) && !IS_EXTERNAL( bp ) &&
	   bp->domain->ba.type == IMPORT_DT ) {
	    user( "Cannot run with imported domains: %s\n", bp->name ) ;
	    return ;
	}
	if( IS_TABULATED( bp )) {
	    int i ;

	    bp->alloc = 1 ;
	    Gen_tabulated_[ bp->index ] = 
		(gen_chunk *)alloc(max_tabulated_elements()*sizeof(gen_chunk));
	    
	    for( i=0 ; i<max_tabulated_elements() ; i++ ) {
		(Gen_tabulated_[ bp->index ]+i)->p = gen_chunk_undefined ;
	    }
	    if( Gen_tabulated_names == NULL )
		gen_init_Gen_tabulated_names();
	}
    }
    gen_cp_ = &Gen_cp_[ 0 ] ;
    gen_hash_ = &Gen_hash_[ 0 ] ;
    Read_spec_mode = 0 ;
    Read_spec_performed = TRUE ;
    va_end( ap ) ;

    /* quick recurse decision tables initializations
     */
    init_gen_quick_recurse_tables();
}

/* GEN_INIT_EXTERNAL defines entry points for free, read and write functions 
   of external types */

void
gen_init_external( which, read, write, free, copy, allocated_memory )
int which ;
char *(*read)() ;
void (*write)() ;
void (*free)() ;
char *(*copy)() ;
int (*allocated_memory)() ;
{
	struct gen_binding *bp = &Domains[ which ] ;
	union domain *dp = bp->domain ;

	if( dp->ba.type != EXTERNAL_DT ) {
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
	dp->ex.allocated_memory = allocated_memory ;
}

/* GEN_MAKE_ARRAY allocates an initialized array of NUM gen_chunks. */

gen_chunk *
gen_make_array( num )
     int num ;
{
  int i ;
  /*NOSTRICT*/
  gen_chunk *ar = (gen_chunk *)alloc( sizeof( gen_chunk )) ;

  for( i=0 ; i<num ; i++ ) 
    ar[ i ].p = gen_chunk_undefined ;

  return( ar ) ;
}

/* GEN_READ reads any object from the FILE stream. Sharing is restored.
 */

gen_chunk *
gen_read( file )
     FILE *file ;
{
  genread_in = file ;
  genread_parse() ;
  return( Read_chunk ) ;
}

/* GEN_READ_TABULATED reads FILE to update the Gen_tabulated_ table. Creates
   if CREATE_P is true. */

int
gen_read_tabulated( file, create_p )
FILE *file ;
int create_p ;
{
    /* gen_chunk *cp ; */
    int domain, index, max ;
    int i ;
    extern int allow_forward_ref ;

    genread_in = file ;
    if( (i=genread_lex()) != READ_INT ) {
	user( "Incorrect data for gen_read_tabulated: %d\n", i ) ;
	exit( 1 ) ;
    }
    domain = genread_lval.val ;

    if( (i=genread_lex()) != READ_INT ) {
	user( "Incorrect second data for gen_read_tabulated: %d\n", i ) ;
	exit( 1 ) ;
    }
    max = genread_lval.val ;

    /* (void) fscanf( file, "%d %d", &domain, &max ) ; */

    if( max != max_tabulated_elements()) {
	user( "gen_read_tabulated: Getting %d, expecting %d elements\n\n", 
	       max,  max_tabulated_elements()) ;
	if(max > max_tabulated_elements()) {
	    user("Current limit (%d) can be redefined by setting environment "
		 "variable NEWGEN_MAX_TABULATED_ELEMENTS\n",
		 max_tabulated_elements());
	  exit(1);
	}
    }
    if( create_p ) {
	if( Gen_tabulated_[ index = Domains[ domain ].index ] == NULL ) {
	    user( "gen_read_tabulated: Trying to read untabulated domain %s\n",
		  Domains[ domain ].name ) ;
	}
	Domains[ domain ].alloc = 1 ;

	for( i = 0 ; i < max_tabulated_elements() ; i++ ) {
	    (Gen_tabulated_[ index ]+i)->p = gen_chunk_undefined ;
	}
    }
    allow_forward_ref = TRUE ;
    genread_parse() ;
    allow_forward_ref = FALSE ;

    newgen_free((char *) ((Read_chunk+1)->p) ) ;
    newgen_free((char *) Read_chunk ) ;
    return( domain ) ;
}

int
gen_read_and_check_tabulated( file, create_p )
FILE *file ;
int create_p ;
{
    int domain ;

    domain = gen_read_tabulated( file, create_p ) ;

    HASH_MAP( k, v, {
	gen_chunk *hash = (gen_chunk *)v ;

        if( hash->i < 0 ) {
            user( "Tabulated element not defined: %s\n", k ) ;
        }
    }, Gen_tabulated_names ) ;
    return( domain ) ;
}

/* GEN_CHECK checks that the gen_chunk received OBJ is of the appropriate TYPE.
 */ 
gen_chunk *
gen_check( obj, t )
gen_chunk *obj ;
int t ;
{
    extern int max_domain_index() ;
    int max_index ;

    if( obj == NULL ) {
	(void) user("gen_check: NULL pointer, expecting type %s\n",
		    Domains[ t ].name) ;
	abort() ;
    }
    max_index = max_domain_index() ;
    message_assert("Improper domain_index", max_index >= 0 ) ;

    if( obj != gen_chunk_undefined && t != obj->i ) {
	user("gen_check: Type clash (expecting %s, getting %s)\n",
	     Domains[ t ].name, 
	     (obj->i >= 0 && obj->i <= max_index ) ?
	     Domains[ obj->i ].name :
	     "???") ;
	abort() ;
    }
    return( obj ) ;
}

/* GEN_TYPE returns the domain number for the object in argument
 * 
 * FC 29/12/94
 */
int
gen_type(obj)
gen_chunk *obj;
{
    int dom;

    message_assert("no domain for NULL object", obj!=(gen_chunk*)NULL);
    message_assert("no domain for undefined object", 
		   !gen_chunk_undefined_p(obj)); 

    dom = obj->i; check_domain(dom);

    return(dom);
}

/*  GEN_DOMAIN_NAME returns the domain name, and may be used for debug
 *  purposes. It should be a valid domain name.
 *  
 *  FC 29/12/94
 */
char *gen_domain_name(t)
int t;
{
    check_domain(t); return(Domains[t].name);
}

extern int error_seen ;

/*  used for consistence checking...
 */
static FILE *black_hole = NULL ;
static void open_black_hole()
{
    if (black_hole == NULL)  
	if ((black_hole=fopen("/dev/null", "r")) == NULL) 
	    fatal("Cannot open /dev/null !") ; /* not reached */
}

/* GEN_CONSISTENT_P dynamically checks the type correctness of OBJ. 
 */
int
gen_consistent_p( obj )
gen_chunk *obj ;
{
    int old_gen_debug = gen_debug ;

    check_read_spec_performed();
    open_black_hole();

    error_seen = 0 ;
    gen_debug = GEN_DBG_CHECK ;
    gen_write( black_hole, obj ) ;
    gen_debug = old_gen_debug ;
    return( error_seen  == 0 ) ;
}

int
gen_tabulated_consistent_p(int domain)
{
    struct gen_binding *bp = &Domains[ domain ];
    gen_chunk * t = Gen_tabulated_[bp->index];
    int i, size = max_tabulated_elements();

    for (i=0; i<size; i++)
    {
	if (t[i].p && !gen_chunk_undefined_p(t[i].p)) {
	    /* fprintf(stderr, "consistence of [%d] (0x%x) %s \n", i,
		    (unsigned int) t[i].p, ((t[i].p)+2)->s); */
	    
	    gen_consistent_p(t[i].p);
	}
    }

    return 1;
}

/* GEN_DEFINED_P checks that the OBJect is fully defined 
*/
static void
defined_null( bp )
struct gen_binding *bp ;
{
    union domain *dp = bp->domain ;

    user( "", (char *)NULL ) ;
    (void) fprintf( stderr, "gen_defined_p: Undefined object of type < " );
    print_domain( stderr, dp ) ;
    (void) fprintf( stderr, "> found\n" ) ;
}
  
int
gen_defined_p( obj )
gen_chunk *obj ;
{
    struct driver dr ;

    check_read_spec_performed();
    open_black_hole();

    error_seen = 0  ;
    dr.null = defined_null ;
    dr.leaf_out = gen_null ;
    dr.leaf_in = write_leaf_in ;
    dr.simple_in = write_simple_in ;
    dr.array_leaf = gen_array_leaf ;
    dr.simple_out = write_simple_out ;
    dr.obj_in = write_obj_in ;
    dr.obj_out = write_obj_out ;
    user_file = black_hole ;

    push_gen_trav_env() ;

    shared_pointers( obj, FALSE ) ;
    gen_trav_obj( obj, &dr ) ;

    pop_gen_trav_env() ;

    return( error_seen == 0 ) ;
}

/* GEN_SHARING_P checks whether OBJ1 uses objects (except tabulated) or
   CONS cells that appear in OBJ2. */

static hash_table pointers = (hash_table)NULL;
static jmp_buf env  ;

static bool
check_sharing(p, type)
char *p ;
char *type ;
{
  if( hash_get( pointers, p ) != HASH_UNDEFINED_VALUE ) {
      user("Sharing of %s detected on %p", type, p ) ;
    longjmp( env, 1 ) ;
    /* NOTREACHED*/
  }
  return( FALSE ) ;
}

static int
sharing_obj_in( obj, dr )
gen_chunk *obj ;
struct driver *dr ;
{ 
    /* char *seen ; */

  if (shared_obj(obj, gen_null, gen_null))
    return( !GO ) ;

  if (IS_TABULATED( &Domains[quick_domain_index(obj)]))
    return( !GO ) ;

  check_sharing( (char *)obj, "CHUNK *" ) ;
  return( GO ) ;
}

static int
sharing_simple_in( obj, dp )
gen_chunk *obj ;
union domain *dp ;
{
  cons *p ;

  switch( dp->ba.type ) {
  case LIST_DT:
    if( obj->l == list_undefined ) {
      return( !GO) ;
    }
    for( p=obj->l ; p!=NIL ; p=p->cdr ) {
      check_sharing( (char *)p, "CONS *" ) ;
    }
  }
  return( persistant_simple_in( obj, dp )) ;
}

bool
gen_sharing_p( obj1, obj2 )
gen_chunk *obj1, *obj2 ;
{
  struct driver dr ;
  bool found ;

  check_read_spec_performed();

  if( pointers == (hash_table)NULL )
      pointers = hash_table_make(hash_pointer, 0);
  else
    hash_table_clear(pointers) ;

  dr.null = dr.leaf_out = dr.simple_out = dr.obj_out =  gen_null ;
  dr.obj_in = sharing_obj_in ;
  dr.simple_in = sharing_simple_in ;
  dr.array_leaf = gen_array_leaf ;
  dr.leaf_in = tabulated_leaf_in ;

  push_gen_trav_env() ;
  shared_pointers(obj2, FALSE );

  HASH_MAP( k, v, {
    hash_put( pointers, k, v ) ;
  }, obj_table ) ;

  shared_pointers( obj1, FALSE ) ;

  if( (found=setjmp( env )) == 0 )
    gen_trav_obj( obj1, &dr ) ;

  pop_gen_trav_env() ;

  return(found) ;
}

/******************************************************************** SIZE */

/* returns the number of bytes allocated for a given structure
 * may need additional fonctions for externals...
 * May be called recursively. If so, already_seen_objects table kept.
 */

static int current_size;
static hash_table already_seen_objects = NULL;

/*  true if obj was already seen in this recursion, and put it at TRUE
 */
static bool 
allocated_memory_already_seen_p(obj)
gen_chunk * obj;
{
    if (hash_get(already_seen_objects, (char *)obj)==(char*)TRUE)
	return TRUE;
    hash_put(already_seen_objects, (char *)obj, (char *) TRUE);
    return FALSE;
}

/* manages EXTERNALS and INLINABLES
 */
static int
allocated_memory_leaf_in( obj, bp )
gen_chunk *obj ;
struct gen_binding *bp ;
{
    if (IS_INLINABLE(bp))
    {
	if (*bp->name=='s' && obj->s && !string_undefined_p(obj->s) &&
	    !allocated_memory_already_seen_p(obj->s))
	    current_size += strlen(obj->s) + 1; /* under approximation! */

	return !GO;
    }

    if (IS_TABULATED(bp) || allocated_memory_already_seen_p(obj)) 
	return FALSE;

    if (IS_EXTERNAL(bp))
    {
	if (bp->domain->ex.allocated_memory)
	    current_size += (*(bp->domain->ex.allocated_memory))(obj->s);
	else
	    user("[gen_allocated_memory] warning: "
		 "external with no allocated memory function\n");
		
	return FALSE;
    }

    return TRUE;
}

/* manages newgen objects and strings...
 */
static int
allocated_memory_obj_in(
    gen_chunk *obj,
    struct driver *dr)
{
    struct gen_binding *bp = &Domains[quick_domain_index(obj)];

    if (allocated_memory_already_seen_p(obj) ||	
	IS_TABULATED(bp) || IS_INLINABLE(bp))
	return !GO;

    /* gen size is quite slow. should precompute sizes...
     */
    current_size += sizeof(gen_chunk*)*gen_size(bp); 

    return GO;
}

/* manages newgen simples (list, set, array)
 */
static int
allocated_memory_simple_in(
    gen_chunk *obj,
    union domain *dp)
{
    if (dp->ba.persistant) return FALSE;

    switch( dp->ba.type ) {
    case BASIS_DT:
	return( GO) ; /* !GO ??? */
    case LIST_DT:
    {
	list l = obj->l;
	
	if (l && !list_undefined_p(l))
	{
	    current_size += list_own_allocated_memory(l);
	    return TRUE;
	}
	else
	    return FALSE;
    }
    case SET_DT:
    {
	set s = obj->t;

	if (!set_undefined_p(s))
	{
	    current_size += set_own_allocated_memory(s);
	    return TRUE;
	}
	else
	    return FALSE;
    }
    case ARRAY_DT:
    {
	gen_chunk *p = obj->p;

	if (!array_undefined_p(p))
	{
	    current_size += array_own_allocated_memory(dp);
	    return TRUE;
	}
	else
	    return FALSE;
    }
    }

    fatal("allocated_memory_simple_in: unknown type %s\n", itoa(dp->ba.type));
    return -1; /* just to avoid a gcc warning */
}

/* re-entry is automatic for this function.
 */
int /* in bytes */
gen_allocated_memory(
    gen_chunk *obj)
{
    bool first_on_stack = (already_seen_objects==NULL);
    int result, saved_size;
    struct driver dr;

    /* save current status 
     */
    saved_size = current_size;
    current_size = 0;
    if (first_on_stack) 
	already_seen_objects = hash_table_make(hash_pointer, 0);
    
    /* build driver for gen_trav...
     */
    dr.null 		= gen_null,
    dr.leaf_in 		= allocated_memory_leaf_in,
    dr.leaf_out  	= gen_null,
    dr.simple_in 	= allocated_memory_simple_in,
    dr.array_leaf 	= gen_array_leaf,
    dr.simple_out 	= gen_null,
    dr.obj_in 		= allocated_memory_obj_in,
    dr.obj_out 		= gen_null;

    /* recursion from obj
     */
    gen_trav_obj(obj, &dr);

    /* restores status and returns result
     */
    result = current_size;
    current_size = saved_size;
    if (first_on_stack) 
    {
	hash_table_free(already_seen_objects); 
	already_seen_objects = NULL;
    }
    
    return result;
}
  
/* -------------------------------------------------------------
 *
 *    quick and Intelligent Recursion Thru Gen_Multi_Recurse
 *
 *    Fabien COELHO, Jun-Sep-Dec 94
 *
 */

/* Useful functions
 *
 * they may be used by some recursion
 *  - when no rewrite is needed
 *  - when the filter is always yes
 *  - when it is false, to stop the recursion on some types
 */
void gen_null(gen_chunk *p){}
bool gen_true(gen_chunk *p){ return TRUE;}
bool gen_false(gen_chunk *p){ return FALSE;}
void gen_core(gen_chunk *p){ abort();}

/* GLOBAL VARIABLES: to deal with decision tables
 *
 *   number_of_domains: 
 *     the number of domains managed by newgen, max is MAX_DOMAIN.
 *
 *   DirectDomainsTable:
 *     DirectDomainsTable[domain_1, domain_2] is TRUE if domain_2
 *     may contains *directly* a domain_1 field.
 *
 *   DecisionTables:
 *     DecisionTables[domain] is the decision table to scan domain.
 *     They are to be computed/set up after specifications' load.
 *     A demand driven approach is implemented.
 *
 */

#define decision_table_undefined ((char)25)
typedef char GenDecisionTableType[MAX_DOMAIN];
typedef GenDecisionTableType gen_tables[MAX_DOMAIN];

static int 
    number_of_domains = -1;

static gen_tables 
    DirectDomainsTable, 
    DecisionTables;

static void
print_decision_table(t)
GenDecisionTableType t;
{
    int i;
    
    fprintf(stderr, "[print_decision_table] %p\n", t);

    for (i=0; i<MAX_DOMAIN; i++)
	if (t[i]) fprintf(stderr, "  go through %s\n", Domains[i].name);
}

/* demand driven computation of the decision table to scan domain.
 * this table is computed by a closure from the initial type matrix
 *
 * the algorithm is straightforward. 
 * tabulated domains are skipped.
 * worst case complexity if O(n^2) for each requested domain.
 * the closure is shorten because already computed tables are used.
 */
static void 
initialize_domain_DecisionTables(domain)
int domain;
{
    GenDecisionTableType not_used;
    int i, j;
 
    if (gen_debug & GEN_DBG_RECURSE)
	fprintf(stderr,
		"[initialize_domain_DecisionTables] domain %s (%d)\n",
		Domains[domain].name, domain);

    for (i=0; i<number_of_domains; i++) 
	not_used[i] = TRUE; 

    /*   init with direct inclusions
     */
    for (i=0; i<number_of_domains; i++) 
	DecisionTables[domain][i] = DirectDomainsTable[domain][i];

    not_used[domain]=FALSE;

    /*   now the closure is computed
     */
    
    while(1)
    {
	/*   look for the next domain to include
	 */
	for (i=0; i<number_of_domains; i++)
	    if (DecisionTables[domain][i] & not_used[i])
		break;
	
	if (i>=number_of_domains) break; /* none */

	not_used[i] = FALSE;

	/*   cannot come from tabulated domains...
	 *   this should be discussed, or put as a parameter...
	 */
	if (IS_TABULATED(&Domains[i])) continue;

	if (*DecisionTables[i]!=decision_table_undefined)
	{
	    /* shorten */
	    if (gen_debug & GEN_DBG_RECURSE)
	    fprintf(stderr, 
		    " - shortening with already computed %s (%d)\n", 
		    Domains[i].name, i);

	    for (j=0; j<number_of_domains; j++)
		DecisionTables[domain][j] |= DecisionTables[i][j],
		not_used[j] &= !DecisionTables[i][j] /*? FALSE : not_used[j] */;
	}
	else
	{
	    if (gen_debug & GEN_DBG_RECURSE)
		fprintf(stderr, 
			" - including %s (%d)\n", 
			Domains[i].name, i);

	    for (j=0; j<number_of_domains; j++)
		DecisionTables[domain][j] |= DirectDomainsTable[i][j];
	}
    }

    if (gen_debug & GEN_DBG_RECURSE)
	fprintf(stderr, " - computed table is\n"),
	print_decision_table(DecisionTables[domain]);
}

/*   walks thru the domain to tag all types for target.
 */
static void 
initialize_domain_DirectDomainsTable(target, dp)
int target;
union domain *dp;
{
    if (dp==NULL) return; /* some domains are NULL */

    switch(dp->ba.type)
    {
    case EXTERNAL_DT:
	break; /* obvious: don't go inside externals! */
    case BASIS_DT:
    case LIST_DT:
    case ARRAY_DT:
    case SET_DT:
	if (gen_debug & GEN_DBG_RECURSE)
	    fprintf(stderr,
		    " - setting %s (%d) contains %s (%d)\n",
		    Domains[target].name, target,
		    dp->se.element->name, dp->se.element-Domains);
	DirectDomainsTable[dp->se.element-Domains][target] = TRUE;
	break;
    case CONSTRUCTED_DT:
    {
	struct domainlist *l=dp->co.components;

	for (; l!=NULL; l=l->cdr)
	    initialize_domain_DirectDomainsTable(target, l->domain);
    }
    case IMPORT_DT:
	break; /* abort() ? TRUE (safe) ? */
    case UNDEF_DT:
	break; /* nothing is done */
    default: 
	fprintf(stderr, "newgen: unexpected domain type (%d)\n", dp->ba.type),
	abort();
    }
} 

static void 
initialize_DirectDomainsTable()
{
    int i,j;

    check_read_spec_performed();

    for (i=0; i<number_of_domains; i++)
    {
	struct gen_binding *bp = &Domains[i];

	if (gen_debug & GEN_DBG_RECURSE)
	    fprintf(stderr, 
		    "[initialized_DirectDomainsTable] analysing %s\n",
		    bp->name);

	if( bp->name == NULL || bp == Tabulated_bp ) continue ; /* ? */

	/*   first put falses 
	 */
	for (j=0; j<number_of_domains; j++) 
	    DirectDomainsTable[j][i]=FALSE;

	initialize_domain_DirectDomainsTable(i, bp->domain);
    }

    if (gen_debug & GEN_DBG_RECURSE)
	for (i=0; i<number_of_domains; i++)
	    fprintf(stderr, "[initialized_DirectDomainsTable] %s (%d)\n", 
		    Domains[i].name, i),
	    print_decision_table(DirectDomainsTable[i]);

}

static void 
initialize_DecisionTables()
{
    int i;

    for(i=0; i<MAX_DOMAIN; i++)
	*DecisionTables[i]=decision_table_undefined;
}

/*    called by gen_read_spec, should be called by a gen_init()
 */
static void 
init_gen_quick_recurse_tables()
{
    int i;

    check_read_spec_performed();

    /*   number_of_domains is first set
     */
    for (number_of_domains=-1, i=0; i<MAX_DOMAIN; i++) 
	if (Domains[i].domain!=NULL && Domains[i].domain->ba.type != UNDEF_DT) 
	    number_of_domains = i;
    number_of_domains++;

    if (gen_debug & GEN_DBG_RECURSE)
	fprintf(stderr, 
		"[init_gen_quick_recurse_tables] %d domains\n",
		number_of_domains);

    initialize_DirectDomainsTable();
    initialize_DecisionTables();   
}

/* returns a decision table for the given domain.
 * demand driven definition of the table.
 */
static GenDecisionTableType
*get_decision_table(domain)
int domain;
{
    check_domain(domain);

    if (*DecisionTables[domain]==decision_table_undefined) 
	initialize_domain_DecisionTables(domain);

    return(&DecisionTables[domain]);
}

/*******************************************************************
 *
 *            GENERALIZED VERSION: GEN MULTI RECURSE
 *
 *       Fabien COELHO, Wed Sep  7 21:39:47 MET DST 1994
 *
 */

typedef bool (*GenFilterType)();
typedef void (*GenRewriteType)();

typedef GenFilterType GenFilterTableType[MAX_DOMAIN];
typedef GenRewriteType GenRewriteTableType[MAX_DOMAIN];

/* the current data needed for a multi recursion are 
 * stored in a multi recurse struct. 
 *
 * - the seen hash_table stores the already encountered obj,
 *   not to walk twice thru them. The previous implementation
 *   used 2 recursions, one to mark the obj to visit, with an
 *   associated number, and the other was the actual recursion.
 *   This version is lazy, and just marks the encountered nodes,
 *   thus allowing the full benefit of the decision table to avoid
 *   walking thru the whole data structure.
 * - the visited domains are marked true in domains.
 *   I could have checked that the filter is not NULL,
 *   but it is clearer this way, I think, for the one who 
 *   will try to understand, if any:-)
 * - the decision table used is in decisions. It is computed
 *   as the logical sum of the decision tables for the domains 
 *   to be walked thru
 * - filters and rewrites store the user decision functions 
 *   for each domain.
 */
struct multi_recurse 
{
    hash_table           seen;
    GenDecisionTableType *domains;
    GenDecisionTableType *decisions;
    GenFilterTableType   *filters;
    GenRewriteTableType  *rewrites;
};

/* the current multi recurse driver.
 * it is cleaner than the gen_recurse version since I added 
 * the decisions table without modifying Pierre code, while here
 * I redefined a current status struct that stores everything 
 * needed. 
 */
static struct multi_recurse
    *current_mrc = (struct multi_recurse *) NULL;

/* MULTI RECURSE FUNCTIONS
 */

/*  true if obj was already seen in this recursion, and put it at TRUE
 */
static bool 
quick_multi_already_seen_p(obj)
gen_chunk * obj;
{
    if (hash_get(current_mrc->seen, (char *)obj)==(char*)TRUE)
	return(TRUE);

    hash_put(current_mrc->seen, (char *)obj, (char *) TRUE);
    return(FALSE);
}

static int
quick_multi_recurse_obj_in(obj, dr)
gen_chunk *obj;
struct driver *dr;
{
    int dom = obj->i;
    check_domain(dom);

    /* don't walk twice thru the same object:
     */
    if (quick_multi_already_seen_p(obj) ||
    /* 
     * temporarily, tabulated objects are not walked thru.
     * the decision could be managed by the table, or *after* the
     * filtering: the current status implied that you cannot enumerate 
     * tabulated elements for instance. 
     *
     * these features/bugs/limitations are compatible with gen_slow_recurse.
     *
     * FI told me that only persistant edges shouldn't be followed.
     * it may mean that tabulated elements are always persistent?
     */
	IS_TABULATED(&Domains[quick_domain_index(obj)]))
	return(!GO);

    /* filter case
     */
    if ((*(current_mrc->domains))[dom]) 
	return((*((*(current_mrc->filters))[dom]))(obj));

    /* else, here is the *maybe* intelligent decision to be made.
     */
    return((*(current_mrc->decisions))[dom]);
}

static void
quick_multi_recurse_obj_out(obj, bp, dr)
gen_chunk *obj;
struct gen_binding *bp;
struct driver *dr;
{
    int dom = obj->i;
    check_domain(dom);

    if ((*(current_mrc->domains))[dom])
	(*((*(current_mrc->rewrites))[dom]))(obj);
}

static int
quick_multi_recurse_simple_in(obj, dp)
gen_chunk *obj ;
union domain *dp ;
{
    int t;

    return(((*(current_mrc->decisions))[dp->se.element-Domains] ||
	    (*(current_mrc->domains))[dp->se.element-Domains]) &&
	   (!dp->se.persistant) &&               /* stay at a given level */
	   ((t=dp->ba.type)==BASIS_DT ? TRUE :
	     t==LIST_DT               ? obj->l != list_undefined :
	     t==SET_DT                ? obj->t != set_undefined :
	     t==ARRAY_DT              ? obj->p != array_undefined : 
	    (fatal("persistant_simple_in: unknown type %s\n", 
		   itoa(dp->ba.type)), FALSE))); 
}

/*  tells the recursion not to go in this object
 *  This may be interesting when the recursion modifies
 *  the visited data structure.
 *  if obj is NULL, the whole recursion is stopped !
 */
void 
gen_recurse_stop(obj)
gen_chunk *obj;
{
    if (obj)
	hash_put(current_mrc->seen, (char *)obj, (char *)TRUE);
    else
	gen_trav_stop_recursion = TRUE;
}

/*  MULTI RECURSION FUNCTION
 *
 *  gen_multi_recurse(obj,
 *                   [domain, filter, rewrite,]*
 *                    NULL);
 *
 *  recurse from object obj,
 *  applies filter_i on encountered domain_i objects,
 *  if true, recurses down from the domain_i object, 
 *       and applies rewrite_i on exit from the object.
 *
 * ??? bug : you can't visit domain 0 if any... 
 * I can't remember what it is used for.
 */
void gen_multi_recurse(gen_chunk * obj, ...)
{
    va_list pvar;
    int i, domain;
    GenFilterTableType new_filter_table;
    GenRewriteTableType new_rewrite_table;
    GenDecisionTableType new_decision_table, new_domain_table, *p_table;
    struct multi_recurse *saved_mrc, new_mrc;
    struct driver dr;

    check_read_spec_performed();

    va_start(pvar, obj);

    /*  the object must be a valid newgen object
     */
    message_assert("null or undefined object to visit",
		   obj!=(gen_chunk*)NULL && obj!=gen_chunk_undefined);

    /*    initialize the new tables
     */
    for(i=0; i<MAX_DOMAIN; i++)
	new_domain_table[i]=FALSE,
	new_decision_table[i]=FALSE,
	new_filter_table[i]=NULL,
	new_rewrite_table[i]=NULL;
    
    /*    read the arguments
     */
    while((domain=va_arg(pvar, int))!=(int)NULL)
    {
	message_assert("domain specified more than once",
		       !new_domain_table[domain]);

	new_domain_table[domain]  = TRUE;
	new_filter_table[domain]  = va_arg(pvar, GenFilterType);
	new_rewrite_table[domain] = va_arg(pvar, GenRewriteType);

	for(i=0, p_table=get_decision_table(domain); i<number_of_domains; i++)
	    new_decision_table[i] |= (*p_table)[i];
    }
    
    va_end(pvar);

    new_mrc.seen      = hash_table_make(hash_pointer, 0),
    new_mrc.domains   = &new_domain_table,
    new_mrc.decisions = &new_decision_table,
    new_mrc.filters   = &new_filter_table,
    new_mrc.rewrites  = &new_rewrite_table;

    dr.null 		= gen_null,
    dr.leaf_in 		= tabulated_leaf_in,
    dr.leaf_out  	= gen_null,
    dr.simple_in 	= quick_multi_recurse_simple_in,
    dr.array_leaf 	= gen_array_leaf,
    dr.simple_out 	= gen_null,
    dr.obj_in 		= quick_multi_recurse_obj_in,
    dr.obj_out 		= quick_multi_recurse_obj_out;

    /*    push the current context
     */
    saved_mrc = current_mrc, current_mrc = &new_mrc;

    /*  recurse!
     */
    gen_trav_stop_recursion = FALSE;
    gen_trav_obj(obj, &dr);
    gen_trav_stop_recursion = FALSE;
    
    /*  restore the previous context
     */
    hash_table_free(new_mrc.seen);
    current_mrc = saved_mrc;
}

/*  upward compatibility, old gen_recurse function syntax
 *  could be a define.
 */

#ifdef gen_recurse 
#undef gen_recurse
#endif

void 
gen_recurse(obj, domain, filter, rewrite)
gen_chunk *obj;
int domain;
bool (*filter)();
void (*rewrite)();
{
    gen_multi_recurse(obj, domain, filter, rewrite, NULL);
}

/*    That is all
 */
