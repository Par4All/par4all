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


/* - set.c
   
   Pierre Jouvelot (3 Avril 1989)

   Set package for any type of pointer. 

   To avoid sharing problem, all the routines are 3-adress: S1 = S2 op S3.
   It is up to the user to know what to do (e.g., freeing some temporary
   memory storage) before S1 is assigned a new value. */

#include <stdio.h>
#include <stdlib.h>
#include "newgen_types.h"
#include "genC.h"
#include "newgen_set.h"

#define INITIAL_SET_SIZE 10

void set_clear(), set_free();

/* Implementation of the Set package. */
    
set set_make( type )
set_type type ;
{
    set hp = (set)malloc( sizeof( struct set )) ;

    if( hp == (set)NULL ) {
	(void) fprintf( stderr, "set_make: cannot allocate\n" ) ;
	exit( 1 ) ;
    }
    hp->table = hash_table_make( type, INITIAL_SET_SIZE ) ;
    hp->type = type ;
    return( hp ) ;
}

set set_singleton( type, p ) 
set_type type ;
char *p ;
{
    set s = set_make( type ) ;

    hash_put( s->table, p, p ) ;
    return( s ) ;
}

set set_assign( s1, s2 )
set s1, s2 ;
{
    if( s1 == s2 ) {
	return( s1 ) ;
    }
    else {
	set_clear( s1 ) ;
	HASH_MAP( k, v, {hash_put( s1->table, k, v ) ;}, s2->table ) ;
	return( s1 ) ;
    }
}

set set_add_element(s1, s2, e )
set s1, s2 ;
char *e;
{
    if( s1 == s2 ) {
	if (! set_belong_p(s1, e))
	    hash_put(s1->table, e, e);
	return( s1 ) ;
    }
    else {
	set_clear( s1 ) ;
	HASH_MAP( k, v, {hash_put( s1->table, k, v ) ;}, s2->table ) ;
	if (! set_belong_p(s1, e))
	    hash_put(s1->table, e, e);
	return( s1 ) ;
    }
}

bool set_belong_p(s, e )
set s;
char *e;
{
    /* GO 7/8/95:
       Problem for set_string type because the value returned by
       hash_get is not the same than the pointer value, only the
       content of the string is the same ...
       
       return( hash_get(s->table, (char *) e) == (char *) e) ;
       */

    return hash_get(s->table, (char *) e) != HASH_UNDEFINED_VALUE;
}

set set_union( s1, s2, s3 )
set s1, s2, s3 ;
{
    if( s1 != s3 ) {
	set_assign( s1, s2 ) ;
	HASH_MAP( k, v, {hash_put( s1->table, k, v ) ;}, s3->table ) ;
    }
    else {
	HASH_MAP( k, v, {hash_put( s1->table, k, v ) ;}, s2->table ) ;
    }
    return( s1 ) ;
}

set set_intersection( s1, s2, s3 )
set s1, s2, s3 ;
{
    if( s1 != s2 && s1 != s3 ) {
	set_clear( s1 ) ;
	HASH_MAP( k, v, {if( hash_get( s2->table, k )
			    != (char *)HASH_UNDEFINED_VALUE ) 
				 hash_put( s1->table, k, v ) ;}, 
		 s3->table ) ;
	return( s1 ) ;
    }
    else {
	set tmp = set_make( s1->type ) ;

	HASH_MAP( k, v, {if( hash_get( s1->table, k ) 
			    != (char *)HASH_UNDEFINED_VALUE ) 
				 hash_put( tmp->table, k, v ) ;}, 
		 (s1 == s2) ? s3->table : s2->table ) ;
	set_assign( s1, tmp ) ;
	set_free( tmp ) ;
	return( s1 ) ;
    }
}

set set_difference( s1, s2, s3 )
set s1, s2, s3 ;
{
    set_assign( s1, s2 ) ;
    HASH_MAP( k, ignore, {hash_del( s1->table, k );}, s3->table ) ;
    return( s1 ) ;
}

set set_del_element( s1, s2, e )
set s1, s2 ;
char *e ;
{
    set_assign( s1, s2 ) ;
    hash_del( s1->table, e );
    return( s1 ) ;
}

bool set_equal( s1, s2 )
set s1, s2 ;
{
    bool equal ;
    
    equal = TRUE ;
    HASH_MAP( k, ignore, {
	if( hash_get( s2->table, k ) == HASH_UNDEFINED_VALUE ) 
		return( FALSE );
    }, s1->table ) ;
    HASH_MAP( k, ignore, {
	if( hash_get( s1->table, k ) == HASH_UNDEFINED_VALUE )
		return( FALSE );
    }, s2->table ) ;
    return( equal ) ;
}

void set_clear( s )
set s ;
{
    hash_table_clear( s->table ) ;
}

void set_free( s )
set s ;
{
    hash_table_free( s->table ) ;
    free( s ) ;
}

