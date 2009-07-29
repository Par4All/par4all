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

/*
   Pierre Jouvelot (3 Avril 1989)

   Set package for any type of pointer.

   To avoid sharing problem, all the routines are 3-adress: S1 = S2 op S3.
   It is up to the user to know what to do (e.g., freeing some temporary
   memory storage) before S1 is assigned a new value.
*/

#include <stdio.h>
#include <stdlib.h>
#include "genC.h"
#include "newgen_set.h"

#define INITIAL_SET_SIZE 10

void set_clear(), set_free();

/* Implementation of the Set package. */
    
set set_make( typ )
set_type typ ;
{
    set hp = (set) alloc(sizeof(set_chunk));

    if( hp == (set)NULL ) {
	(void) fprintf( stderr, "set_make: cannot allocate\n" ) ;
	exit( 1 ) ;
    }
    hp->table = hash_table_make( typ, INITIAL_SET_SIZE ) ;
    hp->type = typ ;
    return( hp ) ;
}

set set_singleton(set_type type, void * p) 
{
    set s = set_make( type ) ;
    hash_put( s->table, p, p ) ;
    return( s ) ;
}


/* Assign a set with the content of another set.

   @param s1 the set to write into
   @param s2 the set to copy

   If the same set is given twice, nothing is done.

   @return the target set.
*/
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

set set_add_element(set s1, set s2, void * e)
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

bool set_belong_p(set s, void * e)
{
    /* GO 7/8/95:
       Problem for set_string type because the value returned by
       hash_get is not the same than the pointer value, only the
       content of the string is the same ...
       
       return( hash_get(s->table, (char *) e) == (char *) e) ;
       */

    return hash_get(s->table, e) != HASH_UNDEFINED_VALUE;
}

set set_union(set s1, set s2, set s3)
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

set set_intersection(set s1, set s2, set s3)
{
    if( s1 != s2 && s1 != s3 ) {
	set_clear( s1 ) ;
	HASH_MAP( k, v, {if( hash_get( s2->table, k )
			    != HASH_UNDEFINED_VALUE ) 
				 hash_put( s1->table, k, v ) ;}, 
		 s3->table ) ;
	return( s1 ) ;
    }
    else {
	set tmp = set_make( s1->type ) ;

	HASH_MAP( k, v, {if( hash_get( s1->table, k ) 
			    != HASH_UNDEFINED_VALUE ) 
				 hash_put( tmp->table, k, v ) ;}, 
		 (s1 == s2) ? s3->table : s2->table ) ;
	set_assign( s1, tmp ) ;
	set_free( tmp ) ;
	return( s1 ) ;
    }
}

set set_difference(set s1, set s2, set s3)
{
    set_assign( s1, s2 ) ;
    HASH_MAP( k, ignore, {hash_del( s1->table, k );}, s3->table ) ;
    return( s1 ) ;
}

set set_del_element(set s1, set s2, void * e)
{
    set_assign( s1, s2 ) ;
    hash_del( s1->table, e );
    return( s1 ) ;
}

/* May be useful for string sets ... NOT TESTED*/
set set_delfree_element(set s1, set s2, void * e)
{
  void * pe;
  set_assign( s1, s2 ) ;
  (void) hash_delget( s1->table, e ,&pe);
  free(pe);
  return( s1 ) ;
}

bool set_equal(set s1, set s2)
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

void set_clear(set s)
{
    hash_table_clear( s->table ) ;
}

void set_free(set s)
{
    hash_table_free(s->table);
    gen_free_area((void**) s, sizeof(set_chunk));
}

bool set_empty_p(set s)
{
    SET_MAP(x, return FALSE, s);
    return TRUE;
}

void
gen_set_closure_iterate(
    void (*iterate)(void *, set),
    set initial,
    bool dont_iterate_twice)
{
    set curr, next, seen;
    set_type t = initial->type;

    seen = set_make(t);
    curr = set_make(t);
    next = set_make(t);

    set_assign(curr, initial);

    while (!set_empty_p(curr))
    {
	SET_MAP(x, iterate(x, next), curr);
	if (dont_iterate_twice)
	{
	    (void) set_union(seen, seen, curr);
	    set_difference(curr, next, seen);
	}
	else
	{
	    set_assign(curr, next);
	}
	set_clear(next);
    }

    set_free(curr);
    set_free(seen);
    set_free(next);

}

/* a set-based implementation of gen_closure
 * that does not go twice in the same object.
 * FC 27/10/95.
 */
void gen_set_closure(
    void (*iterate)(void *, set),
    set initial)
{
    gen_set_closure_iterate(iterate, initial, TRUE);
}

int set_own_allocated_memory(set s)
{
    return sizeof(set_chunk)+hash_table_own_allocated_memory(s->table);
}

/**
 * create a list from a set
 * the set is not freed
 * @warning no assumption can be made on the ordering of returned list
 * @param s set where the data are
 *
 * @return an allocated list of elements from s
 */
list set_to_list(set s)
{
    list l =NIL;
    SET_MAP(v,l=gen_cons(v,l),s);
    return l;
}

/** 
 * turns a list into a set
 * all duplicated elements are lost
 * 
 * @param l list to turn into a set
 * @param st type of elements in the list
 * 
 * @return allocated set of elements from @a l
 * @warning list_to_set(set_to_list(s))!=s
 */
set list_to_set(list l,set_type st)
{
    set s = set_make(st);
    while(!ENDP(l)) {
        set_add_element(s,s,CAR(l).p);
        POP(l);
    }
    return s;
}
