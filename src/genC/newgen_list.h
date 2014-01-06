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
  These are the functions defined in the Newgen list library.
*/

#ifndef newgen_list_included
#define newgen_list_included

/** @addtogroup newgen_list */

/** @{ */

/** The structure used to build lists in NewGen

    cons is a list element, list is a pointer to these elements

    The names are quite related to their Lisp equivalents.
*/
typedef struct cons {
  gen_chunk car; /**< The data payload of a list element */
  struct cons *cdr ; /**< The pointer to the next element. It is NIL if none */
} cons;

/** The empty list (nil in Lisp) */
#define NIL ((list)NULL)

/** Modify a list pointer to point on the next element of the list

    If a list is considered as a stack and elements are pushed by
    insertion at the list head, popping an element from the stack can be
    seen as skipping the first one.

    Not that the list elements by themself are not modified.

    @param l the list to pop
*/
#define POP(l) ((l)=(l)->cdr)

/** Test if a list is empty

    @param l the list to test
    @return true if the list is the empty list, false else
*/
#define ENDP(l) ((l)==NIL)

/** Undefined list definition :-) */
#define list_undefined ((cons *)-3)

/** Return if a list is undefined

    Note it is different than testing for an empty list.
 */
#define list_undefined_p(c) ((c)==list_undefined)

/** Get the value of the first element of a list

    If CAR is applied on an empty list, a segmentation violation will
    occur.

    It is called CAR in the Lisp tradition, dating back to some register
    names in an old IBM computer where it was implemented.

    @paral pcons is the list

    @return the value of the first element of the list. Since it is
    gen_chunk NewGen generic container, it is good behaviour to use
    through a NewGen accessor, such as STATEMENT(CAR(l)) to the
    first PIPS statement of a list of statements
*/
#define CAR(pcons) ((pcons)->car)

/** Get the list less its first element

    Note that the list by itself is not modified, only a pointer to the
    second element is returned. So this pointer represent the original
    list less the firs element.

    If CDR is applied on an empty list, a segmentation violation will
    occur.

    It is called CDR in the Lisp tradition, dating back to some register
    names in an old IBM computer where it was implemented.

    @paral pcons is the list

    @return a pointer to the second element of the list. list following
    value of the first element of the list
*/
#define CDR(pcons) ((pcons)->cdr)

/** Get the adress of the first element of a list

    @paral pcons is the list

    @return a pointer to the value of the first element of the list
*/
#define REFCAR(pc) (&(CAR(pc).p))

/** List element cell constructor (insert an element at the beginning of a
    list)

    Mimmic the cons function in Lisp: construct a list element cell and
    link it to a list. So the element is the first element of a new list
    that goes on with the old one.

    @param _t_ is the type of the list element cell to construct

    @param _i_ is the element to put in the list element cell

    @param _l_ is the list to linked the element cell with a the beginning

    @return the new list with the new element cell at the beginning

    For example, to insert a PIPS statement s at the beginning of a
    statement list sl you can write:

      list new_list = CONS(statement, s, sl);

    Another way is to directly use the specialized NewGen list constructor
    for the type:

      list new_list = gen_statement_cons(s, sl);

    Note that it also works with just the type name in upper case, as:

      list l = CONS(STATEMENT, s, l);
*/
#define CONS(_t_,_i_,_l_) gen_##_t_##_cons((_i_),(_l_))

/** @} */

/* Some CPP magics to get a line-number-dependent "unique" identifier to
   have list iteration variable with unique names and avoid conflicts if
   we have multiple FOREACH in the same statement block:
*/
#define UNIQUE_NAME_1(prefix, x)   prefix##x
#define UNIQUE_NAME_2(prefix, x)   UNIQUE_NAME_1 (prefix, x)
/* Well, it does not work if 2 FOREACH are on the same line, but it should
   not happen if a PIPS programmer does not apply for the code offuscation
   contest... :-) */
#define UNIQUE_NAME  UNIQUE_NAME_2 (iter_, __LINE__)


/** @addtogroup newgen_list */

/** @{ */

/** Apply/map an instruction block on all the elements of a list

    FOREACH(T, v, l) {
      instructions;
    }
    iterates on all the elements of the list l of elements of type T by
    allocating a local element index variable v of type T. Instructions
    access the list element through variable v.

    FOREACH is similar to MAP but is more gdb/emacs/vim... friendly since
    it remains line oriented like the C99 for() used to implement it and
    does not choke on some "," in the instruction block.

    @param _fe_CASTER is the type of elements, aka the newgen type
    name or newgen basic types such as int, string, list.

    @param _fe_item is the variable to allocate and use as an iterator on
    the list elements

    @param _fe_list is the list parameter to iterate on
*/
#define FOREACH(_fe_CASTER, _fe_item, _fe_list)             \
  list UNIQUE_NAME = (_fe_list);                            \
  for( _fe_CASTER##_TYPE _fe_item;                          \
       !ENDP(UNIQUE_NAME) &&                                \
         (_fe_item= _fe_CASTER##_CAST(CAR(UNIQUE_NAME) ));  \
       POP(UNIQUE_NAME))

/** Apply some code on the addresses of all the elements of a list

    @param _map_list_cp is the variable that will iterate on the adresses
    of all the list elements to be accessed in _code

    @param _code is the statement (block) applied on each element address
    in the variable _map_list_cp

    @param _l is the list parameter to iterate on
*/
#define MAPL(_map_list_cp,_code,_l)                   \
  {                                                   \
    list _map_list_cp = (_l) ;                        \
    for(; !ENDP(_map_list_cp); POP(_map_list_cp))     \
      _code;                                          \
  }

/** Apply/map an instruction block on all the elements of a list (old
    fashioned)

    Now you should use the more modern FOREACH implementation instead.

    @param _map_CASTER is the caster of the type of elements, that is the
    newgen type name in uppercase, such as STATEMENT for a PIPS statement.

    @param _map_item is the variable to allocate and use as an iterator on
    the list elements from _map_code

    @param _map_code is the statement (block) applied on each element in
    the variable _map_item

    @param _map_list is the list parameter to iterate on
*/
#define MAP(_map_CASTER, _map_item, _map_code, _map_list) \
  {                                                       \
    list _map_item##_list = (_map_list);                  \
    _map_CASTER##_TYPE _map_item;                         \
    for(; _map_item##_list; POP(_map_item##_list))        \
    {                                                     \
      _map_item = _map_CASTER(CAR(_map_item##_list));     \
      _map_code;                                          \
    }                                                     \
  }

/** Another name to the funtion to insert a bool element at the start
    of a list */
#define gen_BOOL_cons gen_bool_cons

/** Another name to the funtion to insert an integer element at the start
    of a list */
#define gen_INT_cons gen_int_cons

/** Another name to the funtion to insert a list element at the start of a
    list

    It is to build list of lists
*/
#define gen_LIST_cons gen_list_cons

/** Another name to the funtion to insert a list element at the start of a
    list

    It is to build list of lists
*/
#define gen_CONSP_cons gen_list_cons

/** Another name to the funtion to insert a string element at the start of
    a list */
#define gen_STRING_cons gen_string_cons

/** @} */

/* #define CONS(type,x,l) gen_cons((void*) (x), (l)) */

// UNRELATED TO LISTS, and possibly in other files
extern void gen_copy(void *, void *);
extern bool gen_eq(const void *, const void *);
extern void *gen_identity(const void *);
extern void *gen_find_tabulated(const char*, int);
extern list gen_filter_tabulated(bool(*)(gen_chunk*), int);
extern void gen_free_area(void**, int);
extern void gen_mapc_tabulated(void (*)(), int);

// Functions in list.c

// LIST OPERATION
extern list gen_append(list , const list);
extern list gen_concatenate(const list , const list);
extern list gen_copy_seq(const list);
extern list gen_nconc(list, list);
extern list gen_full_copy_list(const list);
extern list gen_make_list(int, ...);
extern list gen_nreverse(list);

// EXTRACTIONS
extern void *gen_find(const void *, const list ,
		      gen_filter2_func_t, gen_extract_func_t);
extern void *gen_find_from_end
  (const void *, const list, gen_filter2_func_t, gen_extract_func_t);
extern void *gen_find_eq(const void *, const list);
extern void *gen_find_if(gen_filter_func_t, const list, gen_extract_func_t);
extern void *gen_find_if_from_end(gen_filter_func_t, list, gen_extract_func_t);

// DESTRUCTORS
extern void gen_free_list(list);
extern void gen_full_free_list(list);

extern list gen_last(const list);
extern void *gen_car(const list);

// OBSERVERS
extern size_t gen_length(const list);
extern size_t list_own_allocated_memory(const list);
extern gen_chunk gen_nth(int, const list);
extern list gen_nthcdr(int, const list);

// MAP
extern void gen_map(gen_iter_func_t, const list);
extern void gen_mapl(gen_iter_func_t, const list);

extern void * gen_reduce(void *, void *(*)(void*, const list), const list);

extern void gen_remove(list *, const void *);
extern void gen_remove_once(list *, const void *);
extern list gen_some(gen_filter_func_t, const list);
extern bool gen_replace_in_list(list, const void *, const void *);
extern void gen_exchange_in_list(list, const void *, const void *);

extern list gen_insert_list(list, const void *, list, bool);
extern void gen_insert_after(const void *, const void *, list);
extern list gen_insert_before(const void *, const void *, list);
extern list gen_once(const void *, list);
extern bool gen_in_list_p(const void *, const list);
extern int gen_occurences(const void *, const list);
extern bool gen_once_p(const list);
extern bool gen_equals(const list, const list, gen_eq_func_t);

// LIST MODIFIERS
extern void gen_sort_list(list, gen_cmp_func_t);
extern void gen_list_patch(list, const void *, const void *);

extern void gen_closure(list (*)(), const list);

extern list gen_copy_string_list(const list);
extern void gen_free_string_list(list);
void gen_fprint(FILE *, const string, const list, gen_string_func_t);

// UTILS
extern list gen_cons(const void *, const list);
extern list gen_bool_cons(bool, const list);
extern list gen_int_cons(_int, const list);
extern list gen_string_cons(string, const list);
extern list gen_list_cons(const list, const list); // list of lists
extern list gen_typed_cons(_int, const void *, const list);
extern list gen_CHUNK_cons(const gen_chunk *, const list);

extern void gen_list_and(list *, const list);
extern void gen_list_and_not(list *, const list);

extern int gen_position(const void *, const list);

extern bool gen_list_cyclic_p (const list ml);
extern list gen_list_head(list *, int);
void gen_substitute_chunk_by_list(list * pl, const void * o, list sl);

#endif /* newgen_list_included */
