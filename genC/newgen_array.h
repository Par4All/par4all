/*
 * $Id$
 *
 * $Log: newgen_array.h,v $
 * Revision 1.8  1998/12/29 16:03:53  coelho
 * fixed type in map.
 *
 * Revision 1.7  1998/12/29 16:01:43  coelho
 * *** empty log message ***
 *
 * Revision 1.6  1998/12/29 16:00:21  coelho
 * char * -> void *
 *
 * Revision 1.5  1997/12/05 12:18:32  coelho
 * list <-> array
 *
 * Revision 1.4  1997/12/04 17:24:14  coelho
 * GEN_ARRAY_MAP added.
 *
 */

struct _gen_array_chunk_t;
typedef struct _gen_array_chunk_t * gen_array_t;

#define gen_array_undefined ((gen_array_t) -12)
#define gen_array_undefined_p(a) ((a)==gen_array_undefined)

/* declarations...
 */
gen_array_t gen_array_make(int);
void gen_array_free(gen_array_t);
void gen_array_full_free(gen_array_t);
void gen_array_addto(gen_array_t, int, void *);
void gen_array_dupaddto(gen_array_t, int, void *);
void gen_array_append(gen_array_t, void *);
void gen_array_dupappend(gen_array_t, void *);
void ** gen_array_pointer(gen_array_t);
int gen_array_nitems(gen_array_t);
int gen_array_size(gen_array_t);
void * gen_array_item(gen_array_t, int);
void gen_array_sort(gen_array_t);
gen_array_t gen_array_from_list(list);
list list_from_gen_array(gen_array_t);

#define GEN_ARRAY_MAP(s, code, array)			\
  {							\
      int _i, _nitems = gen_array_nitems(array);	\
      for(_i=0; _i<_nitems; _i++)			\
      {							\
	  void * s = gen_array_item(array, _i);		\
	  code;						\
      }							\
  }
