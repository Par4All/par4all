/* $Id$
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
void gen_array_addto(gen_array_t, int, char *);
void gen_array_dupaddto(gen_array_t, int, char *);
char **gen_array_pointer(gen_array_t);
int gen_array_nitems(gen_array_t);
int gen_array_size(gen_array_t);
char * gen_array_item(gen_array_t, int);
void gen_array_sort(gen_array_t);
