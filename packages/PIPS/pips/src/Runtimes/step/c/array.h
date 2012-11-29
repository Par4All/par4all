/*
  Based on Glib-2.22.2
*/

#ifndef __ARRAY_H__
#define __ARRAY_H__

#include <stdlib.h>
#include <assert.h>

#ifndef INLINE
#define INLINE extern inline
#endif

typedef struct
{
  void *data;
  size_t len;         // nb element
  size_t alloc;       // size in bytes
  size_t elt_size;
}Array;

#define array_ready_p(array) ((array)->elt_size!=0)
#define array_set(array,type) array_sized_set(array, sizeof(type));
INLINE void array_sized_set(Array *array, const size_t elt_size)
{array->elt_size = elt_size;array->data = NULL;array->len = 0;array->alloc = 0;}
extern void array_reset(Array *array, const void *data, const size_t len);
INLINE void array_unset(Array *array)
{free(array->data);array->elt_size = 0;array->data = NULL;array->len = 0;array->alloc = 0;}

#define array_get_data_from_index(array,type,index) (*((type*) (void*) array_sized_index(array, (size_t)index)))
INLINE void* array_sized_index(Array *array, const size_t index)
{return (void*)((char*)array->data + (array->elt_size * index));}
extern void array_append_vals(Array *array, const void *data, const size_t len);
extern void array_remove_index_fast(Array *array, const size_t index);
extern void array_print(Array *array);
#endif /* __ARRAY_H__ */
