#include <string.h>
#include <stdio.h>
/*
  Based on Glib-2.22.2
*/

#define INLINE static inline
#include "array.h"
#include "trace.h"

void array_print(Array *array)
{
  IN_TRACE("array = %p", array);
  printf("data = %p, len = %d, alloc = %d, elt_size = %d\n", array->data, (int)array->len, (int)array->alloc, (int)array->elt_size);
  OUT_TRACE("end");
}

static int array_maybe_expand(Array *array, const size_t len)
{
  size_t want_alloc;
  size_t nearest_pow;
  int alloc_p;

  IN_TRACE("array =%p, len = %d", array, len);
  /* FSC pourquoi est-ce array->len + len et non un test sur array->len < len? */
  want_alloc = array->elt_size * (array->len + len);
  nearest_pow = 16;
  alloc_p = 0;

  if (want_alloc > array->alloc)
    {
      while (nearest_pow < want_alloc)
	nearest_pow <<= 1;
      array->data = realloc(array->data, nearest_pow);
      if (array->data == NULL)
	{
	  perror("problem when reallocating array->data\n");
	  exit(EXIT_FAILURE);
	}

      array->alloc = nearest_pow;
      assert(array->data);
      alloc_p = 1;
    }

  OUT_TRACE("array =%p alloc_p = %d", array, alloc_p);
  return alloc_p;
}



void array_reset(Array *array, const void *data, const size_t len)
{
  IN_TRACE("array = %p, data = %p, len =%d", array, data, len);

  array->len = 0;
  if (data)
    array_append_vals(array, data, len);
  else
    array_maybe_expand(array, len);

  OUT_TRACE("end");
}



void array_append_vals(Array *array, const void *data, const size_t len)
{
  IN_TRACE("array = %p, data = %p, len = %d", array, data, len);

  array_maybe_expand(array, len);
  memcpy(array_sized_index(array, array->len), data, array->elt_size * len);
  array->len += len;

  OUT_TRACE("end");
}

/* Copy the last element of the array at the index location. This
   removes the element that was located at index location */

void array_remove_index_fast(Array *array, const size_t index)
{
  IN_TRACE("array = %p, index = %d", array, index);

  assert(index < array->len);

  if (index != array->len - 1)
    {
      void *last = array_sized_index(array, array->len - 1);

      memcpy(array_sized_index(array, index), last, array->elt_size);
    }
  array->len -= 1;

  OUT_TRACE("end");
}
