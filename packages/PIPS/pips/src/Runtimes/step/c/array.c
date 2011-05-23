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

static int array_maybe_expand (Array *array, const size_t len)
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



void array_append_vals (Array *array, const void *data, const size_t len)
{
  IN_TRACE("array = %p, data = %p, len = %d", array, data, len);

  array_maybe_expand(array, len);
  memcpy(array_sized_index(array, array->len), data, array->elt_size * len);
  array->len += len;

  OUT_TRACE("end");
}

/* Copy the last element of the array at the index location. This
   removes the element that was located at index location */

void array_remove_index_fast (Array *array, const size_t index)
{
  assert(index < array->len);

  if (index != array->len - 1)
    {
      void *last = array_sized_index(array, array->len - 1);
      
      memcpy(array_sized_index(array, index), last, array->elt_size);
    }
  array->len -= 1;
}


#ifdef TEST_ARRAY

#include <stdio.h>
#include <stdint.h>

int main()
{
  uint32_t i;
  Array myarray, myarray2;
  int c[10];

  printf("Creating myarray of type Array\n");
  array_set(&myarray,int);

  printf("data=%p len=%u alloc=%u elt_size=%u\n", myarray.data, (uint32_t)myarray.len, (uint32_t)myarray.alloc, (uint32_t)myarray.elt_size);

  printf("Allocate 20 int for myarray (still len == 0)\n");
  array_reset(&myarray, NULL, 20);
  printf("Try to add data in myarray without using array_append_vals\n");
  /*  for (i=0;i<myarray.len;i++) myarray.len still equals 0 */
  for (i=0;i<20;i++)
    {
      /* FSC semble remplir array.data */
      /* ne faudrait-il pas renommer array_index en array_data ou array_data_from_index?*/
      array_get_data_from_index(&myarray, int, i) = i;
      ((int *)myarray.data)[i] += 100;
    }
  /* FSC la boucle precedente n'a pas d'effet puisque array->len == 0. Est-ce fait expres? */

  printf("Printing myarray...\n");
  /*  for (i=0;i<myarray.len;i++) myarray.len still equals 0 */
  for (i=0;i<20;i++)
    printf("myarray[%d]=%d\n", i, array_get_data_from_index(&myarray, int, i));

  printf("FSC PROBLEME ICI: ON PEUT AJOUTER DES VALEURS ALORS QUE LEN == 0\n");
  printf("FSC EST-CE PERMIS?\n");
  printf("FSC SINON NE PAS GARDER CET EXEMPLE DANS LE TEST UNITAIRE\n");
  printf("myarray is still empty!\n");

  printf("Add additional data in myarray using array_append_vals\n");
  for (i=0; i<10;i++)
    {
      int v = 1000+i;
      array_append_vals(&myarray, &v, 1);
    }

  printf("Printing myarray...\n");
  for (i=0; i<myarray.len; i++)
    printf("myarray[%d]=%d\n", i, array_get_data_from_index(&myarray, int, i));

  printf("myarray is no more empty!\n");

  printf("Remove element at index 5\n");
  array_remove_index_fast(&myarray, 5);
  printf("Remove element at index 0\n");
  array_remove_index_fast(&myarray, 0);

  printf("Printing myarray...\n");
  for (i=0; i<myarray.len; i++)
    printf("myarray[%d]=%d\n", i, array_get_data_from_index(&myarray, int, i));

  printf("Creating a C array c...\n");
  for (i=0;i<10;i++)
    {
      c[i]=i;
      printf("c[%d]=%d\n",i,c[i]);
    }

  printf("Resetting myarray...\n");
  array_reset(&myarray, NULL, 0);
  printf("data=%p len=%u alloc=%u elt_size=%u\n", myarray.data, (uint32_t)myarray.len, (uint32_t)myarray.alloc, (uint32_t)myarray.elt_size);

  printf("Add 20 elements in myarray from c\n");
  array_append_vals(&myarray, c, 2);
  array_append_vals(&myarray, c, 4);
  array_append_vals(&myarray, c, 6);
  array_append_vals(&myarray, c, 8);
  printf("Printing myarray...\n");
  for (i=0; i<myarray.len; i++)
    printf("myarray[%d]=%d\n", i, array_get_data_from_index(&myarray, int, i));

  printf("Unset myarray...\n");
  array_unset(&myarray);

  printf("\nCreate a new Array myarray2\n");
  array_set(&myarray2, int);
  /* FSC A quoi ce printf puisque len == 0? */
  for (i=0;i<myarray2.len;i++)
    printf("myarray2[%d]=%d\n",i,array_get_data_from_index(&myarray2,int,i));

  printf("Initialize myarray2 with 5 elements from c\n");
  array_reset(&myarray2, c, 5);
  for (i=0;i<myarray2.len;i++)
    printf("myarray2[%d]=%d\n",i,array_get_data_from_index(&myarray2,int,i));
  printf("unset myarray2\n");
  array_unset(&myarray2);

  printf("\nCreate a new Array myarray2\n");
  array_set(&myarray2, int);
  printf("Initialize myarray2 with 5 elements from c\n");
  array_reset(&myarray2, c, 5);
  for (i=0;i<myarray2.len;i++)
    printf("myarray2[%d]=%d\n",i,array_get_data_from_index(&myarray2,int,i));
  printf("reset myarray2\n");
  array_reset(&myarray2, NULL, 0);
  for (i=0;i<myarray2.len;i++)
    printf("myarray2[%d]=%d\n",i,array_get_data_from_index(&myarray2,int,i));
  printf("unset myarray2\n");
  array_unset(&myarray2);

  return 0;
}
#endif
