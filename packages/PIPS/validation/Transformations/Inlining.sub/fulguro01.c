// excerpt from fulguro that makes unfolding fail
// because of a conflict between vector_array and an anywhere location.
// see ticket #549

int flgr_get_array_fgUINT16(int* array, int pos)
{
  return array[pos];
}

void flgr_set_array_fgUINT16(int* array, int pos, int value)
{
  array[pos]=value;
}

void flgr_get_data_array_vector_fgUINT16(int *vector_array, int *data_array, int spp, int pos)
{
   register int val;
   register int i, k;
   i = pos*spp;

   for(k = 0; k <= spp-1; k += 1) {
      val = flgr_get_array_fgUINT16(data_array, i);
      flgr_set_array_fgUINT16(vector_array, k, val);
      i++;
   }
}
