typedef unsigned short fgUINT16 ;

typedef struct {
    void **array;
}FLGR_Data2D;

typedef struct {
    int spp; 
    int bps; 
    void *array;
} FLGR_Vector;

void flgr2d_set_data_vector_fgUINT16(FLGR_Data2D *dat, int row, int col, FLGR_Vector *vct)
{
   fgUINT16 *array_s = (fgUINT16 *) vct->array;
   fgUINT16 *array_d = (fgUINT16 *) (dat->array)[row];
   int I_0 = vct->spp;
   {
      register fgUINT16 val;
      register int i;
      register int k;
      unsigned short _return0;
      for ((k = 0, i = col*I_0);k<I_0;(k++, i++)) {
         _return0 = array_s[k];
         val = _return0;
         array_d[i] = val;
      }
   }
}
