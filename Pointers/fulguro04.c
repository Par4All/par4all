/* Excerpt from fulguro03 */

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
   fgUINT16 *array_d = (fgUINT16 *) (dat->array)[row];
   return;
}
