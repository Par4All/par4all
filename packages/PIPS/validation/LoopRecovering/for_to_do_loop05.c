void convol(int isi, int isj, float new_image[isi][isj], float image[isi][isj], int ksi, int ksj, float kernel[ksi][ksj])
{
   int i;
   int j;
   int ki;
   int kj;

   for (i = 0;i<isi;i++)
      for (j = 0;j<isj;j++)
         new_image[i][j] = image[i][j];
}
