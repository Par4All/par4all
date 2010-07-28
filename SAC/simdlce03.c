void convol(int isi, int isj, float new_image[isi][isj], float image[isi][isj], float kernel[3][3])
{
   //     The convolution kernel is not applied on the outer part
   //     of the image
   
   int i, j, ki, kj;
   //PIPS generated variable
   float RED0[3], F_0, F_1, F_2, F_3, F_4, F_40, F_30, F_20, F_10, F_00;
   //PIPS:SAC generated variable
   float PADDING_VALUE;
   
   
   for(i = 0; i <= isi-1; i += 1)
      for(j = 0; j <= isj-1; j += 1)
         new_image[i][j] = image[i][j];
   
   
l400:
   for(i = 1; i <= isi-2; i += 1)
l300:
      for(j = 1; j <= isj-2; j += 1) {
         new_image[i][j] = 0.;
         RED0[0] = 0.000000;
         RED0[1] = 0.000000;
         RED0[2] = 0.000000;
l200:
         for(ki = 0; ki <= 2; ki += 1) {
            //PIPS:SAC generated float vector(s)
            float v4sf_vec0[4], v4sf_vec1[4], v4sf_vec2[4];
            //PIPS:SAC generated variable
            float aligned[4] = {0, 0, 0, 0};
            //PIPS:SAC generated float vector(s)
            float v4sf_vec3[4], v4sf_vec4[4];
l100:            ;
            SIMD_LOAD_V4SF(v4sf_vec2, &kernel[ki][0]);
            SIMD_LOAD_V4SF(v4sf_vec1, &image[i+ki-1][j-1]);
            SIMD_MULPS(v4sf_vec0, v4sf_vec1, v4sf_vec2);
            SIMD_STORE_V4SF(v4sf_vec0, &aligned[0]);
            SIMD_LOAD_V4SF(v4sf_vec3, &RED0[0]);
            SIMD_ADDPS(v4sf_vec3, v4sf_vec3, v4sf_vec0);
            SIMD_STORE_MASKED_V4SF(v4sf_vec3, &RED0[0]);
            kj = 3;
         }
         F_30 = RED0[0]+new_image[i][j];
         F_40 = RED0[1]+F_30;
         new_image[i][j] = RED0[2]+F_40;
         new_image[i][j] = new_image[i][j]/9;
      }
   ;
}
int main()
{
    int isi=10,isj=10;
    float kernel[3][3],in[isi][isj],out[isi][isj];
    convol(isi,isj,in,out,kernel);
}
