void microcode(int i, int isi, int isj, int *image, int *kernel, int *new_image)
{
   int F_17;
   int F_16;
   int F_15;
   int I_22;
   int I_21;
   int F_14;
   int F_13;
   int F_12;
   int I_20;
   int F_11;
   int F_10;
   int F_9;
   int I_19;
   int I_18;
   int I_17;
   int I_14;
   int I_9;
   int I_5;
   int __scalar__0;
   int j;
   int i_1_0;
   int ki_0;
   int kj_0;
   I_17 = isj-2;
   for(i_1_0 = i; i_1_0 <= MIN(i+127, isi-2); i_1_0 += 1)
l99994:
      for(j = 1; j <= I_17; j += 1) {
         __scalar__0 = 0;
         I_18 = i_1_0-1;
         I_19 = isj*I_18;
         I_5 = j+I_19;
         F_9 = image[I_5-1]*kernel[0];
         __scalar__0 = __scalar__0+F_9;
         F_10 = image[I_5]*kernel[1];
         __scalar__0 = __scalar__0+F_10;
         F_11 = image[I_5+1]*kernel[2];
         __scalar__0 = __scalar__0+F_11;
         kj_0 = 3;
         I_20 = isj*i_1_0;
         I_9 = j+I_20;
         F_12 = image[I_9-1]*kernel[3];
         __scalar__0 = __scalar__0+F_12;
         F_13 = image[I_9]*kernel[4];
         __scalar__0 = __scalar__0+F_13;
         F_14 = image[I_9+1]*kernel[5];
         __scalar__0 = __scalar__0+F_14;
         kj_0 = 3;
         I_21 = i_1_0+1;
         I_22 = isj*I_21;
         I_14 = j+I_22;
         F_15 = image[I_14-1]*kernel[6];
         __scalar__0 = __scalar__0+F_15;
         F_16 = image[I_14]*kernel[7];
         __scalar__0 = __scalar__0+F_16;
         F_17 = image[I_14+1]*kernel[8];
         __scalar__0 = __scalar__0+F_17;
         kj_0 = 3;
         ki_0 = 3;
         __scalar__0 = __scalar__0/9;
         new_image[j+isj*i_1_0] = __scalar__0;
      }
   ;
}
