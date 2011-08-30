short ening(int n, int img[n][n], int img_out[n-4][n])
{
   //PIPS generated variable
   int x, y;
   //PIPS generated variable
   int I_0, I_1;

l99998:
#pragma terapix
   for(I_0 = 0; I_0 <= (n-5)/124; I_0 += 1) {


l99989:
      for(I_1 = 0; I_1 <= (n-1)/260; I_1 += 1) {


         if (n-5>124*I_0+123)
            if (n-1>260*I_1+259)
l99996:
               for(y = 124*I_0; y <= 124*I_0+123; y += 1)
l99997:
                  for(x = 260*I_1; x <= 260*I_1+259; x += 1)
                     img_out[y][x] = MIN(MIN(MIN(MIN(img[y][x], img[y+1][x]), img[y+2][x]), img[y+3][x]), img[y+4][x]);
            else
l99994:
               for(y = 124*I_0; y <= 124*I_0+123; y += 1)
l99995:
                  for(x = 260*I_1; x <= n-1; x += 1)
                     img_out[y][x] = MIN(MIN(MIN(MIN(img[y][x], img[y+1][x]), img[y+2][x]), img[y+3][x]), img[y+4][x]);
         else if (n-1>260*I_1+259)
l99992:
            for(y = 124*I_0; y <= n-5; y += 1)
l99993:
               for(x = 260*I_1; x <= 260*I_1+259; x += 1)
                  img_out[y][x] = MIN(MIN(MIN(MIN(img[y][x], img[y+1][x]), img[y+2][x]), img[y+3][x]), img[y+4][x]);
         else
l99990:
            for(y = 124*I_0; y <= n-5; y += 1)
l99991:
               for(x = 260*I_1; x <= n-1; x += 1)
                  img_out[y][x] = MIN(MIN(MIN(MIN(img[y][x], img[y+1][x]), img[y+2][x]), img[y+3][x]), img[y+4][x]);
      }
   }
}

