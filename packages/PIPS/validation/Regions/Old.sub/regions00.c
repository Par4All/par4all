void runner_l99998(int n, int img[n][n], int img_out[n-4][n])
{
   //PIPS generated variable
   int x, y;
   //PIPS generated variable
   int I_0, I_1;

   for(I_0 = 2; I_0 <= (n-3)/124; I_0 += 1) {


      for(I_1 = 0; I_1 <= (n-1)/260; I_1 += 1) {


         if (n-3>124*I_0+123)
            if (n-1>260*I_1+259)
               for(y = 124*I_0; y <= 124*I_0+123; y += 1)
                   ////<<<<< regions over img are from accurate >>>>>>
                  for(x = 260*I_1; x <= 260*I_1+259; x += 1)
                     img_out[y-2][x] = MIN(MIN(MIN(MIN(img[y-2][x], img[y-1][x]), img[y][x]), img[y+1][x]), img[y+2][x]);
            else
               for(y = 124*I_0; y <= 124*I_0+123; y += 1)
                  for(x = 260*I_1; x <= n-1; x += 1)
                     img_out[y-2][x] = MIN(MIN(MIN(MIN(img[y-2][x], img[y-1][x]), img[y][x]), img[y+1][x]), img[y+2][x]);
         else if (n-1>260*I_1+259)
            for(y = 124*I_0; y <= n-3; y += 1)
               for(x = 260*I_1; x <= 260*I_1+259; x += 1)
                  img_out[y-2][x] = MIN(MIN(MIN(MIN(img[y-2][x], img[y-1][x]), img[y][x]), img[y+1][x]), img[y+2][x]);
         else
            for(y = 124*I_0; y <= n-3; y += 1)
               for(x = 260*I_1; x <= n-1; x += 1)
                  img_out[y-2][x] = MIN(MIN(MIN(MIN(img[y-2][x], img[y-1][x]), img[y][x]), img[y+1][x]), img[y+2][x]);
      }
   }
}

