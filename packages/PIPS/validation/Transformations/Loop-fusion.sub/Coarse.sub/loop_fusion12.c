


int main() {
   // Same as loop_fusion07.c, but with more than one iteration in the outer 
   // loop

   double A[2][10000];
   {
      int lv1, lv2;
      for(lv1 = 0; lv1 <= 1; lv1 += 1)
         for(lv2 = 0; lv2 <= 9999; lv2 += 1)
            A[lv1][lv2] = (double) 1.0;
   }
   double B[2][10000];
   {
      int lv1, lv2;
      for(lv1 = 0; lv1 <= 1; lv1 += 1)
         for(lv2 = 0; lv2 <= 9999; lv2 += 1)
            B[lv1][lv2] = (double) 1.0;
   }
   double C[2][10000];
   {
      int lv1, lv2;
      for(lv1 = 0; lv1 <= 1; lv1 += 1)
         for(lv2 = 0; lv2 <= 9999; lv2 += 1)
            C[lv1][lv2] = A[lv1][lv2]+B[lv1][lv2];
   }

}
