float ran1(idum)
int *idum;
{
        static long ix1,ix2,ix3;
        static float r[98];
        float temp;
        static int iff=0;
        int j;
        void nrerror();

        if (*idum < 0 || iff == 0) {
                iff=1;
                ix1=(54773 -(*idum)) % 259200;
                ix1=(7141*ix1+54773) % 259200;
                ix2=ix1 % 134456;
                ix1=(7141*ix1+54773) % 259200;
                ix3=ix1 % 243000;
                for (j=1;j<=97;j++) {
                        ix1=(7141*ix1+54773) % 259200;
                        ix2=(8121*ix2+28411) % 134456;
                        r[j]=(ix1+ix2*(1.0/134456))*(1.0/259200);
                }
                *idum=1;
        }
        ix1=(7141*ix1+54773) % 259200;
        ix2=(8121*ix2+28411) % 134456;
        ix3=(4561*ix3+51349) % 243000;
        j=1 + ((97*ix3)/243000);
        if (j > 97 || j < 1) nrerror("RAN1: This cannot happen.");
        temp=r[j];
        r[j]=(ix1+ix2*(1.0/134456))*(1.0/259200);
        return temp;
}
