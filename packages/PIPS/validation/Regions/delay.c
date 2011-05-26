/* delay.c - delay by D time samples */

void delay(int D, double w[1+D])                          /* \(w[0]\) = input, \(w[D]\) = output */
{
       int i;

       for (i=D; i>=1; i--)               /* reverse-order updating */
              w[i] = w[i-1];

}
