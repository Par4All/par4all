#include "freia.h"

int freia_66(int32_t * kernel)
{
   int n = 10;
   freia_dataio fdin, fdout;
   freia_common_open_input(&fdin, 0);
   freia_common_open_output(&fdout, 1, 1024, 720, 16);

   freia_data2d *in, *t, *out;

   in  = freia_common_create_data(16, 1024, 720);
   t   = freia_common_create_data(16, 1024, 720);
   out = freia_common_create_data(16, 1024, 720);

   // input
   freia_common_rx_image(in, &fdin);

   freia_aipo_copy(out, in);
   freia_aipo_copy(t, in);

   while (n--)
   {
     freia_aipo_erode_8c(out, out, kernel);
     freia_aipo_sup(t, out, t);
     freia_aipo_dilate_8c(out, t, kernel);
     freia_aipo_sup(t, out, t);
   }

   // output
   freia_common_tx_image(out, &fdout);

   // cleanup
   freia_common_destruct_data(in);
   freia_common_destruct_data(t);
   freia_common_destruct_data(out);

   freia_common_close_input(&fdin);
   freia_common_close_output(&fdout);

   return 0;
}
