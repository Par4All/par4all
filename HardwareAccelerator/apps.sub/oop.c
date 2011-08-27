#include <stdio.h>
#include "freia.h"

/********************************************************************* UTILS */
void echangerEntiers(int* x, int* y)
{
  int t = *x;
  *x = *y;
  *y = t;
}

int ligneInitTabEqt
(uint32_t *lut, int width, int x1, int y1, int x2, int y2, int coul)
{
  int d, dx, dy, aincr, bincr, xincr, yincr, x, y;

  for(d=0;d<width;d++) lut[d]=0;

  if (abs(x2 - x1) < abs(y2 - y1))
  {
    if (y1 > y2)
    {
      echangerEntiers(&x1, &x2);
      echangerEntiers(&y1, &y2);
    }

    xincr = x2 > x1 ? 1 : -1;
    dy = y2 - y1;
    dx = abs(x2 - x1);
    d = 2 * dx - dy;
    aincr = 2 * (dx - dy);
    bincr = 2 * dx;
    x = x1;
    y = y1;

    lut[x]=y;

    for (y = y1+1; y <= y2; ++y) {
      if (d >= 0) {
        x += xincr;
        d += aincr;
      } else
        d += bincr;
      lut[x]=y;
    }

  } else {

    if (x1 > x2)
    {
      echangerEntiers(&x1, &x2);
      echangerEntiers(&y1, &y2);
    }

    yincr = y2 > y1 ? 1 : -1;
    dx = x2 - x1;
    dy = abs(y2 - y1);
    d = 2 * dy - dx;
    aincr = 2 * (dy - dx);
    bincr = 2 * dy;
    x = x1;
    y = y1;
    lut[x]=y;

    for (x = x1+1; x <= x2; ++x) {
      if (d >= 0) {
        y += yincr;
        d += aincr;
      } else
        d += bincr;
      lut[x]=y;
    }
  }

  return 1;
}

int ligneHorizontale(freia_data2d *img,  int x, int y, int w, int coul)
{
  return freia_common_draw_line(img, x, y, x+w, y, coul);
}

int barre(freia_data2d *img,  int x, int y, int w, int h, int coul)
{
  int i, max=y+h;
  for(i=y;i<max;i++)
    ligneHorizontale(img,x, i, w, coul);
  return 1;
}

int getMaxInLigneEpaisse(
  const freia_data2d *img, uint32_t *tab, int32_t x1,
  int32_t y1, int32_t x2, int32_t y2, int32_t w,
  int32_t *coordx, int32_t *coordy)
{
  int32_t j,i, max=0, cy, cx;

  *coordx=x2;
  *coordy=y2;

  for(j=-w/2;j<w/2;j++) {

    for(i=x1;i<x2;i++) {
      cy=tab[i]+j;
      cx=i;
      if((freia_common_get(img,cx,cy)==255) && (cx<*coordx)) {
        max=255;
        *coordx=cx;
        *coordy=cy;
      }
    }
  }

  return max;
}


/***************************************************************** AIPO CODE */

int main(int argc, char *argv[])
{
  freia_dataio fdin, fdout, fdout1;

  freia_data2d *i0, *i1, *i2, *i3, *im, *t0, *t1, *t2, *t3, *t4;
  freia_data2d *imgtmp;
  freia_data2d *imgg1, *imgg2, *imgsav;

  register unsigned int nbpix;
  register int32_t x, y;
  int32_t valx, valy;
  register int32_t valmax, valx_pre = 0;

  register int32_t startx = 33, starty = 272, endx = 152, endy = 132;
  register int32_t countfalsedetect = 0;
  register int32_t epaisseur = 17;
  register int32_t mx1, my1, mx2, my2, idx = 10;

  register int32_t dx;
  uint32_t *tabeqt;
  register freia_status end = 0;
  freia_data2d *imtmp_0, *imtmp_1, *imtmp_2, *imtmp_3;

  freia_initialize(argc, argv);

  freia_common_open_input(&fdin, 0);

  i0 = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);

  freia_common_open_output(&fdout, 0, fdin.framewidth, fdin.frameheight, 8);
  freia_common_open_output(&fdout1, 1, fdin.framewidth, fdin.frameheight, 8);
  i1 = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);
  i2 = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);
  i3 = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);
  im = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);

  t0 = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);
  t1 = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);
  t2 = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);
  t3 = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);
  t4 = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);

  imgg1 = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);
  imgg2 = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);

  imgtmp = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);

   tabeqt = freia_common_alloc(fdin.framewidth*sizeof(*tabeqt));
   x = fdin.framewidth;
   y = fdin.frameheight;
   dx = 10;

   nbpix = y*x;

   ligneInitTabEqt(tabeqt, fdin.framewidth, 33, 272, 152, 132, 255);

   mx1 = 33;
   my1 = 272;
   mx2 = 152;
   my2 = 132;
   valx = 152;
   end = freia_common_rx_image(i0, &fdin);

   while (end==FREIA_OK)
   {
     // ??? POINTER SHUFFLING is not supported by pips freia compiler!
     imgsav = t4;
     t4 = t3;
     t3 = t2;
     t2 = t1;
     t1 = t0;
     t0 = imgsav;

     freia_aipo_copy(i1, i0);
     freia_aipo_copy(t0, i1);

     imtmp_3 = freia_common_create_data(imgtmp->bpp, imgtmp->widthWa, imgtmp->heightWa);
     freia_aipo_dilate_8c(imtmp_3, t0, freia_morpho_kernel_8c);
     freia_aipo_erode_8c(imgtmp, t0, freia_morpho_kernel_8c);
     freia_aipo_sub(imgtmp, imtmp_3, imgtmp);
     freia_common_destruct_data(imtmp_3);

     imtmp_2 = freia_common_create_data(i1->bpp, i1->widthWa, i1->heightWa);
     freia_aipo_dilate_8c(imtmp_2, t2, freia_morpho_kernel_8c);
     freia_aipo_erode_8c(i1, t2, freia_morpho_kernel_8c);
     freia_aipo_sub(i1, imtmp_2, i1);
     freia_common_destruct_data(imtmp_2);

     freia_aipo_absdiff(i1, imgtmp, i1);
     freia_aipo_erode_8c(i1, i1, freia_morpho_kernel_8c);
     freia_aipo_dilate_8c(i1, i1, freia_morpho_kernel_8c);

     imtmp_1 = freia_common_create_data(imgtmp->bpp, imgtmp->widthWa, imgtmp->heightWa);
     freia_aipo_dilate_8c(imtmp_1, t2, freia_morpho_kernel_8c);
     freia_aipo_erode_8c(imgtmp, t2, freia_morpho_kernel_8c);
     freia_aipo_sub(imgtmp, imtmp_1, imgtmp);

     freia_common_destruct_data(imtmp_1);

     imtmp_0 = freia_common_create_data(i2->bpp, i2->widthWa, i2->heightWa);
     freia_aipo_dilate_8c(imtmp_0, t4, freia_morpho_kernel_8c);
     freia_aipo_erode_8c(i2, t4, freia_morpho_kernel_8c);
     freia_aipo_sub(i2, imtmp_0, i2);
     freia_common_destruct_data(imtmp_0);

     freia_aipo_absdiff(i2, imgtmp, i2);
     freia_aipo_erode_8c(i2, i2, freia_morpho_kernel_8c);
     freia_aipo_dilate_8c(i2, i2, freia_morpho_kernel_8c);
     freia_aipo_inf(i3, i2, i1);

     freia_aipo_threshold(i3, i3, 15, 255, 1);
     valx_pre = valx;
     valmax = getMaxInLigneEpaisse(i3, tabeqt, mx1, my1, mx2, my2, 17, &valx, &valy);
     dx = 10;
     if (valmax==0) {
       countfalsedetect++;
       valx = valx_pre;
       if (countfalsedetect>45) {
         countfalsedetect = 1000;
         //valx=(valx_pre+5)%mx2;
         valx += 5;
         if (valx>mx2)
           valx = mx2;
         if (valx<mx1)
           valx = mx1;
         dx = 10;
       }
     }
     else {
       countfalsedetect = 0;
       dx = 20;
     }
     barre(t0, valx, 0, 2, x, 255);

     mx1 = valx-dx;
     if (mx1<33)
       mx1 = 33;
     my1 = tabeqt[mx1];

     mx2 = dx+valx;
     if (mx2>152)
       mx2 = 152;
     my2 = tabeqt[mx2];

     valx_pre = -valx+152;

     freia_common_tx_image(t0, &fdout);
     end = freia_common_rx_image(i0, &fdin);
   }

   freia_common_destruct_data(i0);
   freia_common_destruct_data(i1);
   freia_common_destruct_data(i2);
   freia_common_destruct_data(i3);
   freia_common_destruct_data(im);
   freia_common_destruct_data(t0);
   freia_common_destruct_data(t1);
   freia_common_destruct_data(t2);
   freia_common_destruct_data(t3);
   freia_common_destruct_data(t4);
   freia_common_destruct_data(imgg1);
   freia_common_destruct_data(imgg2);
   freia_common_destruct_data(imgtmp);

   freia_common_close_input(&fdin);
   freia_common_close_output(&fdout);

   freia_shutdown();
   return 0;
}
