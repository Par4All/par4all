#include <stdlib.h>
#include "freia.h"
#include "geometrie.h"

int ligneHorizontale(freia_data2d *img,  int x, int y, int w, int coul)
{
  return freia_common_draw_line(img, x, y, x+w, y, coul);
}


int ligneVerticale(freia_data2d *img,  int x, int y, int w, int coul)
{
 return freia_common_draw_line(img, x, y, x, y+w, coul);
}


void echangerEntiers(int* x, int* y)
{
  int t = *x;
  *x = *y;
  *y = t;
}

int ligne(freia_data2d *img,   int x1, int y1, int x2, int y2, int coul)
{
  return freia_common_draw_line(img, x1, y1, x2, y2, coul);
}
 

int ligneInitTabEqt(uint32_t *lut, int width, int x1, int y1, int x2, int y2, int coul) {
  int d, dx, dy, aincr, bincr, xincr, yincr, x, y;

  for(d=0;d<width;d++) lut[d]=0;


  if (abs(x2 - x1) < abs(y2 - y1)) {
    /* parcours par l'axe vertical */

    if (y1 > y2) {
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
    /* parcours par l'axe horizontal */
    
    if (x1 > x2) {
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

int ligneEpaisse(freia_data2d *img,  int x1, int y1, int x2, int y2, int w, int coul) {
  int i1=y1-(w>>1);
  int i2=y2-(w>>1);
  int j;

  for(j=0;j<w;j++) freia_common_draw_line(img, x1,i1+j,x2,i2+j,coul);

  return 1;
}

int rectangle(freia_data2d *img,  int x, int y, int w, int h, int coul) {
  freia_common_draw_rect(img, x, y, x+w, y+w, coul);
 
  return 1;
}

int barre(freia_data2d *img,  int x, int y, int w, int h, int coul) {
  int i;
  int max=y+h;

  for(i=y;i<max;i++) ligneHorizontale(img,x, i, w, coul);

  return 1;
}

freia_status freia_auto_median(freia_data2d *imout, freia_data2d *imin, int32_t connexity )
{
  // Perform an automedian
  // not in-place safe

  freia_data2d *imtmp;
  freia_status ret;
  imtmp = freia_common_create_data(imin->bpp, imin->widthWa, imin->heightWa);
  ret =  freia_cipo_close(imtmp, imin, connexity, 1);
  ret |= freia_cipo_open(imtmp, imtmp, connexity, 1);
  ret |= freia_cipo_close(imtmp, imtmp,connexity, 1);
  ret |= freia_aipo_inf(imout, imtmp, imin);

  ret |= freia_cipo_open(imtmp, imin, connexity, 1);
  ret |= freia_cipo_close(imtmp, imtmp, connexity, 1);
  ret |= freia_cipo_open(imtmp, imtmp, connexity, 1);
  
  ret |= freia_aipo_sup(imout, imout, imtmp);
  
  ret |= freia_common_destruct_data(imtmp);
  return ret;
}

int getMaxInLigneEpaisse(freia_data2d *img, uint32_t *tab, int32_t x1,
  int32_t y1, int32_t x2, int32_t y2, int32_t w,  int32_t *coordx, int32_t *coordy) {
  int32_t j,i;
  int32_t max=0;

  int32_t cy, cx;

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

