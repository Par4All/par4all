#ifndef __GEOMETRIE_H
#define __GEOMETRIE_H

// stuff with freia images
int ligneHorizontale(freia_data2d *img, int x, int y, int w, int coul);
int ligneVerticale(freia_data2d *img, int x, int y, int w, int coul);
int ligne(freia_data2d *img ,int x1, int y1, int x2, int y2, int coul);
int ligneEpaisse(freia_data2d *img, int x1, int y1, int x2, int y2, int w, int coul);
int rectangle(freia_data2d *img, int x, int y, int w, int h, int coul);
int barre(freia_data2d *img, int x, int y, int w, int h, int coul);
int cercle(freia_data2d *img, int cx, int cy, int rayon, int coul);
int disque(freia_data2d *img, int cx, int cy, int rayon, int coul);
freia_status freia_auto_median(freia_data2d *, freia_data2d *, int32_t);
int getMaxInLigneEpaisse(freia_data2d *, uint32_t *, int32_t,
int32_t, int32_t, int32_t, int32_t,  int32_t *, int32_t *);

// other stuff
int ligneInitTabEqt(uint32_t *, int, int, int, int, int, int);

#endif


