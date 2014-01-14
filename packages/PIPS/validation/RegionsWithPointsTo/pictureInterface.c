/********************************************************************************
* (C) Copyright 20012 CAPS entreprise
*********************************************************************************
* \brief		: Toy application on picture convolutions.
* \author		: Laurent Morin
*******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <tiffio.h>
#include "pictureInterface.h"

#define RasterType short
#define	DISPFIELD_L(t, tag) { uint32 v; if (TIFFGetField(t, tag, &v)) fprintf(stdout, #tag ": %u\n", v); }
#define	DISPFIELD_S(t, tag) { uint16 v; if (TIFFGetField(t, tag, &v)) fprintf(stdout, #tag ": %hu\n", v); }
#define	DISPFIELD_F(t, tag) { float  v; if (TIFFGetField(t, tag, &v)) fprintf(stdout, #tag ": %g\n", v); }
#define	DISPFIELD_C(t, tag) { char  *v; if (TIFFGetField(t, tag, &v)) fprintf(stdout, #tag ": %s\n", v); }

#define	SAT(c)			(c < 0 ? 0 : c > 255 ? 255 : c)
typedef TIFF *PictureHandler;


int dumpRaster(const uint32 fact, const uint32 heigh, const uint32 width, PictureHandler tif, RasterType raster[fact][BYTES_PER_PIXEL][heigh][width])
{
  if (!tif || !raster) 
    {
      fprintf(stderr, "!tif || !raster\n");
      return -1;
    }
  size_t npixels = width * heigh;
  PixelC *rawRasterBuffer = (PixelC*) _TIFFmalloc(npixels * BYTES_PER_PIXEL * sizeof (PixelC));
  PixelC (*rawRaster)[BYTES_PER_PIXEL] = (PixelC (*)[BYTES_PER_PIXEL]) rawRasterBuffer;
  if (!rawRaster) 
    {
      fprintf(stderr, "!rawRaster\n");
      return -1;
    }
  uint32 f, i, j;
  for (f = 0; f < fact; f++) {
    for (i = 0; i < heigh; i++) {
      for (j = 0; j < width; j++) {
        RasterType rv = raster[f][E_R][i][j];
        RasterType gv = raster[f][E_G][i][j];
        RasterType bv = raster[f][E_B][i][j];
        PixelC r = (PixelC) SAT(rv);
        PixelC g = (PixelC) SAT(gv);
        PixelC b = (PixelC) SAT(bv);
        PixelC a = 0xFF;
        /* rawRaster[i*width + j] = (a << 24) + (b << 16) + (g << 8) + r; */

        if (!f) {
          rawRaster[i*width + j][E_R] = r;
          rawRaster[i*width + j][E_G] = g;
          rawRaster[i*width + j][E_B] = b;
          if (BYTES_PER_PIXEL > 3) {
            a = (PixelC) raster[f][E_A][i][j];
            rawRaster[i*width + j][E_A] = a;
          }
        }
        else {
          rawRaster[i*width + j][E_R] = (rawRaster[i*width + j][E_R] + r) / 2;
          rawRaster[i*width + j][E_G] = (rawRaster[i*width + j][E_G] + g) / 2;
          rawRaster[i*width + j][E_B] = (rawRaster[i*width + j][E_B] + b) / 2;
          if (BYTES_PER_PIXEL > 3) {
            RasterType av = raster[f][E_A][i][j];
            a = (PixelC) SAT(av);
            rawRaster[i*width + j][E_A] = (a + rawRaster[i*width + j][E_A]) / 2;
          }
        }
      }
    }
  }

  uint32 rowsperstrip = 0;
  TIFFGetField(tif, TIFFTAG_ROWSPERSTRIP, &rowsperstrip);


  size_t maxFullBlock = heigh / rowsperstrip;
  size_t blockSize =  rowsperstrip * width * BYTES_PER_PIXEL;
  size_t block;

  for (block=0 ; block < maxFullBlock; block++ ) {
    PixelC * raster_strip = (PixelC *) &rawRaster[(heigh - ((block+1) * rowsperstrip))*width+0];
    if( TIFFWriteEncodedStrip(tif, block, raster_strip, blockSize ) == -1 ) 
      {
	fprintf(stderr, " TIFFWriteEncodedStrip 1 failed\n");
	return -1;
      }
  }

  size_t lastBlockLoc =  maxFullBlock * rowsperstrip;
  if (heigh > lastBlockLoc) {
    PixelC * raster_strip = (PixelC *) &rawRaster[0];
    blockSize =  (heigh - lastBlockLoc) * width * BYTES_PER_PIXEL;
    if( TIFFWriteEncodedStrip(tif, maxFullBlock, raster_strip, blockSize ) == -1 ) 
      {
	fprintf(stderr, " TIFFWriteEncodedStrip 2 failed\n");
	return -1;
      }
  }

  _TIFFfree(rawRaster);

  return 0;
}

