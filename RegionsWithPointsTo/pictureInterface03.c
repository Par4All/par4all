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
//typedef TIFF *PictureHandler;


int buildRaster(const uint32 fact, PictureHandler tif, RasterType **raster)
{
  uint32 heigh = getLength(tif);
  uint32 width = getWitdth(tif);
  size_t npixels = width * heigh;

  uint32 *rawRaster = (uint32*) malloc(npixels * sizeof (uint32));
  
  uint32 f, i, j;
  for (f = 0; f < fact; f++) {
    for (i = 0; i < heigh; i++) {
      for (j = 0; j < width; j++) {
        PixelC r, g, b;
        (*raster)[f*(BYTES_PER_PIXEL*heigh*width) + E_R*(heigh*width) + i*width + j] = (RasterType) r;
        (*raster)[f*(BYTES_PER_PIXEL*heigh*width) + E_G*(heigh*width) + i*width + j] = (RasterType) g;
        (*raster)[f*(BYTES_PER_PIXEL*heigh*width) + E_B*(heigh*width) + i*width + j] = (RasterType) b;
      }
    }
   
    return 0;

  }
}
