/********************************************************************************
* (C) Copyright 20012 CAPS entreprise
*********************************************************************************
* \brief		: Toy application on picture convolutions.
* \author		: Laurent Morin
*******************************************************************************/

#ifndef  PICTUREINTERFACE_G
#define  PICTUREINTERFACE_G

#include <tiffio.h>

// -- Parameters --
#define STEP
#ifdef STEP
/* les typedef ne sont pas supportes comme type de donner a echanger */
#define RasterType short
#else
typedef short RasterType;			/*!< Type used as a promotion for the convolution. */
#endif
static const int BYTES_PER_PIXEL = 3;		/*!< Number of channels, either 3 or 4 (alpha layer). */

// -- Picture Interface --
typedef TIFF *PictureHandler;
typedef unsigned char PixelC;

extern int setExpansionFactor(int factor);

extern int loadImage(const char *filename, PictureHandler *tif);
extern int describeImage(PictureHandler tif);
extern int closeImage(PictureHandler tif);
extern int openImage(const char *filename, PictureHandler *tif);
extern int initNewImage(const char *filename, uint32 w, uint32 h, PictureHandler tif);
extern int newImage(const char *filename, uint32 w, uint32 h, PictureHandler *tif);
extern uint32 getWitdth(PictureHandler tif);
extern uint32 getLength(PictureHandler tif);
extern int newRaster(const uint32 fact, const uint32 heigh, const uint32 width, RasterType **raster);
extern int pushImage(PictureHandler tif);
extern int pushAndSetImage(PictureHandler tif);
extern int buildRaster(const uint32 fact, PictureHandler tif, RasterType **raster);
extern int dumpRaster(uint32 fact, uint32 heigh, uint32 width, PictureHandler tif, RasterType raster[fact][BYTES_PER_PIXEL][heigh][width]);

extern int closeRaster(RasterType *raster);
extern void error(const char *msg);

enum {
  E_R = 0,
  E_G,
  E_B,
  E_A,
  E_SIZE
};

#endif
