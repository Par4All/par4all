// Piece of FREIA application, videosurvey

#include <stdio.h>
#include "inlining01.h"

int main(int argc, char *argv[])
{
  freia_dataio fdin;
  freia_dataio fdstab;
  freia_dataio fdehn;
  freia_dataio fdmotion;

  freia_data2d *imcurrent;
  freia_data2d *imprevious;
  freia_data2d *imstab;
  freia_data2d *imtmp1;
  freia_data2d *imtmp2;
  freia_data2d *imehn;

  freia_data2d *imtmp3;
  freia_data2d *imbg16;
  freia_data2d *imbg;
  freia_data2d *immotion;

  int32_t *shiftarray;

  int32_t maxmotion;
  int32_t minmotion;
  int32_t binvalue = 128;

  int32_t xshift, yshift;
  int32_t xshiftacc, yshiftacc;

  freia_status end = FREIA_OK;

  // Param set by user
  _Bool stabilize = 1;
  _Bool contrastenhance = 1;
  _Bool motiondetect = 1;

  uint32_t horizon = 10;
  int32_t maximal_shape = 21;
  int32_t minimal_contrast = 50;
  int32_t motion_a = 10;
  int32_t motion_b = 100-motion_a;
  int32_t motion_th = 30;
  int32_t motion_trig = 75;
  //PIPS generated variable
  freia_status _return0, _return1, _return2, _return3, _return4, _return5, _return6, _return7, _return8, _return9, _return10, _return11, _return12, _return13, _return14, _return15, _return16, _return17;

  freia_initialize(argc, argv);
  // Init an alloc
  freia_common_open_input(&fdin, 0);

  imcurrent = freia_common_create_data(16, fdin.framewidth, fdin.frameheight);
  imprevious = freia_common_create_data(16, fdin.framewidth, fdin.frameheight);
  imtmp1 = freia_common_create_data(16, fdin.framewidth, fdin.frameheight);

  imstab = freia_common_create_data(16, fdin.framewidth-2*horizon, fdin.frameheight-2*horizon);
  imtmp2 = freia_common_create_data(16, imstab->width, imstab->height);

  imtmp3 = freia_common_create_data(16, imstab->width, imstab->height);
  imehn = freia_common_create_data(16, imstab->width, imstab->height);
  imbg16 = freia_common_create_data(16, imstab->width, imstab->height);
  imbg = freia_common_create_data(16, imstab->width, imstab->height);
  immotion = freia_common_create_data(16, imstab->width, imstab->height);

  freia_common_open_output(&fdstab, 0, imstab->width, imstab->height, imstab->bpp);
  freia_common_open_output(&fdehn, 1, imstab->width, imstab->height, imstab->bpp);
  freia_common_open_output(&fdmotion, 2, imstab->width, imstab->height, imstab->bpp);
  shiftarray = (int32_t *) freia_common_alloc(sizeof(uint32_t)*(horizon*2+1)*(horizon*2+1)*3);

  freia_aipo_set_constant(imbg16, 0);
  freia_aipo_xor(imtmp1, imtmp1, imtmp1);

  // application start
  xshiftacc = 0;
  yshiftacc = 0;


  end = freia_common_rx_image(imcurrent, &fdin);

 l99994: ;
  if (end==FREIA_OK) {
  }
  else {
    goto main_break_1;
  }

  // Stabilization
  freia_aipo_copy(imprevious, imcurrent);
  end = freia_common_rx_image(imcurrent, &fdin);
  if (end!=FREIA_OK)
    goto main_break_1;

  if (stabilize) {
    // save previous frame

    // detect shift
    ;
    {
      //PIPS generated variable
      int32_t *xshift0 = &xshift, *yshift0 = &yshift;
      //PIPS generated variable
      freia_data2d *imin0 = imcurrent, *imref0 = imprevious, *imtmp0 = imtmp1;
      //PIPS generated variable
      int32_t *shiftarray0 = shiftarray;
      //PIPS generated variable
      uint32_t blocksize0 = 64, step0 = 64, horizon0 = horizon;
      {

	int32_t tu = blocksize0-2*horizon0;
	uint32_t i, j, bestshift;
	uint32_t xmin, ymin, nbelt;
	int32_t min;

	//init shift array
	nbelt = 0;

	// process each block of a frame
	for(i = 0; i <= imin0->height-step0-1; i += step0)
	  for(j = 0; j <= imin0->width-step0-1; j += step0) {

	    //set imin working area to block(i/step,j/step)
	    freia_common_set_wa(imin0, j+horizon0, i+horizon0, tu, tu);

	    //set imref working area to block(i/step,j/step)
	    freia_common_set_wa(imref0, j, i, blocksize0, blocksize0);

	    //set imtmp working area to block(i/step,j/step)
	    freia_common_set_wa(imtmp0, j+horizon0, i+horizon0, 2*horizon0+1, 2*horizon0+1);

	    //compute correlation by SAD
	    ;
	    {
	      //PIPS generated variable
	      freia_data2d *imout0 = imtmp0, *imin1 = imin0, *imref1 = imref0;
	      //PIPS generated variable
	      uint32_t horizon1 = horizon0;
	      {
		uint32_t m, n;
		uint32_t sav_xStartWa, sav_yStartWa;
		uint32_t sav_widthWa, sav_heightWa;
		uint32_t sad;
		uint32_t wsize;
		if (!(1&&1&&1)) {
		  fprintf(stderr, "ERROR: file %s, line %d, function %s: ""Null pointers\n", "../../api/cipo/src/freiaComplexOpLinear.c", 28, __FUNCTION__);
		  while (0)
		    ;
		  _return9 = FREIA_INVALID_PARAM;
		  goto l99989;
		}
		if (1!=1) {

		  fprintf(stderr, "ERROR: file %s, line %d, function %s: ""bpp of images are not compatibles\n", "../../api/cipo/src/freiaComplexOpLinear.c", 33, __FUNCTION__);
		  while (0)
		    ;
		  _return9 = FREIA_SIZE_ERROR;
		  goto l99989;
		}
		if (1!=1) {

		  fprintf(stderr, "ERROR: file %s, line %d, function %s: ""bpp of images are not compatibles\n", "../../api/cipo/src/freiaComplexOpLinear.c", 38, __FUNCTION__);
		  while (0)
		    ;
		  _return9 = FREIA_SIZE_ERROR;
		  goto l99989;
		}
		if (1!=1) {

		  fprintf(stderr, "ERROR: file %s, line %d, function %s: ""working areas of images are not compatibles\n", "../../api/cipo/src/freiaComplexOpLinear.c", 43, __FUNCTION__);
		  while (0)
		    ;
		  _return9 = FREIA_SIZE_ERROR;
		  goto l99989;
		}
		if (1!=1) {

		  fprintf(stderr, "ERROR: file %s, line %d, function %s: ""working areas of images are not compatibles\n", "../../api/cipo/src/freiaComplexOpLinear.c", 48, __FUNCTION__);
		  while (0)
		    ;
		  _return9 = FREIA_SIZE_ERROR;
		  goto l99989;
		}
		if (1!=1) {

		  fprintf(stderr, "ERROR: file %s, line %d, function %s: ""image output width must be (2*horizon+1)\n", "../../api/cipo/src/freiaComplexOpLinear.c", 53, __FUNCTION__);
		  while (0)
		    ;
		  _return9 = FREIA_SIZE_ERROR;
		  goto l99989;
		}
		if (1!=1) {

		  fprintf(stderr, "ERROR: file %s, line %d, function %s: ""image output height must be (2*horizon+1)\n", "../../api/cipo/src/freiaComplexOpLinear.c", 58, __FUNCTION__);
		  while (0)
		    ;
		  _return9 = FREIA_SIZE_ERROR;
		  goto l99989;
		}


		wsize = imin1->heightWa*imin1->widthWa;

		freia_common_get_wa(imref1, &sav_xStartWa, &sav_yStartWa, &sav_widthWa, &sav_heightWa);

		for(n = 0; n <= 2*horizon1+1-1; n += 1)
		  for(m = 0; m <= 2*horizon1+1-1; m += 1) {
		    freia_common_set_wa(imref1, sav_xStartWa+m, sav_yStartWa+n, imin1->widthWa, imin1->heightWa);
		    {
		      //PIPS generated variable
		      freia_data2d *imin10 = imin1, *imin20 = imref1;
		      //PIPS generated variable
		      uint32_t *sad0 = &sad;
		      {
			int32_t vol;
			freia_status ret;
			freia_data2d *imtmp;
			if (1!=1) {

			  fprintf(stderr, "ERROR: file %s, line %d, function %s: ""working areas of images are not compatibles\n", "../../api/cipo/src/freiaComplexOpMeasure.c", 20, __FUNCTION__);
			  while (0)
			    ;
			  _return15 = FREIA_SIZE_ERROR;
			  goto l99983;
			}
			if (1!=1) {

			  fprintf(stderr, "ERROR: file %s, line %d, function %s: ""bpp of images are not compatibles\n", "../../api/cipo/src/freiaComplexOpMeasure.c", 25, __FUNCTION__);
			  while (0)
			    ;
			  _return15 = FREIA_SIZE_ERROR;
			  goto l99983;
			}

			imtmp = freia_common_create_data(imin10->bpp, imin10->widthWa, imin10->heightWa);

			ret = freia_aipo_absdiff(imtmp, imin10, imin20);

			ret |= freia_aipo_global_vol(imtmp, &vol);

			ret |= freia_common_destruct_data(imtmp);

			*sad0 = (uint32_t) vol;


			_return15 = ret;
			goto l99983;
		      l99983: ;
		      }
		    }
		    _return15;
		    sad = sad/wsize;
		    freia_common_set(imout0, m, n, sad);
		  }

		freia_common_set_wa(imref1, sav_xStartWa, sav_yStartWa, sav_widthWa, sav_heightWa);


		_return9 = FREIA_OK;
		goto l99989;
	      l99989: ;
	      }
	    }
	    _return9;

	    //get min coord
	    freia_aipo_global_min_coord(imtmp0, &min, &xmin, &ymin);
	    addShiftToArray(shiftarray0, &nbelt, horizon0-((int32_t) xmin), horizon0-((int32_t) ymin));
	  }

	//choose best shift
	bestshift = 1;
	for(i = 0; i <= nbelt-1; i += 1)
	  if (bestshift<=shiftarray0[i*3+2]) {
	    bestshift = shiftarray0[i*3+2];
	    *xshift0 = shiftarray0[i*3];
	    *yshift0 = shiftarray0[i*3+1];
	  }

	freia_common_reset_wa(imin0);
	freia_common_reset_wa(imref0);
	freia_common_reset_wa(imtmp0);


	_return0 = FREIA_OK;
	goto l99999;
      l99999: ;
      }
    }
    _return0;

    // accumulate shift
    xshiftacc += xshift;
    yshiftacc += yshift;

    xshiftacc = xshiftacc>horizon?horizon:xshiftacc;
    yshiftacc = yshiftacc>horizon?horizon:yshiftacc;
    xshiftacc = xshiftacc<-1*horizon?-1*horizon:xshiftacc;
    yshiftacc = yshiftacc<-1*horizon?-1*horizon:yshiftacc;
    freia_common_set_wa(imcurrent, xshiftacc+horizon, yshiftacc+horizon, fdin.framewidth-2*horizon, fdin.frameheight-2*horizon);
    freia_aipo_copy(imstab, imcurrent);
    freia_common_reset_wa(imcurrent);
  }
  else {
    freia_common_set_wa(imcurrent, xshiftacc+horizon, yshiftacc+horizon, fdin.framewidth-2*horizon, fdin.frameheight-2*horizon);
    freia_aipo_copy(imstab, imcurrent);
    freia_common_reset_wa(imcurrent);
  }
  // save stab


  // Contrast enhancement
  if (contrastenhance) {
    {
      //PIPS generated variable
      int I_0;
      I_0 = maximal_shape/2+1;
      {
	freia_status ret;


	{
	  freia_status ret;


	  {
	    int i;
	    if (1!=1) {

	      fprintf(stderr, "ERROR: file %s, line %d, function %s: ""working areas of images are not compatibles\n", "../../api/cipo/src/freiaComplexOpMorpho.c", 25, __FUNCTION__);
	      while (0)
		;
	      _return13 = FREIA_SIZE_ERROR;
	      goto l99985;
	    }
	    if (1!=1) {

	      fprintf(stderr, "ERROR: file %s, line %d, function %s: ""bpp of images are not compatibles\n", "../../api/cipo/src/freiaComplexOpMorpho.c", 30, __FUNCTION__);
	      while (0)
		;
	      _return13 = FREIA_SIZE_ERROR;
	      goto l99985;
	    }

	    if (I_0==0) {
	      freia_aipo_copy(imtmp2, imstab);
	      _return13 = FREIA_OK;
	      goto l99985;
	    }

	    if (8==4) {
	      freia_aipo_erode_8c(imtmp2, imstab, freia_morpho_kernel_4c);
	      for(i = 1; i <= I_0-1; i += 1)
		freia_aipo_erode_8c(imtmp2, imtmp2, freia_morpho_kernel_4c);
	    }
	    else if (8==6) {
	      freia_aipo_erode_6c(imtmp2, imstab, freia_morpho_kernel_6c);
	      for(i = 1; i <= I_0-1; i += 1)
		freia_aipo_erode_6c(imtmp2, imtmp2, freia_morpho_kernel_6c);
	    }
	    else if (8==8) {
	      freia_aipo_erode_8c(imtmp2, imstab, freia_morpho_kernel_8c);
	      for(i = 1; i <= I_0-1; i += 1)
		freia_aipo_erode_8c(imtmp2, imtmp2, freia_morpho_kernel_8c);
	    }
	    else {
	      _return13 = FREIA_INVALID_PARAM;
	      goto l99985;
	    }


	    _return13 = FREIA_OK;
	    goto l99985;
	  l99985: ;
	  }
	  ret = _return13;
	  {
	    freia_status ret = FREIA_OK;
	    int32_t volcurrent;
	    int32_t volprevious;
	    if (1!=1) {

	      fprintf(stderr, "ERROR: file %s, line %d, function %s: ""working areas of images are not compatibles\n", "../../api/cipo/src/freiaComplexOpMorpho.c", 341, __FUNCTION__);
	      while (0)
		;
	      _return14 = FREIA_SIZE_ERROR;
	      goto l99984;
	    }
	    if (1!=1) {

	      fprintf(stderr, "ERROR: file %s, line %d, function %s: ""bpp of images are not compatibles\n", "../../api/cipo/src/freiaComplexOpMorpho.c", 346, __FUNCTION__);
	      while (0)
		;
	      _return14 = FREIA_SIZE_ERROR;
	      goto l99984;
	    }

	    ret = freia_aipo_global_vol(imtmp2, &volcurrent);
	    do {
	      volprevious = volcurrent;
	      {
		freia_status ret;


		{
		  int i;
		  if (1!=1) {

		    fprintf(stderr, "ERROR: file %s, line %d, function %s: ""working areas of images are not compatibles\n", "../../api/cipo/src/freiaComplexOpMorpho.c", 63, __FUNCTION__);
		    while (0)
		      ;
		    _return17 = FREIA_SIZE_ERROR;
		    goto l99981;
		  }
		  if (1!=1) {

		    fprintf(stderr, "ERROR: file %s, line %d, function %s: ""bpp of images are not compatibles\n", "../../api/cipo/src/freiaComplexOpMorpho.c", 68, __FUNCTION__);
		    while (0)
		      ;
		    _return17 = FREIA_SIZE_ERROR;
		    goto l99981;
		  }

		  if (1==0) {
		    freia_aipo_copy(imtmp2, imtmp2);
		    _return17 = FREIA_OK;
		    goto l99981;
		  }

		  if (8==4) {
		    freia_aipo_dilate_8c(imtmp2, imtmp2, freia_morpho_kernel_4c);
		    for(i = 1; i <= 1-1; i += 1)
		      freia_aipo_dilate_8c(imtmp2, imtmp2, freia_morpho_kernel_4c);
		  }
		  else if (8==6) {
		    freia_aipo_dilate_6c(imtmp2, imtmp2, freia_morpho_kernel_6c);
		    for(i = 1; i <= 1-1; i += 1)
		      freia_aipo_dilate_6c(imtmp2, imtmp2, freia_morpho_kernel_6c);
		  }
		  else if (8==8) {
		    freia_aipo_dilate_8c(imtmp2, imtmp2, freia_morpho_kernel_8c);
		    for(i = 1; i <= 1-1; i += 1)
		      freia_aipo_dilate_8c(imtmp2, imtmp2, freia_morpho_kernel_8c);
		  }
		  else {
		    _return17 = FREIA_INVALID_PARAM;
		    goto l99981;
		  }


		  _return17 = FREIA_OK;
		  goto l99981;
		l99981: ;
		}
		ret = _return17;
		ret |= freia_aipo_inf(imtmp2, imtmp2, imstab);


		_return16 = ret;
		goto l99982;
	      l99982: ;
	      }
	      ret |= _return16;
	      ret |= freia_aipo_global_vol(imtmp2, &volcurrent);
	    }
	    while (volcurrent!=volprevious);


	    _return14 = ret;
	    goto l99984;
	  l99984: ;
	  }
	  ret |= _return14;


	  _return11 = ret;
	  goto l99987;
	l99987: ;
	}
	ret = _return11;
	ret |= freia_aipo_sub(imtmp2, imstab, imtmp2);


	_return2 = ret;
	goto l99997;
      l99997: ;
      }
    }
    _return2;
    freia_aipo_threshold(imehn, imtmp2, minimal_contrast, 255, 0);
    {
      freia_status ret = FREIA_OK;
      int32_t volcurrent;
      int32_t volprevious;
      if (1!=1) {

	fprintf(stderr, "ERROR: file %s, line %d, function %s: ""working areas of images are not compatibles\n", "../../api/cipo/src/freiaComplexOpMorpho.c", 341, __FUNCTION__);
	while (0)
	  ;
	_return1 = FREIA_SIZE_ERROR;
	goto l99998;
      }
      if (1!=1) {

	fprintf(stderr, "ERROR: file %s, line %d, function %s: ""bpp of images are not compatibles\n", "../../api/cipo/src/freiaComplexOpMorpho.c", 346, __FUNCTION__);
	while (0)
	  ;
	_return1 = FREIA_SIZE_ERROR;
	goto l99998;
      }

      ret = freia_aipo_global_vol(imehn, &volcurrent);
      do {
	volprevious = volcurrent;
	{
	  freia_status ret;


	  {
	    int i;
	    if (1!=1) {

	      fprintf(stderr, "ERROR: file %s, line %d, function %s: ""working areas of images are not compatibles\n", "../../api/cipo/src/freiaComplexOpMorpho.c", 63, __FUNCTION__);
	      while (0)
		;
	      _return12 = FREIA_SIZE_ERROR;
	      goto l99986;
	    }
	    if (1!=1) {

	      fprintf(stderr, "ERROR: file %s, line %d, function %s: ""bpp of images are not compatibles\n", "../../api/cipo/src/freiaComplexOpMorpho.c", 68, __FUNCTION__);
	      while (0)
		;
	      _return12 = FREIA_SIZE_ERROR;
	      goto l99986;
	    }

	    if (1==0) {
	      freia_aipo_copy(imehn, imehn);
	      _return12 = FREIA_OK;
	      goto l99986;
	    }

	    if (8==4) {
	      freia_aipo_dilate_8c(imehn, imehn, freia_morpho_kernel_4c);
	      for(i = 1; i <= 1-1; i += 1)
		freia_aipo_dilate_8c(imehn, imehn, freia_morpho_kernel_4c);
	    }
	    else if (8==6) {
	      freia_aipo_dilate_6c(imehn, imehn, freia_morpho_kernel_6c);
	      for(i = 1; i <= 1-1; i += 1)
		freia_aipo_dilate_6c(imehn, imehn, freia_morpho_kernel_6c);
	    }
	    else if (8==8) {
	      freia_aipo_dilate_8c(imehn, imehn, freia_morpho_kernel_8c);
	      for(i = 1; i <= 1-1; i += 1)
		freia_aipo_dilate_8c(imehn, imehn, freia_morpho_kernel_8c);
	    }
	    else {
	      _return12 = FREIA_INVALID_PARAM;
	      goto l99986;
	    }


	    _return12 = FREIA_OK;
	    goto l99986;
	  l99986: ;
	  }
	  ret = _return12;
	  ret |= freia_aipo_inf(imehn, imehn, imtmp2);


	  _return10 = ret;
	  goto l99988;
	l99988: ;
	}
	ret |= _return10;
	ret |= freia_aipo_global_vol(imehn, &volcurrent);
      }
      while (volcurrent!=volprevious);


      _return1 = ret;
      goto l99998;
    l99998: ;
    }
    _return1;
  }
  else
    freia_aipo_copy(imehn, imstab);


  // Motion Detection
  if (motiondetect) {

    // absolute difference background and current frame
    freia_aipo_absdiff(immotion, imehn, imbg);

    // background update
    freia_aipo_cast(imtmp3, imehn);
    freia_aipo_mul_const(imtmp3, imtmp3, motion_a);
    freia_aipo_mul_const(imbg16, imbg16, motion_b);
    freia_aipo_add(imbg16, imbg16, imtmp3);
    freia_aipo_div_const(imbg16, imbg16, motion_a+motion_b);
    freia_aipo_cast(imbg, imbg16);


    // measures
    freia_aipo_global_max(immotion, &maxmotion);
    freia_aipo_global_min(immotion, &minmotion);
    if (maxmotion-minmotion>motion_trig)
      binvalue = maxmotion*motion_th/100;

    // threshold
    freia_aipo_threshold(immotion, immotion, binvalue, 255, 1);

    // openning
    ;
    {
      freia_status ret;


      {
	int i;
	if (1!=1) {

	  fprintf(stderr, "ERROR: file %s, line %d, function %s: ""working areas of images are not compatibles\n", "../../api/cipo/src/freiaComplexOpMorpho.c", 25, __FUNCTION__);
	  while (0)
	    ;
	  _return7 = FREIA_SIZE_ERROR;
	  goto l99991;
	}
	if (1!=1) {

	  fprintf(stderr, "ERROR: file %s, line %d, function %s: ""bpp of images are not compatibles\n", "../../api/cipo/src/freiaComplexOpMorpho.c", 30, __FUNCTION__);
	  while (0)
	    ;
	  _return7 = FREIA_SIZE_ERROR;
	  goto l99991;
	}

	if (1==0) {
	  freia_aipo_copy(imtmp2, immotion);
	  _return7 = FREIA_OK;
	  goto l99991;
	}

	if (8==4) {
	  freia_aipo_erode_8c(imtmp2, immotion, freia_morpho_kernel_4c);
	  for(i = 1; i <= 1-1; i += 1)
	    freia_aipo_erode_8c(imtmp2, imtmp2, freia_morpho_kernel_4c);
	}
	else if (8==6) {
	  freia_aipo_erode_6c(imtmp2, immotion, freia_morpho_kernel_6c);
	  for(i = 1; i <= 1-1; i += 1)
	    freia_aipo_erode_6c(imtmp2, imtmp2, freia_morpho_kernel_6c);
	}
	else if (8==8) {
	  freia_aipo_erode_8c(imtmp2, immotion, freia_morpho_kernel_8c);
	  for(i = 1; i <= 1-1; i += 1)
	    freia_aipo_erode_8c(imtmp2, imtmp2, freia_morpho_kernel_8c);
	}
	else {
	  _return7 = FREIA_INVALID_PARAM;
	  goto l99991;
	}


	_return7 = FREIA_OK;
	goto l99991;
      l99991: ;
      }
      ret = _return7;
      {
	int i;
	if (1!=1) {

	  fprintf(stderr, "ERROR: file %s, line %d, function %s: ""working areas of images are not compatibles\n", "../../api/cipo/src/freiaComplexOpMorpho.c", 63, __FUNCTION__);
	  while (0)
	    ;
	  _return5 = FREIA_SIZE_ERROR;
	  goto l99993;
	}
	if (1!=1) {

	  fprintf(stderr, "ERROR: file %s, line %d, function %s: ""bpp of images are not compatibles\n", "../../api/cipo/src/freiaComplexOpMorpho.c", 68, __FUNCTION__);
	  while (0)
	    ;
	  _return5 = FREIA_SIZE_ERROR;
	  goto l99993;
	}

	if (1==0) {
	  freia_aipo_copy(imtmp2, imtmp2);
	  _return5 = FREIA_OK;
	  goto l99993;
	}

	if (8==4) {
	  freia_aipo_dilate_8c(imtmp2, imtmp2, freia_morpho_kernel_4c);
	  for(i = 1; i <= 1-1; i += 1)
	    freia_aipo_dilate_8c(imtmp2, imtmp2, freia_morpho_kernel_4c);
	}
	else if (8==6) {
	  freia_aipo_dilate_6c(imtmp2, imtmp2, freia_morpho_kernel_6c);
	  for(i = 1; i <= 1-1; i += 1)
	    freia_aipo_dilate_6c(imtmp2, imtmp2, freia_morpho_kernel_6c);
	}
	else if (8==8) {
	  freia_aipo_dilate_8c(imtmp2, imtmp2, freia_morpho_kernel_8c);
	  for(i = 1; i <= 1-1; i += 1)
	    freia_aipo_dilate_8c(imtmp2, imtmp2, freia_morpho_kernel_8c);
	}
	else {
	  _return5 = FREIA_INVALID_PARAM;
	  goto l99993;
	}


	_return5 = FREIA_OK;
	goto l99993;
      l99993: ;
      }
      ret |= _return5;


      _return4 = ret;
      goto l99995;
    l99995: ;
    }
    _return4;


    {
      freia_status ret;
      freia_data2d *imtmp;
      if (1!=1) {

	fprintf(stderr, "ERROR: file %s, line %d, function %s: ""working areas of images are not compatibles\n", "../../api/cipo/src/freiaComplexOpMorpho.c", 222, __FUNCTION__);
	while (0)
	  ;
	_return3 = FREIA_SIZE_ERROR;
	goto l99996;
      }
      if (1!=1) {

	fprintf(stderr, "ERROR: file %s, line %d, function %s: ""bpp of images are not compatibles\n", "../../api/cipo/src/freiaComplexOpMorpho.c", 227, __FUNCTION__);
	while (0)
	  ;
	_return3 = FREIA_SIZE_ERROR;
	goto l99996;
      }

      imtmp = freia_common_create_data(immotion->bpp, immotion->widthWa, immotion->heightWa);


      {
	int i;
	if (1!=1) {

	  fprintf(stderr, "ERROR: file %s, line %d, function %s: ""working areas of images are not compatibles\n", "../../api/cipo/src/freiaComplexOpMorpho.c", 63, __FUNCTION__);
	  while (0)
	    ;
	  _return6 = FREIA_SIZE_ERROR;
	  goto l99992;
	}
	if (1!=1) {

	  fprintf(stderr, "ERROR: file %s, line %d, function %s: ""bpp of images are not compatibles\n", "../../api/cipo/src/freiaComplexOpMorpho.c", 68, __FUNCTION__);
	  while (0)
	    ;
	  _return6 = FREIA_SIZE_ERROR;
	  goto l99992;
	}

	if (1==0) {
	  freia_aipo_copy(imtmp, imtmp2);
	  _return6 = FREIA_OK;
	  goto l99992;
	}

	if (8==4) {
	  freia_aipo_dilate_8c(imtmp, imtmp2, freia_morpho_kernel_4c);
	  for(i = 1; i <= 1-1; i += 1)
	    freia_aipo_dilate_8c(imtmp, imtmp, freia_morpho_kernel_4c);
	}
	else if (8==6) {
	  freia_aipo_dilate_6c(imtmp, imtmp2, freia_morpho_kernel_6c);
	  for(i = 1; i <= 1-1; i += 1)
	    freia_aipo_dilate_6c(imtmp, imtmp, freia_morpho_kernel_6c);
	}
	else if (8==8) {
	  freia_aipo_dilate_8c(imtmp, imtmp2, freia_morpho_kernel_8c);
	  for(i = 1; i <= 1-1; i += 1)
	    freia_aipo_dilate_8c(imtmp, imtmp, freia_morpho_kernel_8c);
	}
	else {
	  _return6 = FREIA_INVALID_PARAM;
	  goto l99992;
	}


	_return6 = FREIA_OK;
	goto l99992;
      l99992: ;
      }
      ret = _return6;
      {
	int i;
	if (1!=1) {

	  fprintf(stderr, "ERROR: file %s, line %d, function %s: ""working areas of images are not compatibles\n", "../../api/cipo/src/freiaComplexOpMorpho.c", 25, __FUNCTION__);
	  while (0)
	    ;
	  _return8 = FREIA_SIZE_ERROR;
	  goto l99990;
	}
	if (1!=1) {

	  fprintf(stderr, "ERROR: file %s, line %d, function %s: ""bpp of images are not compatibles\n", "../../api/cipo/src/freiaComplexOpMorpho.c", 30, __FUNCTION__);
	  while (0)
	    ;
	  _return8 = FREIA_SIZE_ERROR;
	  goto l99990;
	}

	if (1==0) {
	  freia_aipo_copy(immotion, imtmp2);
	  _return8 = FREIA_OK;
	  goto l99990;
	}

	if (8==4) {
	  freia_aipo_erode_8c(immotion, imtmp2, freia_morpho_kernel_4c);
	  for(i = 1; i <= 1-1; i += 1)
	    freia_aipo_erode_8c(immotion, immotion, freia_morpho_kernel_4c);
	}
	else if (8==6) {
	  freia_aipo_erode_6c(immotion, imtmp2, freia_morpho_kernel_6c);
	  for(i = 1; i <= 1-1; i += 1)
	    freia_aipo_erode_6c(immotion, immotion, freia_morpho_kernel_6c);
	}
	else if (8==8) {
	  freia_aipo_erode_8c(immotion, imtmp2, freia_morpho_kernel_8c);
	  for(i = 1; i <= 1-1; i += 1)
	    freia_aipo_erode_8c(immotion, immotion, freia_morpho_kernel_8c);
	}
	else {
	  _return8 = FREIA_INVALID_PARAM;
	  goto l99990;
	}


	_return8 = FREIA_OK;
	goto l99990;
      l99990: ;
      }
      ret |= _return8;
      ret |= freia_aipo_sub(immotion, imtmp, immotion);

      freia_common_destruct_data(imtmp);
      _return3 = FREIA_OK;
      goto l99996;
    l99996: ;
    }
    _return3;

    // save contrast motion detection
    freia_aipo_sub_const(immotion, immotion, 1);
    freia_aipo_and_const(immotion, immotion, 1);
    freia_aipo_mul(immotion, imstab, immotion);
    freia_common_tx_image(immotion, &fdmotion);
  }

  fprintf(stdout, "INFO: %s: ""frame %d processed\n", __FUNCTION__, fdin.frameindex);
  goto l99994;
 main_break_1: ;


  freia_common_free(shiftarray);

  freia_common_destruct_data(imcurrent);
  freia_common_destruct_data(imprevious);
  freia_common_destruct_data(imtmp1);
  freia_common_destruct_data(imstab);
  freia_common_destruct_data(imtmp2);
  freia_common_destruct_data(imtmp3);
  freia_common_destruct_data(imehn);
  freia_common_destruct_data(imbg16);
  freia_common_destruct_data(imbg);
  freia_common_destruct_data(immotion);

  freia_common_close_input(&fdin);
  freia_common_close_output(&fdstab);
  freia_common_close_output(&fdehn);
  freia_common_close_output(&fdmotion);

  freia_shutdown();
  return 0;
}
