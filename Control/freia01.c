typedef int int32_t;
typedef int uint32_t;
typedef int freia_status;
typedef struct {int framewidth;int frameheight;int frameindex;} freia_dataio;
typedef struct {int width; int height; int * bpp; int widthWa; int heightWa;}  freia_data2d;

#define FREIA_OK (1)
#define FREIA_SIZE_ERROR (2)

int main(void)
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

  /* Param set by user */
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
  int _return0, _return1, _return2, _return3;


  /*****************/
  /* Init an alloc */
  /*****************/
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

  /*********************/
  /* application start */
  /*********************/
  xshiftacc = 0;
  yshiftacc = 0;


  end = freia_common_rx_image(imcurrent, &fdin);
 l99995:;

  if (!(end==FREIA_OK)) goto break_1;

  /*******************/
  /* Stabilization   */
  /*******************/
  freia_aipo_copy(imprevious, imcurrent);
  end = freia_common_rx_image(imcurrent, &fdin);
  if (end!=FREIA_OK) goto break_1;

  if (stabilize==1) {
    /* save previous frame */

    /* detect shift */
    findGlobalShift(&xshift, &yshift, imcurrent, imprevious, imtmp1, shiftarray, 64, 32, 10);

    /* accumulate shift */
    xshiftacc += xshift;
    yshiftacc += yshift;
    freia_common_set_wa(imcurrent, xshiftacc+horizon, yshiftacc+horizon, 288-2*horizon, 224-2*horizon);
    freia_aipo_copy(imstab, imcurrent);
    freia_common_reset_wa(imcurrent);
    freia_common_tx_image(imstab, &fdstab);
  }
  else {
    freia_common_set_wa(imcurrent, xshiftacc+horizon, yshiftacc+horizon, 288-2*horizon, 224-2*horizon);
    freia_aipo_copy(imstab, imcurrent);
    freia_common_reset_wa(imcurrent);
  }
  /* save stab */


  /*************************/
  /* Contrast enhancement */
  /*************************/
  if (contrastenhance==1) {
    {
      int I_0;
      I_0 = maximal_shape/2+1;
      {
	freia_status ret;

	ret = freia_cipo_geodesic_reconstruct_open(imtmp2, imstab, 8, I_0);
	ret |= freia_aipo_sub(imtmp2, imstab, imtmp2);


	_return2 = ret;
	goto l99997;
      l99997:            ;
      }
    }
    _return2;
    freia_aipo_threshold(imehn, imtmp2, minimal_contrast, 255, 0);
    {
      freia_status ret = FREIA_OK;
      int32_t volcurrent;
      int32_t volprevious;
      if (1!=1) {
	//fprintf(stderr, "working areas of images are not
	//compatibles\n""ERROR: file %s, line %d, function %s: ",
	//"../../api/cipo/src/freiaComplexOpMorpho.c", 244,
	//__FUNCTION__);
	;
	while (0)
	  ;
	_return3 = FREIA_SIZE_ERROR;
	goto l99996;
      }
      else
	if (1!=1) {

	  //fprintf(stderr, "bpp of images are not
	  //compatibles\n""ERROR: file %s, line %d, function %s: ",
	  //"../../api/cipo/src/freiaComplexOpMorpho.c", 249,
	  //__FUNCTION__);
	  ;;
	  while (0)
	    ;
	  _return3 = FREIA_SIZE_ERROR;
	  goto l99996;
	}
	else

	  ret = freia_aipo_global_vol(imehn, &volcurrent);
      do {
	ret |= freia_cipo_geodesic_dilate(imehn, imehn, imtmp2, 8, 1);
	volprevious = volcurrent;
	ret |= freia_aipo_global_vol(imehn, &volcurrent);
      }
      while (volcurrent!=volprevious);


      _return3 = ret;
      goto l99996;
    l99996:         ;
    }
    _return3;
    freia_common_tx_image(imehn, &fdehn);
  }
  else
    freia_aipo_copy(imehn, imstab);


  /********************/
  /* Motion Detection */
  /********************/
  if (motiondetect==1) {

    /* absolute difference background and current frame */
    freia_aipo_absdiff(immotion, imehn, imbg);

    /* background update */
    freia_aipo_cast(imtmp3, imehn);
    freia_aipo_mul_const(imtmp3, imtmp3, motion_a);
    freia_aipo_mul_const(imbg16, imbg16, motion_b);
    freia_aipo_add(imbg16, imbg16, imtmp3);
    freia_aipo_div_const(imbg16, imbg16, motion_a+motion_b);
    freia_aipo_cast(imbg, imbg16);


    /* max measure */
    freia_aipo_global_max(immotion, &maxmotion);

    /* min measure */
    freia_aipo_global_min(immotion, &minmotion);
    if (maxmotion-minmotion>motion_trig)
      binvalue = maxmotion*motion_th/100;

    /* threashold */
    freia_aipo_threshold(immotion, immotion, binvalue, 255, 1);

    /* openning */
    ;
    {
      freia_status ret;

      ret = freia_cipo_erode(imtmp2, immotion, 8, 1);
      ret |= freia_cipo_dilate(imtmp2, imtmp2, 8, 1);


      _return1 = ret;
      goto l99998;
    l99998:         ;
    }
    _return1;
    {
      freia_status ret;
      freia_data2d *imtmp;
      if (1!=1) {
	//fprintf(stderr, "working areas of images are not
	//compatibles\n""ERROR: file %s, line %d, function %s: ",
	//"../../api/cipo/src/freiaComplexOpMorpho.c", 125,
	//__FUNCTION__);
	;
	while (0)
	  ;
	_return0 = FREIA_SIZE_ERROR;
	goto l99999;
      }
      else
	if (1!=1) {

	  //fprintf(stderr, "bpp of images are not
	  //compatibles\n""ERROR: file %s, line %d, function %s: ",
	  //"../../api/cipo/src/freiaComplexOpMorpho.c", 130,
	  //__FUNCTION__);
	  ;
	  ;
	  while (0)
	    ;
	  _return0 = FREIA_SIZE_ERROR;
	  goto l99999;
	}
	else
	  imtmp = freia_common_create_data(immotion->bpp, immotion->widthWa, immotion->heightWa);

      ret = freia_cipo_dilate(imtmp, imtmp2, 8, 1);
      ret |= freia_cipo_erode(immotion, imtmp2, 8, 1);
      ret |= freia_aipo_sub(immotion, imtmp, immotion);

      freia_common_destruct_data(imtmp);
      _return0 = FREIA_OK;
      goto l99999;
    l99999:         ;
    }
    _return0;

    /* save contrast motion detection */
    freia_aipo_sub_const(immotion, immotion, 1);
    freia_aipo_and_const(immotion, immotion, 1);
    freia_aipo_mul(immotion, imstab, immotion);
    freia_common_tx_image(immotion, &fdmotion);
  }
  else
    //fprintf(stdout, "frame %d processed\n""INFO: %s: ",
    //__FUNCTION__, fdin.frameindex);
    ;
  goto l99995;
 break_1:   ;


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

  return 0;
}
