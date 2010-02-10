#include <flgrCoreDispatch.h>
#include <flgrCoreSlideWindow.h>

#define flgr_free_align flgr_free
#define flgr_malloc_align flgr_malloc
#define FLGR_MACRO_RASTER_SLIDE_WINDOW_2D(dtype,get_nhb_op)	\
  int i,j,w,h,spp = imgsrc->spp;				\
  dtype *vector_array;						\
  dtype *data_array;						\
  FLGR_Vector *result;						\
  FLGR_Data2D *nhbEven,*nhbOdd,*nhbrs;				\
  FLGR_NhbBox2D *extr;						\
								\
  					\
								\
  w=imgsrc->size_x;						\
  h=imgsrc->size_y;						\
								\
  result = flgr_vector_create(imgsrc->spp,imgsrc->type);	\
								\
  vector_array = (dtype *) result->array;			\
								\
  extr = flgr2d_create_neighbor_box(nhb);			\
								\
  nhbEven=flgr2d_create_neighborhood_from(nhb);			\
  nhbOdd=flgr2d_create_neighborhood_from(nhb);			\
								\
  flgr2d_fill_nhbs_for_6_connexity(nhbEven,nhbOdd,nhb,nhb_sym);	\
								\
  if(imgdest==imgsrc) {						\
    flgr2d_apply_raster_scan_method_##dtype(nhbOdd);		\
    flgr2d_apply_raster_scan_method_##dtype(nhbEven);		\
  }								\
								\
  for(i=0 ; i<h; i++) {						\
								\
    data_array = (dtype *) (imgdest->array[i]);			\
								\
    for(j=0 ; j<w ; j++) {					\
								\
      nhbrs = (((i%2)==1) ? nhbOdd : nhbEven);			\
								\
      get_nhb_op##_##dtype(extr,imgsrc,nhbrs,i,j);		\
								\
      (*computeNhb)(result,extr);				\
      								\
      flgr_set_data_array_vector_##dtype(data_array,		\
					 vector_array,		\
					 spp,j);		\
    }								\
  }								\
								\
  flgr2d_destroy(nhbOdd);					\
  flgr2d_destroy(nhbEven);					\
  flgr2d_destroy_neighbor_box(extr);				\
  flgr_vector_destroy(result);					\
								\
  return

#define FLGR_APPLY_RASTER_SCAN_METHOD_2D(dtype)				\
  FLGR_Vector *vec = flgr_vector_create(nhb->spp,nhb->type);		\
  int i,j;								\
  int w = nhb->size_x;							\
  int h = nhb->size_y;							\
									\
  						\
									\
  flgr_vector_populate_from_scalar_##dtype(vec,0);			\
									\
  for(j=(w/2+1);j<w;j++) flgr2d_set_data_vector_##dtype(nhb,h/2,j,vec);	\
									\
  for(i = (h/2)+1 ; i<h ; i++) {					\
    for(j=0;j<w;j++) flgr2d_set_data_vector_##dtype(nhb,i,j,vec);	\
  }									\
									\
  flgr_vector_destroy(vec);						\
									\
  return
#define FLGR_MACRO_GET_NHB_2D_IN_DATA(dtype)				\
  int i,j,m;								\
  int k,l,n;								\
  int nbrow,nbcol;							\
  int startrow,stoprow;							\
  int startcol,stopcol;							\
  int nhbsize_xdiv2;							\
  int nhbsize_ydiv2;							\
  int spp = dat->spp;							\
  dtype **pnhb = (dtype **) nhb->array;					\
  dtype **pdat = (dtype **) dat->array;					\
  dtype *list_data;							\
  dtype *list_nhb;							\
  int *list_coord_x;							\
  int *list_coord_y;							\
  int *size = extr->size;						\
  dtype nhb_val,data_val;						\
									\
  						\
									\
  nhbsize_xdiv2=(nhb->size_x)>>1;					\
  nhbsize_ydiv2=(nhb->size_y)>>1;					\
  nbrow=dat->size_y; nbcol=dat->size_x;					\
									\
  extr->center_coord_y=row;						\
  extr->center_coord_x=col;						\
									\
  startrow=row - nhbsize_ydiv2;						\
  startrow = (startrow<0) ? 0 : startrow;				\
									\
  stoprow=row + nhbsize_ydiv2;						\
  stoprow = (stoprow>=nbrow) ? nbrow-1 : stoprow;			\
									\
  startcol=col - nhbsize_xdiv2;						\
  startcol = (startcol<0) ? 0 : startcol;				\
									\
  stopcol=col + nhbsize_xdiv2;						\
  stopcol = (stopcol>=nbcol) ? nbcol-1 : stopcol;			\
									\
									\
  flgr_get_data_array_vector_##dtype((dtype*) (extr->center_data_val->array), \
				     (dtype*) (dat->array[row]),	\
				     spp, col);				\
									\
  flgr_get_data_array_vector_##dtype((dtype*) (extr->center_nhb_val->array), \
				     (dtype*) (nhb->array[nhbsize_ydiv2]), \
				     spp, nhbsize_xdiv2);		\
									\
									\
									\
  for(n=0 ; n<spp ; n++) {						\
    list_data = (dtype*) extr->list_data_val[n];			\
    list_nhb = (dtype*) extr->list_nhb_val[n];				\
    list_coord_x = (int*) extr->list_coord_x[n];			\
    list_coord_y = (int*) extr->list_coord_y[n];			\
    m=0;								\
    k=startrow-row+nhbsize_ydiv2;					\
    for(i=startrow ; i<=stoprow ; i++,k++) {				\
									\
      l=startcol-col+nhbsize_xdiv2;					\
    									\
      for(j=startcol ; j<=stopcol ; j++,l++) {				\
									\
	nhb_val = flgr2d_get_data_array_##dtype(pnhb, k, l*spp+n);	\
	if(nhb_val!=0) {						\
	  data_val = flgr2d_get_data_array_##dtype(pdat,i, j*spp+n);	\
	  list_coord_x[m] = l;						\
	  list_coord_y[m] = k;						\
	  list_data[m] = data_val;					\
	  list_nhb[m] = nhb_val;					\
	  m++;								\
	}								\
 									\
      }									\
    }									\
    size[n]=m;								\
  }									\
									\
  return
#define FLGR_MACRO_CREATE2D(dtype, dtypename)				\
  int i;								\
  FLGR_Data2D *dat;							\
									\
  						\
									\
  if(size_y<0) {							\
    POST_ERROR("Number of rows is negative !\n");			\
    return NULL;							\
  }									\
									\
  if(size_x<0) {							\
    POST_ERROR("Number of columns is negative !\n");			\
    return NULL;							\
  }									\
									\
  if(spp<1) {								\
    POST_ERROR("Sample Per Pixel (spp) must be > 0 !\n");		\
    return NULL;							\
  }									\
									\
  dat = (FLGR_Data2D*) flgr_malloc(sizeof(FLGR_Data2D));		\
  dat->row = (FLGR_Data1D**) flgr_malloc((size_y+16)*			\
					 sizeof(FLGR_Data1D*));		\
  dat->array = flgr_malloc((size_y+16)*sizeof(dtype*));			\
  dat->dim = 2;								\
  dat->size_struct = sizeof(FLGR_Data2D);				\
  dat->bps = flgr_get_bps_from_type(dtypename);				\
  dat->spp = spp;							\
  dat->shape = shape;							\
  dat->connexity = connexity;						\
  dat->size_y=size_y;							\
  dat->size_x=size_x;							\
  dat->type = dtypename;						\
  dat->link_overlap = -1;						\
  dat->link_position = -1;						\
  dat->link_number = -1;						\
									\
  for(i=0;i<(size_y+16);i++) {						\
    dat->row[i] = flgr1d_create_##dtype(size_x, spp,FLGR_NO_SHAPE);	\
    dat->row[i]->ref2d = i;						\
    dat->array[i] = dat->row[i]->array;					\
  }									\
									\
  return dat
#define FLGR_MACRO_CREATE1D(dtype, dtypename)				\
  FLGR_Data1D *dat;							\
  						\
									\
  if(length<0) {							\
    POST_ERROR("Negative length!\n");					\
    return NULL;							\
  }									\
  if(spp<1) {								\
    POST_ERROR("Sample Per Pixel (spp) must be > 0 !\n");		\
    return NULL;							\
  }									\
  dat = (FLGR_Data1D*) flgr_malloc(sizeof(FLGR_Data1D));		\
  dat->dim = 1;								\
  dat->size_struct = sizeof(FLGR_Data1D);				\
  dat->bps = flgr_get_bps_from_type(dtypename);				\
  dat->ref2d = -1;							\
  dat->shape = shape;							\
  dat->spp   = spp;							\
  dat->length = length;							\
  dat->array_phantom = flgr_malloc_align((length*dat->bps*spp)/8+64,16);\
  dat->array = dat->array_phantom+32;					\
  dat->type = dtypename;						\
									\
  return dat
#define FLGR_MACRO_FILL_2D_SHP(dtype)					\
  FLGR_Vector *vec1 = flgr_vector_create(nhb->spp,nhb->type);		\
  FLGR_Data2D *tmp = flgr2d_create_from(nhb);				\
  int h=nhb->size_y;							\
  int w=nhb->size_x;							\
  int size=((height<width) ? height:width);				\
  double size_square  = ((double) (size/2)) / (1+sqrt(2));		\
  double size_square2 = floor(size_square);				\
  int nbsqr = (int) ((size_square-size_square2) < 0.5f ?		\
		     size_square2 : (size_square2+1));			\
  int i;								\
  									\
  flgr2d_clear_all(nhb);						\
  flgr_vector_populate_from_scalar_##dtype(vec1,1);			\
   									\
  if(shape == FLGR_HEX) {						\
    flgr2d_set_data_vector_##dtype(nhb,h/2,w/2,vec1);			\
    for(i=0;i<size/2;i++) flgr2d_native_dilate_6_connexity_##dtype(nhb);\
    if( ((size/2)%2)==1 ) {						\
      flgr2d_fill_nhb_even_rows_##dtype(tmp,nhb);			\
      flgr2d_copy(nhb,tmp);						\
    }									\
    									\
  }else if(shape == FLGR_RECT) {					\
    flgr2d_draw_filled_rectangle_##dtype(nhb,w/2-width/2, h/2-height/2,	\
					 width,height,vec1);		\
									\
  }else if(shape == FLGR_DIAMOND) {					\
    flgr2d_set_data_vector_##dtype(nhb,h/2,w/2,vec1);			\
    for(i=0;i<size/2;i++)						\
      flgr2d_native_dilate_4_connexity_##dtype(nhb);			\
									\
  }else if(shape == FLGR_OCTAGON) {					\
    flgr2d_set_data_vector_##dtype(nhb,h/2,w/2,vec1);			\
    for(i=0 ; i<size/2-nbsqr ; i++)					\
      flgr2d_native_dilate_4_connexity_##dtype(nhb);			\
    									\
    for(i=0 ; i<nbsqr ; i++)						\
      flgr2d_native_dilate_8_connexity_##dtype(nhb);			\
 									\
  }else if(shape == FLGR_DISC) {					\
    flgr2d_draw_disc_##dtype(nhb,w/2,h/2,size/2,vec1);			\
									\
  }else if(shape == FLGR_ELLIPSE) {					\
    flgr2d_draw_filled_ellipse_##dtype(nhb,w/2,h/2,			\
				       width/2,height/2,vec1);		\
									\
  }else if(shape == FLGR_SLASH) {					\
    flgr2d_draw_line_##dtype(nhb, w/2-width/2, height-1+h/2-height/2,	\
			     width-1+w/2-width/2, h/2-height/2, vec1);	\
									\
  }else if(shape == FLGR_BSLASH) {					\
    flgr2d_draw_line_##dtype(nhb,w/2-width/2, h/2-height/2,		\
			     width-1+w/2-width/2, height-1+h/2-height/2,vec1); \
									\
  }else if(shape == FLGR_CROSS) {					\
    flgr2d_draw_vertical_line_##dtype(nhb,w/2,h/2-height/2,height,vec1); \
    flgr2d_draw_horizontal_line_##dtype(nhb,w/2-width/2,h/2,width,vec1); \
									\
  }else if(shape == FLGR_CROSSX) {					\
    flgr2d_draw_line_##dtype(nhb,w/2-width/2, height-1+h/2-height/2,	\
			     width-1+w/2-width/2,h/2-height/2,vec1);	\
    flgr2d_draw_line_##dtype(nhb, w/2-width/2, h/2-height/2,		\
			     width-1+w/2-width/2,height-1+h/2-height/2,vec1); \
  }									\
									\
  flgr_vector_destroy(vec1);						\
  flgr2d_destroy(tmp);							\
									\
  return
#define FLGR_MACRO_COPY2D(dtypeDest, dtypeSrc)			\
  int i;							\
  FLGR_Data1D **dest = datdest->row;				\
  FLGR_Data1D **src = datsrc->row;				\
								\
  					\
								\
  for(i=0;i<datsrc->size_y;i++) {				\
    flgr1d_copy_##dtypeDest##_##dtypeSrc(dest[i],src[i]);	\
  }								\
  return
#define FLGR_MACRO_COPY1D_SAME_TYPE					\
  memcpy(datdest->array, datsrc->array, (datdest->bps*datdest->length*datdest->spp)/8)
#define FLGR_MACRO_DRAW_2D_2D_DISC(dtype)				\
  int d, y, x;								\
									\
									\
									\
  d = 3 - (2 * radius);							\
  x = 0;								\
  y = radius;								\
									\
  while (y >= x) {							\
    flgr2d_draw_horizontal_line_##dtype(dat, cx-x, cy-y, 2*x+1, color);	\
    flgr2d_draw_horizontal_line_##dtype(dat, cx-x, cy+y, 2*x+1, color);	\
    flgr2d_draw_horizontal_line_##dtype(dat, cx-y, cy-x, 2*y+1, color);	\
    flgr2d_draw_horizontal_line_##dtype(dat, cx-y, cy+x, 2*y+1, color);	\
									\
    if (d < 0)								\
      d = d + (4 * x) + 6;						\
    else {								\
      d = d + 4 * (x - y) + 10;						\
      y--;								\
    }									\
									\
    x++;								\
  }									\
  return
#define FLGR_MACRO_DRAW_2D_2D_FILLED_ELLIPSE(dtype)			\
  /* e(x,y) = b^2*x^2 + a^2*y^2 - a^2*b^2 */				\
  int x = 0, y = b;							\
  unsigned int width = 1;						\
  long a2 = (long)a*a, b2 = (long)b*b;					\
  long crit1 = -(a2/4 + a%2 + b2);					\
  long crit2 = -(b2/4 + b%2 + a2);					\
  long crit3 = -(b2/4 + b%2);						\
  long t = -a2*y; /* e(x+1/2,y-1/2) - (a^2+b^2)/4 */			\
  long dxt = 2*b2*x, dyt = -2*a2*y;					\
  long d2xt = 2*b2, d2yt = 2*a2;					\
									\
  						\
									\
  while (y>=0 && x<=a) {						\
    if (t + b2*x <= crit1 ||     /* e(x+1,y-1/2) <= 0 */		\
	t + a2*y <= crit3) {     /* e(x+1/2,y) <= 0 */			\
      x++, dxt += d2xt, t += dxt;					\
      width += 2;							\
    }									\
    else if (t - a2*y > crit2) { /* e(x+1/2,y-1) > 0 */			\
      flgr2d_draw_horizontal_line_##dtype(dat,cx-x,cy-y,width,color);	\
      if (y!=0)								\
	flgr2d_draw_horizontal_line_##dtype(dat,cx-x,cy+y,width,color); \
      y--, dyt += d2yt, t += dyt;					\
    }									\
    else {								\
      flgr2d_draw_horizontal_line_##dtype(dat,cx-x,cy-y,width,color);	\
      if (y!=0)								\
	flgr2d_draw_horizontal_line_##dtype(dat,cx-x,cy+y,width,color); \
      x++, dxt += d2xt, t += dxt;					\
      y--, dyt += d2yt, t += dyt;					\
      width += 2;							\
    }									\
  }									\
  if (b == 0)								\
    flgr2d_draw_horizontal_line_##dtype(dat,cx-a, cy, 2*a+1,color);	\
  									\
  return
#define FLGR_MACRO_DRAW_2D_FILLED_RECT(dtype)				\
  int i;								\
									\
  						\
									\
  for(i=y; i<y+size_y ; i++)						\
    flgr2d_draw_horizontal_line_##dtype(dat,x, i, size_x , color);	\
  return
#define FLGR_MACRO_DRAW_2D_HORZ_LINE(dtype)	\
  int i;					\
						\
  			\
						\
  for(i=x;i<x+size_x;i++) {			\
    flgr2d_draw_point_##dtype(dat,i,y,color);	\
  }						\
  return
#define FLGR_MACRO_DRAW_2D_LINE(dtype)			\
  int d, dx, dy, aincr, bincr, xincr, yincr, x, y;	\
							\
							\
							\
  if (abs(x2 - x1) < abs(y2 - y1)) {			\
    /* vertical axis */					\
							\
    if (y1 > y2) {					\
      dataExchange(&x1, &x2);				\
      dataExchange(&y1, &y2);				\
    }							\
							\
    xincr = x2 > x1 ? 1 : -1;				\
    dy = y2 - y1;					\
    dx = abs(x2 - x1);					\
    d = 2 * dx - dy;					\
    aincr = 2 * (dx - dy);				\
    bincr = 2 * dx;					\
    x = x1;						\
    y = y1;						\
							\
    flgr2d_draw_point_##dtype(dat,x,y,color);		\
							\
    for (y = y1+1; y <= y2; ++y) {			\
      if (d >= 0) {					\
	x += xincr;					\
	d += aincr;					\
      } else						\
	d += bincr;					\
							\
      flgr2d_draw_point_##dtype(dat,x,y,color);		\
    }							\
							\
  } else {						\
    /* horizontal axis */				\
							\
    if (x1 > x2) {					\
      dataExchange(&x1, &x2);				\
      dataExchange(&y1, &y2);				\
    }							\
							\
    yincr = y2 > y1 ? 1 : -1;				\
    dx = x2 - x1;					\
    dy = abs(y2 - y1);					\
    d = 2 * dy - dx;					\
    aincr = 2 * (dy - dx);				\
    bincr = 2 * dy;					\
    x = x1;						\
    y = y1;						\
							\
    flgr2d_draw_point_##dtype(dat,x,y,color);		\
							\
    for (x = x1+1; x <= x2; ++x) {			\
      if (d >= 0) {					\
	y += yincr;					\
	d += aincr;					\
      } else						\
	d += bincr;					\
							\
      flgr2d_draw_point_##dtype(dat,x,y,color);		\
    }							\
  }							\
  return
#define FLGR_MACRO_DRAW2D_POINT(dtype)					\
  						\
									\
  if(y<0) return;							\
  if(x<0) return;							\
  if(y>=dat->size_y) return;						\
  if(x>=dat->size_x)  return;						\
									\
  flgr2d_set_data_vector_##dtype(dat, y, x, color);			\
									\
  return
#define FLGR_MACRO_DRAW_2D_VERT_LINE(dtype)	\
  int i;					\
  			\
						\
						\
  for(i=y;i<y+size_y;i++) {			\
    flgr2d_draw_point_##dtype(dat,x,i,color);	\
  }						\
  return
#define FLGR_MACRO_FILL_NHB_EVEN_ROWS(dtype)				\
  int mid=datsrc->size_y/2;						\
  int h=datsrc->size_y;							\
  int w=datsrc->size_x;							\
  int i,j;								\
  FLGR_Vector *tmp = flgr_vector_create(datsrc->spp,datsrc->type);	\
  						\
									\
									\
  for(i=mid ; i>=0 ; i-=2) {						\
    for(j=0;j<w;j++) {							\
      flgr2d_get_data_vector_##dtype(datsrc,i,j,tmp);			\
      flgr2d_set_data_vector_##dtype(datdest,i,j, tmp);			\
    }									\
  }									\
									\
  for(i=mid-1 ; i>=0 ; i-=2) {						\
    for(j=0;j<w-1;j++) {						\
      flgr2d_get_data_vector_##dtype(datsrc,i,j,tmp);			\
      flgr2d_set_data_vector_##dtype(datdest,i,j+1,tmp);		\
    }									\
    flgr2d_get_data_vector_##dtype(datsrc,i,w-1,tmp);			\
    flgr2d_set_data_vector_##dtype(datdest,i,0,tmp);			\
  }									\
									\
  for(i=mid+1 ; i<h ; i+=2) {						\
    for(j=0;j<w-1;j++) {						\
      flgr2d_get_data_vector_##dtype(datsrc,i,j,tmp);			\
      flgr2d_set_data_vector_##dtype(datdest,i,j+1,tmp);		\
    }									\
    flgr2d_get_data_vector_##dtype(datsrc,i,w-1,tmp);			\
    flgr2d_set_data_vector_##dtype(datdest,i,0,tmp);			\
  }									\
									\
  for(i=mid+2 ; i<h ; i+=2) {						\
    for(j=0;j<w;j++) {							\
      flgr2d_get_data_vector_##dtype(datsrc,i,j,tmp);			\
      flgr2d_set_data_vector_##dtype(datdest,i,j,tmp);			\
    }									\
  }									\
									\
  flgr_vector_destroy(tmp);						\
									\
  return
#define FLGR_MACRO_FILL_NHB_ODD_ROWS(dtype)				\
  int mid=datsrc->size_y/2;						\
  int h=datsrc->size_y;							\
  int w=datsrc->size_x;							\
  int i,j;								\
  FLGR_Vector *tmp = flgr_vector_create(datsrc->spp,datsrc->type);	\
									\
  						\
									\
  for(i=mid ; i>=0 ; i-=2) {						\
    for(j=0;j<w;j++) {							\
      flgr2d_get_data_vector_##dtype(datsrc,i,j,tmp);			\
      flgr2d_set_data_vector_##dtype(datdest,i,j, tmp);			\
    }									\
  }									\
									\
  for(i=mid-1 ; i>=0 ; i-=2) {						\
    for(j=0;j<w-1;j++) {						\
      flgr2d_get_data_vector_##dtype(datsrc,i,j+1,tmp);			\
      flgr2d_set_data_vector_##dtype(datdest,i,j,tmp);			\
    }									\
    flgr2d_get_data_vector_##dtype(datsrc,i,0,tmp);			\
    flgr2d_set_data_vector_##dtype(datdest,i,w-1,tmp);			\
  }									\
									\
  for(i=mid+1 ; i<h ; i+=2) {						\
    for(j=0;j<w-1;j++) {						\
      flgr2d_get_data_vector_##dtype(datsrc,i,j+1,tmp);			\
      flgr2d_set_data_vector_##dtype(datdest,i,j,tmp);			\
    }									\
    flgr2d_get_data_vector_##dtype(datsrc,i,0,tmp);			\
    flgr2d_set_data_vector_##dtype(datdest,i,w-1,tmp);			\
  }									\
									\
  for(i=mid+2 ; i<h ; i+=2) {						\
    for(j=0;j<w;j++) {							\
      flgr2d_get_data_vector_##dtype(datsrc,i,j,tmp);			\
      flgr2d_set_data_vector_##dtype(datdest,i,j,tmp);			\
    }									\
  }									\
									\
  flgr_vector_destroy(tmp);						\
  return
#define FLGR_MACRO_GET_DATA2D_VECTOR(dtype)				\
  dtype *array_s;							\
  dtype *array_d = (dtype*) (vct->array);				\
  						\
									\
  row = flgr_normalize_coordinate(row,dat->size_y);			\
  col = flgr_normalize_coordinate(col,dat->size_x);			\
  array_s = (dtype*) (dat->array[row]);					\
  flgr_get_data_array_vector_##dtype(array_d, array_s, vct->spp, col)
#define FLGR_MACRO_IMPORT2D_RAW(dtype)			\
  FLGR_Data1D **dest = datdest->row;			\
  int spp = datdest->spp;				\
  int size = (datdest->size_x*sizeof(dtype)*spp);	\
  int i;						\
							\
  				\
							\
  for(i=0 ; i < datdest->size_y ; i++) {		\
    flgr1d_import_raw_##dtype(dest[i],raw);		\
    raw += size;					\
  }							\
  return
#define FLGR_MACRO_MIRROR2D_HORIZONTAL(dtype)		\
  FLGR_Vector *vec1;					\
  int i,j,k;						\
							\
  				\
							\
  vec1 = flgr_vector_create(datsrc->spp,datsrc->type);	\
  							\
  for(i=0 ; i < datsrc->size_y ; i++) {			\
    for(j=0,k=datsrc->size_x-1 ; k >= 0 ; k--,j++) {	\
      flgr2d_get_data_vector_##dtype(datsrc,i,j,vec1);	\
      flgr2d_set_data_vector_##dtype(datdest,i,k,vec1);	\
    }							\
  }							\
  							\
  flgr_vector_destroy(vec1);				\
  return
#define FLGR_MACRO_NATIVE_DILATE(dtype,					\
				 c00,c01,c02,				\
				 c10,c11,c12,				\
				 c20,c21,c22)				\
  FLGR_Data2D *nhbcopy;							\
  dtype **seodd;							\
  dtype **seeven;							\
  dtype **se;								\
  int i,j;								\
  int k,l;								\
  int m,n;								\
									\
  dtype valse;								\
									\
  FLGR_Vector *vecPixValue = flgr_vector_create(nhb->spp,nhb->type);	\
  FLGR_Vector *vecPixMax   = flgr_vector_create(nhb->spp,nhb->type);	\
  FLGR_Vector *vecSeValue  = flgr_vector_create(nhb->spp,nhb->type);	\
									\
  						\
									\
  nhbcopy = flgr2d_create_from(nhb);					\
  flgr2d_copy(nhbcopy,nhb);						\
									\
  seodd = (dtype**) flgr_malloc(sizeof(dtype*)*3);			\
  seodd[0] = (dtype*) flgr_malloc(sizeof(dtype)*3);			\
  seodd[1] = (dtype*) flgr_malloc(sizeof(dtype)*3);			\
  seodd[2] = (dtype*) flgr_malloc(sizeof(dtype)*3);			\
									\
  seeven = (dtype**) flgr_malloc(sizeof(dtype*)*3);			\
  seeven[0] = (dtype*) flgr_malloc(sizeof(dtype)*3);			\
  seeven[1] = (dtype*) flgr_malloc(sizeof(dtype)*3);			\
  seeven[2] = (dtype*) flgr_malloc(sizeof(dtype)*3);			\
									\
  flgr2d_set_data_array_##dtype(seodd,0,0,c02);				\
  flgr2d_set_data_array_##dtype(seodd,0,1,c01);				\
  flgr2d_set_data_array_##dtype(seodd,0,2,c00);				\
  flgr2d_set_data_array_##dtype(seodd,1,0,c10);				\
  flgr2d_set_data_array_##dtype(seodd,1,1,c11);				\
  flgr2d_set_data_array_##dtype(seodd,1,2,c12);				\
  flgr2d_set_data_array_##dtype(seodd,2,0,c22);				\
  flgr2d_set_data_array_##dtype(seodd,2,1,c21);				\
  flgr2d_set_data_array_##dtype(seodd,2,2,c20);				\
  									\
  flgr2d_set_data_array_##dtype(seeven,0,0,c00);			\
  flgr2d_set_data_array_##dtype(seeven,0,1,c01);			\
  flgr2d_set_data_array_##dtype(seeven,0,2,c02);			\
  flgr2d_set_data_array_##dtype(seeven,1,0,c10);			\
  flgr2d_set_data_array_##dtype(seeven,1,1,c11);			\
  flgr2d_set_data_array_##dtype(seeven,1,2,c12);			\
  flgr2d_set_data_array_##dtype(seeven,2,0,c20);			\
  flgr2d_set_data_array_##dtype(seeven,2,1,c21);			\
  flgr2d_set_data_array_##dtype(seeven,2,2,c22);			\
									\
  for(i=0; i<nhb->size_y; i++) {					\
    for(j=0; j<nhb->size_x; j++) {					\
									\
      flgr_vector_populate_from_scalar_##dtype(vecPixMax,0);		\
      									\
      for(k=(i-1),m=0 ; k<=(i+1) ; k++,m++) {				\
	if((k>=0) && (k<nhb->size_y)) {					\
	  for(l=(j-1),n=0 ; l<=(j+1) ; l++,n++) {			\
	    if((l>=0) && (l<nhb->size_x)) {				\
									\
	      se = (((i%2)==1) ? seodd:seeven);				\
									\
	      valse = flgr2d_get_data_array_##dtype(se,m,n);		\
	      flgr_vector_populate_from_scalar_##dtype(vecSeValue,	\
						       valse);		\
									\
	      flgr2d_get_data_vector_no_norm_##dtype(nhbcopy,k,l,	\
						     vecPixValue);	\
									\
	      flgr_vector_mult_##dtype(vecPixValue,			\
				       vecPixValue,vecSeValue);		\
									\
	      flgr_vector_sup_##dtype(vecPixMax,			\
				      vecPixMax,vecPixValue);		\
									\
	    }								\
	  }								\
	}								\
      }									\
									\
      flgr2d_set_data_vector_##dtype(nhb,i,j,vecPixMax);		\
 									\
    }									\
  }									\
									\
  flgr_vector_destroy(vecPixValue);					\
  flgr_vector_destroy(vecPixMax);					\
  flgr_vector_destroy(vecSeValue);					\
									\
  flgr2d_destroy(nhbcopy);						\
  flgr_free(seodd[0]);							\
  flgr_free(seodd[1]);							\
  flgr_free(seodd[2]);							\
  flgr_free(seodd);							\
  flgr_free(seeven[0]);							\
  flgr_free(seeven[1]);							\
  flgr_free(seeven[2]);							\
  flgr_free(seeven);							\
  return

#define FLGR_MACRO_GET_DATA2D_VECTOR_NO_NORM(dtype)			\
  dtype *array_s = (dtype*) (dat->array[row]);				\
  dtype *array_d = (dtype*) (vct->array);				\
  						\
									\
  flgr_get_data_array_vector_##dtype(array_d, array_s, vct->spp, col)

#define FLGR_MACRO_SET_DATA2D_VECTOR(dtype)				\
  dtype *array_s = (dtype*) (vct->array);				\
  dtype *array_d = (dtype*) (dat->array[row]);				\
  						\
									\
  flgr_set_data_array_vector_##dtype(array_d, array_s, vct->spp, col)

#define FLGR_MACRO_VECTOR_DIADIC_OP(dtype,arithop)	\
  int k;						\
  int spp = vctdest->spp;				\
  dtype *p1 = (dtype*) vct1->array;			\
  dtype *p2 = (dtype*) vct2->array;			\
  dtype *pdest = (dtype*) vctdest->array;		\
  dtype a1,a2;						\
							\
  				\
							\
  for(k=0 ; k<spp ; k++) {				\
    a1 = flgr_get_array_##dtype(p1,k);			\
    a2 = flgr_get_array_##dtype(p2,k);			\
    flgr_set_array_##dtype(pdest,k,arithop(a1,a2));	\
  }							\
  return
#define FLGR_MACRO_FLGR_VECTOR_POPULATE_SCALAR(dtype)	\
  int k;						\
  dtype *vctar = (dtype*) vctdest->array;		\
							\
  				\
							\
  for(k=0 ; k<vctdest->spp ; k++) {			\
    flgr_set_array_##dtype(vctar,k,scalar);		\
  }							\
  return
#define FLGR_MACRO_VECTOR_DIADIC_OP(dtype,arithop)	\
  int k;						\
  int spp = vctdest->spp;				\
  dtype *p1 = (dtype*) vct1->array;			\
  dtype *p2 = (dtype*) vct2->array;			\
  dtype *pdest = (dtype*) vctdest->array;		\
  dtype a1,a2;						\
							\
  				\
							\
  for(k=0 ; k<spp ; k++) {				\
    a1 = flgr_get_array_##dtype(p1,k);			\
    a2 = flgr_get_array_##dtype(p2,k);			\
    flgr_set_array_##dtype(pdest,k,arithop(a1,a2));	\
  }							\
  return
#define FLGR_MACRO_GET_NHB_CONV_2D(dtype)			\
  int i,k;							\
  int spp = extr->spp;						\
  dtype *presult = (dtype *) result->array;			\
  dtype *list_data_val;						\
  dtype *list_nhb_val;						\
  int *size = extr->size;					\
  fgFLOAT64 a,b,sum;						\
  fgFLOAT64 tmp;						\
								\
  								\
								\
  for(k=0 ; k<spp ; k++) {					\
    list_data_val = (dtype *) extr->list_data_val[k];		\
    list_nhb_val = (dtype *) extr->list_nhb_val[k];		\
								\
    tmp=0;							\
    sum=0;							\
								\
    for(i=0 ; i<size[k] ; i++){					\
      a = (fgFLOAT64) list_data_val[i];				\
      b = (fgFLOAT64) list_nhb_val[i];				\
      tmp = tmp + (a*b);					\
      sum = sum + (fabs(b));					\
								\
    }								\
								\
    if(sum!=0)							\
      flgr_set_array_##dtype(presult,k,(dtype) (tmp/sum));	\
  }								\
								\
  return

void flgr2d_get_nhb_convolution_fgUINT16(FLGR_Vector *result, FLGR_NhbBox2D *extr) {
  FLGR_MACRO_GET_NHB_CONV_2D(fgUINT16);
}

  fgUINT16 flgr2d_get_data_array_fgUINT16(fgUINT16** array, int row, int col) {
    return flgr_get_array_fgUINT16(array[row],col);
  }
  void flgr_get_data_array_vector_fgUINT16(fgUINT16 *vector_array, fgUINT16 *data_array, int spp, int pos) {
    register fgUINT16 val;
    register int i,k;

    for(k=0,i=pos*spp ; k<spp ; k++,i++) {
      val = flgr_get_array_fgUINT16(data_array,i);
      flgr_set_array_fgUINT16(vector_array,k,val);
    }
  }
  fgUINT16 flgr_get_array_fgUINT16(fgUINT16* array, int pos) {
    return array[pos];
  }
  void flgr_set_array_fgUINT16(fgUINT16* array, int pos, fgUINT16 value) {
    array[pos]=value;
  }

  void flgr_set_data_array_vector_fgUINT16(fgUINT16 *data_array, fgUINT16 *vector_array, int spp, int pos) {
    register fgUINT16 val;
    register int i,k;

    for(k=0,i=pos*spp ; k<spp ; k++,i++) {
      val = flgr_get_array_fgUINT16(vector_array,k);
      flgr_set_array_fgUINT16(data_array,i,val);
    }
  }
#if 0
 void *memcpy(void *dest, const void *src, size_t n)
{
	size_t i;
	char *d=(char*)dest;
	char *s=(char*)src;
	for(i=0;i<n;i++)
		d[i]=s[i];
	return dest;
}
void *memset(void *s, int c, size_t n)
{
	size_t i;
	char * t = (char*)s;
	for(i=0;i<n;i++)
		t[i]=c;
	return s;

}
#endif
  static __inline__ fgUINT16 flgr_defop_sup_fgUINT16(fgUINT16 a,fgUINT16 b) {
    return (a<b?b:a);
  }
void flgr_vector_sup_fgUINT16(FLGR_Vector *vctdest, FLGR_Vector *vct1,FLGR_Vector *vct2) {
  FLGR_MACRO_VECTOR_DIADIC_OP(fgUINT16,flgr_defop_sup_fgUINT16);
}
  static __inline__ fgUINT16 flgr_defop_mult_fgUINT16(fgUINT16 a,fgUINT16 b) {
    return a*b;
  }

void flgr_vector_populate_from_scalar_fgUINT16(FLGR_Vector *vctdest, fgUINT16 scalar) {
  FLGR_MACRO_FLGR_VECTOR_POPULATE_SCALAR(fgUINT16);
}
void flgr_vector_mult_fgUINT16(FLGR_Vector *vctdest, FLGR_Vector *vct1,FLGR_Vector *vct2) {
  FLGR_MACRO_VECTOR_DIADIC_OP(fgUINT16,flgr_defop_mult_fgUINT16);
}
FLGR_Ret flgr_vector_destroy(FLGR_Vector *vct) {
  

  if(vct == NULL) {
    POST_ERROR("Null object\n");
    return FLGR_RET_NULL_OBJECT;
  }

  if(vct->array == NULL) {
    POST_ERROR("Vector array is null\n");
    return FLGR_RET_NULL_OBJECT;
  }

  flgr_free_align(vct->array);

  flgr_free(vct);
 
  

  return FLGR_RET_OK;
}
FLGR_Ret flgr_is_vector_type_valid(FLGR_Type type) {
  

  if(type==FLGR_UINT8) {
    return FLGR_RET_OK;
  }else if(type==FLGR_UINT16) {
    return FLGR_RET_OK;
  }else if(type==FLGR_UINT32) {
    return FLGR_RET_OK;
  }else if(type==FLGR_UINT64) {
    return FLGR_RET_OK;
  }else if(type==FLGR_INT8) {
    return FLGR_RET_OK;
  }else if(type==FLGR_INT16) {
    return FLGR_RET_OK;
  }else if(type==FLGR_INT32) {
    return FLGR_RET_OK;
  }else if(type==FLGR_INT64) {
    return FLGR_RET_OK;
  }else if(type==FLGR_FLOAT32) {
    return FLGR_RET_OK;
  }else if(type==FLGR_FLOAT64) {
    return FLGR_RET_OK;
  }else if(type==FLGR_BIT) {
    return FLGR_RET_OK;
  }else return FLGR_RET_TYPE_UNKNOWN;
}
FLGR_Vector *flgr_vector_create(int spp, FLGR_Type type){
  FLGR_Vector *vct;

  

  if(flgr_is_vector_type_valid(type)!=FLGR_RET_OK) {
    POST_ERROR("unknwon type\n");
    return NULL;
  }
  
  if(spp<1) {
    POST_ERROR("Sample Per Pixel (spp) cannot be < 1\n");
    return NULL; 
  }

  vct = flgr_malloc(sizeof(FLGR_Vector));

  vct->bps = flgr_get_bps_from_type(type);
  vct->spp = spp;
  vct->type = type;

  vct->array = flgr_malloc_align((spp*vct->bps)/8+16,16);

  if(vct->array == NULL) {
    POST_ERROR("Allocation error !\n");
    flgr_free(vct);
    return NULL;
  }

  return vct;
}
int flgr_normalize_coordinate(int axis_coord, int axis_length) {
  int axis_true = (-1*axis_coord-1);
  int axis_false;
  int axis_test;

  

  axis_coord = ((axis_coord<0) ? (axis_true) : axis_coord);

  axis_test = ((axis_coord/axis_length)%2);
  axis_true = axis_length - (axis_coord%axis_length) - 1;
  axis_false = axis_coord%axis_length;

  axis_coord = ((axis_coord>=axis_length) && (axis_test)) ? (axis_true) : (axis_coord);
  axis_coord = ((axis_coord>=axis_length) && !(axis_test)) ? (axis_false) : (axis_coord);

/*   if(axis_coord<0) { */
/*     axis_coord=-1*axis_coord-1; */
/*   } */

/*   if( axis_coord >= axis_length ) { */
/*     if( ((axis_coord/axis_length)%2) == 1 ) */
/*       axis_coord = axis_length - (axis_coord%axis_length) - 1; */
/*     else */
/*       axis_coord = axis_coord%axis_length; */
/*   } */

  return axis_coord;
}
void *flgr_malloc(size_t size) {
  void *tmp = malloc(size);

  

  if(tmp==NULL) {
    POST_ERROR("Could not allocate data, returning NULL pointer !\n");
    return NULL;
  }

  return tmp;
}
FLGR_Ret flgr_is_data_type_valid(FLGR_Type type) {
  

  if(type==FLGR_UINT8) {
    return FLGR_RET_OK;
  }else if(type==FLGR_UINT16) {
    return FLGR_RET_OK;
  }else if(type==FLGR_UINT32) {
    return FLGR_RET_OK;
  }else if(type==FLGR_INT8) {
    return FLGR_RET_OK;
  }else if(type==FLGR_INT16) {
    return FLGR_RET_OK;
  }else if(type==FLGR_INT32) {
    return FLGR_RET_OK;
  }else if(type==FLGR_FLOAT32) {
    return FLGR_RET_OK;
  }else if(type==FLGR_FLOAT64) {
    return FLGR_RET_OK;
  }else if(type==FLGR_BIT) {
    return FLGR_RET_OK;
  }else return FLGR_RET_TYPE_UNKNOWN;
}
FLGR_Type flgr_get_type_from_string(char *type) {
  

  if(strcmp(type,"fgBIT")==0) return FLGR_BIT;
  if(strcmp(type,"fgUINT8")==0) return FLGR_UINT8;
  if(strcmp(type,"fgUINT16")==0) return FLGR_UINT16;
  if(strcmp(type,"fgUINT32")==0) return FLGR_UINT32;
  if(strcmp(type,"fgUINT64")==0) return FLGR_UINT64;
  if(strcmp(type,"fgINT8")==0) return FLGR_INT8;
  if(strcmp(type,"fgINT16")==0) return FLGR_INT16;
  if(strcmp(type,"fgINT32")==0) return FLGR_INT32;
  if(strcmp(type,"fgINT64")==0) return FLGR_INT64;
  if(strcmp(type,"fgFLOAT32")==0) return FLGR_FLOAT32;
  if(strcmp(type,"fgFLOAT64")==0) return FLGR_FLOAT64;

  POST_ERROR("Unknown type %s\n",type);
  return FLGR_UINT8;
}
int flgr_get_bps_from_type(FLGR_Type type) {
  

  switch(type) {
  case FLGR_BIT     : return 1;
  case FLGR_UINT8   : return (sizeof(fgUINT8))<<3;
  case FLGR_UINT16  : return (sizeof(fgUINT16))<<3;
  case FLGR_UINT32  : return (sizeof(fgUINT32))<<3;
  case FLGR_UINT64  : return (sizeof(fgUINT64))<<3;
  case FLGR_INT8    : return (sizeof(fgINT8))<<3;
  case FLGR_INT16   : return (sizeof(fgINT16))<<3;
  case FLGR_INT32   : return (sizeof(fgINT32))<<3;
  case FLGR_INT64   : return (sizeof(fgINT64))<<3;
  case FLGR_FLOAT32 : return (sizeof(fgFLOAT32))<<3;
  case FLGR_FLOAT64 : return (sizeof(fgFLOAT64))<<3;
  default:
    return FLGR_RET_TYPE_UNKNOWN;
  }

}
void flgr_free(void *ptr) {
  

  free(ptr);
}
void flgr_backtrace_print(void) {
#if defined(__GNUC__)
  void *array[128];
  size_t size;
  char **strings;
  size_t i;
     
  size = backtrace(array, 128);
  strings = backtrace_symbols(array, size);
     
  fprintf(stderr,"\tObtained %zd stack frames :\n", size);
     
  for (i = 0; i < size; i++)
    fprintf(stderr,"[BACKTRACE]\t\t%s\n", strings[i]);
     
  free(strings);
#endif
  return;
}
void flgr2d_set_data_vector_fgUINT16(FLGR_Data2D *dat, int row, int col, FLGR_Vector *vct) {
  FLGR_MACRO_SET_DATA2D_VECTOR(fgUINT16);
}
void flgr2d_native_dilate_8_connexity_fgUINT16(FLGR_Data2D *nhb) {
  FLGR_MACRO_NATIVE_DILATE(fgUINT16,1,1,1,1,1,1,1,1,1);
}
void flgr2d_native_dilate_6_connexity_fgUINT16(FLGR_Data2D *nhb) {
  FLGR_MACRO_NATIVE_DILATE(fgUINT16,0,1,1,1,1,1,0,1,1);
}
void flgr2d_get_data_vector_no_norm_fgUINT16(FLGR_Data2D *dat, int row, int col, FLGR_Vector *vct) {
  FLGR_MACRO_GET_DATA2D_VECTOR_NO_NORM(fgUINT16);
}
void flgr2d_native_dilate_4_connexity_fgUINT16(FLGR_Data2D *nhb) {
  FLGR_MACRO_NATIVE_DILATE(fgUINT16,0,1,0,1,1,1,0,1,0);
}
FLGR_Ret flgr2d_mirror_vertical_hmorph(FLGR_Data2D *dat) {
  FLGR_Data1D *row_tmp;
  void *tmp;
  int i,j;

  

  if(dat==NULL) {
    POST_ERROR("Null objects!\n");
    return FLGR_RET_NULL_OBJECT;
  }

  for(i=0,j=dat->size_y-1; i<dat->size_y/2 ; i++,j--) {
    tmp=dat->array[i];
    dat->array[i]=dat->array[j];
    dat->array[j]=tmp;

    row_tmp = dat->row[i];
    dat->row[i] = dat->row[j];
    dat->row[j] = row_tmp;

    dat->row[i]->ref2d=i;
    dat->row[j]->ref2d=j;

  }

  return FLGR_RET_OK;
}

void flgr2d_mirror_horizontal_fgUINT16(FLGR_Data2D *datdest, FLGR_Data2D *datsrc) {
  FLGR_MACRO_MIRROR2D_HORIZONTAL(fgUINT16);
}
//////////////////////////////////////////////////////////////////
/*! mirror horizontally an image (regarding a vertical axis ...)
 *  @param datdest : a pointer to FLGR_Data2D
 *  @param datsrc  : a pointer to FLGR_Data2D
 *  @returns FLGR_RET_OK, ...
 */
//////////////////////////////////////////////////////////////////
FLGR_Ret flgr2d_mirror_horizontal(FLGR_Data2D *datdest, FLGR_Data2D *datsrc) {
  FLGR_Ret ret;
  


  if((datdest == NULL) || (datsrc == NULL)) {
    POST_ERROR("Null objects\n");
    return FLGR_RET_NULL_OBJECT;
  }

  ret = flgr2d_is_data_same_attributes(datdest, datsrc, __FUNCTION__);
  if(ret != FLGR_RET_OK) return ret;
  
flgr2d_mirror_horizontal_fgUINT16(datdest, datsrc);
//  FLGR_DISPATCH_PROCEDURE(datdest->type, flgr2d_mirror_horizontal, datdest, datsrc);
}
FLGR_Ret flgr2d_is_data_same_type(FLGR_Data2D *data1, FLGR_Data2D *data2) {
  FLGR_Ret ret;
  


  if((data1==NULL) || (data2==NULL)) {
    POST_ERROR("Null objects!\n");
    return FLGR_RET_NULL_OBJECT;
  }

  if( (ret=flgr_is_data_type_valid(data1->type)) != FLGR_RET_OK ) return ret;
  if( (ret=flgr_is_data_type_valid(data2->type)) != FLGR_RET_OK ) return ret;

  if(data1->type != data2->type) {
    return FLGR_RET_TYPE_DIFFERENT;
  }

  return FLGR_RET_OK;
}

FLGR_Ret flgr2d_is_data_same_spp(FLGR_Data2D *dat1, FLGR_Data2D *dat2) {
  

  if((dat1==NULL) || (dat2==NULL)) {
    POST_ERROR("Null objects!\n");
    return FLGR_RET_NULL_OBJECT;
  }


  if(dat1->spp <0) return  FLGR_RET_VECTOR_SIZE_DIFFERENT;
  if(dat2->spp <0) return FLGR_RET_VECTOR_SIZE_DIFFERENT;

  if(dat1->spp != dat2->spp) return FLGR_RET_VECTOR_SIZE_DIFFERENT;

  return FLGR_RET_OK;
}
FLGR_Ret flgr2d_is_data_same_size(FLGR_Data2D *data1, FLGR_Data2D *data2) {

  

  if((data1==NULL) || (data2==NULL)) {
    POST_ERROR("Null objects!\n");
    return FLGR_RET_NULL_OBJECT;
  }

  if((data1->size_y!=data2->size_y)||(data1->size_x!=data2->size_x)) {
    return FLGR_RET_SIZE_ERROR;
  }

  return FLGR_RET_OK;
}
FLGR_Ret flgr2d_is_data_same_attributes(FLGR_Data2D *data1, FLGR_Data2D *data2, const char *callingFunction) {
  FLGR_Ret ret;

  

  if((data1==NULL) || (data2==NULL)) {
    POST_ERROR("Null objects!\n");
    return FLGR_RET_NULL_OBJECT;
  }

  if( (ret=flgr_is_data_type_valid(data1->type))!=FLGR_RET_OK) {
    EPRINTF("ERROR: Function %s: unknown type\r\n", callingFunction);
    return ret;
  }

  if( (ret=flgr_is_data_type_valid(data2->type))!=FLGR_RET_OK) {
    EPRINTF("ERROR: Function %s: unknown type\r\n", callingFunction);
    return ret;
  }

  if((ret=flgr2d_is_data_same_type(data1, data2))!=FLGR_RET_OK) {
    EPRINTF("ERROR: Function %s: source and destination have a different type\r\n", callingFunction);
    return ret;
  }

  if((ret=flgr2d_is_data_same_spp(data1, data2))!=FLGR_RET_OK) {
    EPRINTF("ERROR: Function %s: source and destination have a different vector size(spp)\r\n", callingFunction);
    return ret;
  }

  if((data1->size_y!=data2->size_y)||(data1->size_x!=data2->size_x)) {
    EPRINTF("ERROR: Function %s: source and destination have a different size\r\n", callingFunction);
    return FLGR_RET_SIZE_ERROR;
  }

  return FLGR_RET_OK;
}
void flgr1d_import_raw_fgUINT16(FLGR_Data1D *datdest, void *raw) {
  FLGR_Data1D datsrc[1];
  

  datsrc->bps=datdest->bps;
  datsrc->spp=datdest->spp;
  datsrc->length=datdest->length;
  datsrc->array=(fgUINT16*)raw;

  if(raw!=NULL) {
    FLGR_MACRO_COPY1D_SAME_TYPE;
  }
}
void flgr2d_import_raw_fgUINT16(FLGR_Data2D *datdest, void* raw) {
  FLGR_MACRO_IMPORT2D_RAW(fgUINT16);
}
FLGR_Ret flgr2d_import_raw_ptr(FLGR_Data2D *datdest, void* raw) {
  

  if((datdest==NULL)) {
    POST_ERROR("Null objects!\n");
    return FLGR_RET_NULL_OBJECT;
  }
  flgr2d_import_raw_fgUINT16(datdest,raw);

//  FLGR_DISPATCH_PROCEDURE(datdest->type,flgr2d_import_raw,datdest,raw);
}
void flgr2d_get_data_vector_fgUINT16(FLGR_Data2D *dat, int row, int col, FLGR_Vector *vct) {
  FLGR_MACRO_GET_DATA2D_VECTOR(fgUINT16);
}
void flgr2d_fill_nhb_odd_rows_fgUINT16(FLGR_Data2D *datdest, FLGR_Data2D *datsrc) {
  FLGR_MACRO_FILL_NHB_ODD_ROWS(fgUINT16);
}
FLGR_Ret flgr2d_fill_nhb_odd_rows(FLGR_Data2D *datdest, FLGR_Data2D *datsrc) {
  FLGR_Ret ret;

  

  if((datsrc==NULL) || (datdest==NULL)) {
    POST_ERROR("Null objects!\n");
    return FLGR_RET_NULL_OBJECT;
  }

  ret=flgr2d_is_data_same_attributes(datdest,datsrc,__FUNCTION__);
  if(ret != FLGR_RET_OK) return ret;

  flgr2d_fill_nhb_odd_rows_fgUINT16(datdest,datsrc);
 // FLGR_DISPATCH_PROCEDURE(datdest->type,flgr2d_fill_nhb_odd_rows,datdest,datsrc);
}
FLGR_Ret flgr2d_fill_nhb_even_rows(FLGR_Data2D *datdest, FLGR_Data2D *datsrc) {
  FLGR_Ret ret;

  

  if((datsrc==NULL) || (datdest==NULL)) {
    POST_ERROR("Null objects!\n");
    return FLGR_RET_NULL_OBJECT;
  }

  ret=flgr2d_is_data_same_attributes(datdest,datsrc,__FUNCTION__);
  if(ret != FLGR_RET_OK) return ret;

  flgr2d_fill_nhb_even_rows_fgUINT16(datdest,datsrc);
  //FLGR_DISPATCH_PROCEDURE(datdest->type,flgr2d_fill_nhb_even_rows,datdest,datsrc);
}
int flgr2d_data_is_connexity(FLGR_Data2D *data, FLGR_Connexity connexity) {
  

  if(data==NULL) {
    POST_ERROR("Null objects!\n");
    return 0;
  }

  return (data->connexity==connexity);
}
FLGR_Ret flgr2d_fill_nhbs_for_6_connexity(FLGR_Data2D *nhbEven, FLGR_Data2D *nhbOdd, FLGR_Data2D *nhb, int SYM) {
  FLGR_Ret ret;

  

  if((nhb==NULL) || (nhbOdd==NULL) || (nhbEven==NULL)) {
    POST_ERROR("Null objects!\n");
    return FLGR_RET_NULL_OBJECT;
  }


  if((ret=flgr2d_is_data_same_attributes(nhbEven,nhbOdd,__FUNCTION__))!=FLGR_RET_OK) {
    return ret;
  }

  if((ret=flgr2d_is_data_same_attributes(nhbEven,nhb,__FUNCTION__))!=FLGR_RET_OK) {
    return ret;
  }

  if(SYM==FLGR_NHB_SYM) {
    if(flgr2d_data_is_connexity(nhb,FLGR_6_CONNEX)==FLGR_TRUE) {
      flgr2d_mirror_horizontal(nhbOdd,nhb);
      flgr2d_mirror_vertical_hmorph(nhbOdd);
      flgr2d_fill_nhb_even_rows(nhbEven,nhbOdd);
    }else {
      flgr2d_mirror_horizontal(nhbEven,nhb);
      flgr2d_mirror_vertical_hmorph(nhbEven);
      flgr2d_copy(nhbOdd,nhbEven);
    }
  }else {
    if(flgr2d_data_is_connexity(nhb,FLGR_6_CONNEX)==FLGR_TRUE) {
      flgr2d_copy(nhbEven,nhb);
      flgr2d_fill_nhb_odd_rows(nhbOdd,nhbEven);
    }else {
      flgr2d_copy(nhbEven,nhb);
      flgr2d_copy(nhbOdd,nhb);
    }
  }

  return FLGR_RET_OK;
}
void flgr2d_fill_nhb_even_rows_fgUINT16(FLGR_Data2D *datdest, FLGR_Data2D *datsrc) {
  FLGR_MACRO_FILL_NHB_EVEN_ROWS(fgUINT16);
}
void flgr2d_draw_vertical_line_fgUINT16(FLGR_Data2D *dat, int x, int y, int size_y, FLGR_Vector *color) {
  FLGR_MACRO_DRAW_2D_VERT_LINE(fgUINT16);
}
void flgr2d_draw_point_fgUINT16(FLGR_Data2D *dat, int x, int y, FLGR_Vector *color) {
  FLGR_MACRO_DRAW2D_POINT(fgUINT16);
}
void dataExchange(int* x, int* y)
{
  int t = *x;
  *x = *y;
  *y = t;
}
void flgr2d_draw_line_fgUINT16(FLGR_Data2D *dat, int x1, int y1, int x2, int y2, FLGR_Vector *color) {
  FLGR_MACRO_DRAW_2D_LINE(fgUINT16);
}
void flgr2d_draw_horizontal_line_fgUINT16(FLGR_Data2D *dat, int x, int y, int size_x, FLGR_Vector *color) {
  FLGR_MACRO_DRAW_2D_HORZ_LINE(fgUINT16);
}
void flgr2d_draw_filled_rectangle_fgUINT16(FLGR_Data2D *dat, int x, int y, int size_x, int size_y, FLGR_Vector *color) {
  FLGR_MACRO_DRAW_2D_FILLED_RECT(fgUINT16);
}
void flgr2d_draw_filled_ellipse_fgUINT16(FLGR_Data2D *dat, int cx, int cy, int a, int b, FLGR_Vector *color) {
  FLGR_MACRO_DRAW_2D_2D_FILLED_ELLIPSE(fgUINT16);
}
void flgr2d_draw_disc_fgUINT16(FLGR_Data2D *dat, int cx, int cy, int radius, FLGR_Vector *color) {
  FLGR_MACRO_DRAW_2D_2D_DISC(fgUINT16);
}
FLGR_Data2D *flgr2d_create_from(FLGR_Data2D *datsrc) {
  

  if(datsrc==NULL) {
    POST_ERROR("Null objects!\n");
    return NULL;
  }
  return flgr2d_create(datsrc->size_y,datsrc->size_x,datsrc->spp,datsrc->type,datsrc->shape,datsrc->connexity);
}
void flgr1d_copy_fgUINT16_fgUINT16(FLGR_Data1D *datdest, FLGR_Data1D *datsrc) {
  FLGR_MACRO_COPY1D_SAME_TYPE;
}
void flgr2d_copy_fgUINT16_fgUINT16(FLGR_Data2D *datdest, FLGR_Data2D *datsrc) {
  FLGR_MACRO_COPY2D(fgUINT16,fgUINT16);
}
FLGR_Ret flgr2d_copy(FLGR_Data2D *datdest, FLGR_Data2D *datsrc) {
  int error=0;
  FLGR_Ret ret;

  

  if((datdest==NULL) || (datsrc==NULL)) {
    POST_ERROR("Null objects!\n");
    return FLGR_RET_NULL_OBJECT;
  }

  if((ret=flgr2d_is_data_same_size(datsrc, datdest))!=FLGR_RET_OK) {
    POST_ERROR("src and dest dat have different size !\n");
    return ret;
  }

  /*
  if(datsrc->type==FLGR_UINT8) {
    FLGR_MACRO_COPY2D_SRC_FIX_TYPE(fgUINT8,fgBIT,fgUINT8,fgUINT16,fgUINT32,
				   fgINT8,fgINT16,fgINT32,fgFLOAT32,fgFLOAT64);

  }else*/ if(datsrc->type==FLGR_UINT16) {
    flgr2d_copy_fgUINT16_fgUINT16( datdest , datsrc);

  }/*else if(datsrc->type==FLGR_UINT32) {
    FLGR_MACRO_COPY2D_SRC_FIX_TYPE(fgUINT32,fgBIT,fgUINT8,fgUINT16,fgUINT32,
				   fgINT8,fgINT16,fgINT32,fgFLOAT32,fgFLOAT64);

  }else if(datsrc->type==FLGR_INT8) {
    FLGR_MACRO_COPY2D_SRC_FIX_TYPE(fgINT8,fgBIT,fgUINT8,fgUINT16,fgUINT32,
				   fgINT8,fgINT16,fgINT32,fgFLOAT32,fgFLOAT64);

  }else if(datsrc->type==FLGR_INT16) {
    FLGR_MACRO_COPY2D_SRC_FIX_TYPE(fgINT16,fgBIT,fgUINT8,fgUINT16,fgUINT32,
				   fgINT8,fgINT16,fgINT32,fgFLOAT32,fgFLOAT64);

  }else if(datsrc->type==FLGR_INT32) {
    FLGR_MACRO_COPY2D_SRC_FIX_TYPE(fgINT32,fgBIT,fgUINT8,fgUINT16,fgUINT32,
				   fgINT8,fgINT16,fgINT32,fgFLOAT32,fgFLOAT64);

  }else if(datsrc->type==FLGR_FLOAT32) {
    FLGR_MACRO_COPY2D_SRC_FIX_TYPE(fgFLOAT32,fgBIT,fgUINT8,fgUINT16,fgUINT32,
				   fgINT8,fgINT16,fgINT32,fgFLOAT32,fgFLOAT64);

  }else if(datsrc->type==FLGR_FLOAT64) {
    FLGR_MACRO_COPY2D_SRC_FIX_TYPE(fgFLOAT64,fgBIT,fgUINT8,fgUINT16,fgUINT32,
				   fgINT8,fgINT16,fgINT32,fgFLOAT32,fgFLOAT64);

  }else if(datsrc->type==FLGR_BIT) {
    FLGR_MACRO_COPY2D_SRC_FIX_TYPE(fgBIT,fgBIT,fgUINT8,fgUINT16,fgUINT32,
				   fgINT8,fgINT16,fgINT32,fgFLOAT32,fgFLOAT64);

  }*/else {
    error=1;
  }

  if(error==1) {
    POST_ERROR("dest data type unknown!\n");
    return FLGR_RET_TYPE_UNKNOWN;
  }

  return FLGR_RET_OK;

}

FLGR_Ret flgr1d_clear_all(FLGR_Data1D *data) {
  

  if(data==NULL) {
    POST_ERROR("Null objects!\n");
    return FLGR_RET_NULL_OBJECT;
  }

  memset(data->array, 0, (data->length*data->bps*data->spp)/8+32);
  return FLGR_RET_OK;
}
FLGR_Ret flgr2d_clear_all(FLGR_Data2D *data) {
  int i;

  

  if(data==NULL) {
    POST_ERROR("Null objects!\n");
    return FLGR_RET_NULL_OBJECT;
  }


  for(i=0 ; i<data->size_y ; i++) flgr1d_clear_all(data->row[i]);

  return FLGR_RET_OK;
}
void flgr2d_fill_neighborhood_fgUINT16(FLGR_Data2D *nhb, FLGR_Shape shape, int width, int height) {
  FLGR_MACRO_FILL_2D_SHP(fgUINT16);
}
FLGR_Ret flgr2d_fill_neighborhood(FLGR_Data2D *nhb, FLGR_Shape shape, int width, int height) {
  if(nhb==NULL) {
    POST_ERROR("Null objects!\n");
    return FLGR_RET_NULL_OBJECT;
  }

  if(nhb->shape!=shape) {
    POST_WARNING("given shape does not correspond to FLGR_Data2D shape\n");
  }

  if(width>nhb->size_x) {
    POST_WARNING("width too high\n");
    width=nhb->size_x;
  }

  if(width<0) {
    POST_WARNING("width < 0\n");
    width=0;
  }

  if(width%2==0) {
    width++;
  }

  if(height>nhb->size_y) {
    POST_WARNING("height too high\n");
    height=nhb->size_y;
  }

  if(height<0) {
    POST_WARNING("height < 0\n");
    height=0;
  }

  if(height%2==0) {
    height++;
  }



	flgr2d_fill_neighborhood_fgUINT16(nhb,shape,width,height); 
  //FLGR_DISPATCH_PROCEDURE(nhb->type,flgr2d_fill_neighborhood,nhb,shape,width,height);

}
void flgr2d_destroy_neighbor_box(FLGR_NhbBox2D *extr) {
  int k;
  

  if(extr==NULL) {
    POST_ERROR("Null objects!\n");
    return;
  }

  for(k=0 ; k<extr->spp ; k++) {
    flgr_free(extr->list_data_val[k]);
    flgr_free(extr->list_nhb_val[k]);
    flgr_free(extr->list_coord_x[k]);
    flgr_free(extr->list_coord_y[k]);
  }

 
  flgr_vector_destroy(extr->center_data_val);
  flgr_vector_destroy(extr->center_nhb_val);

  flgr_free(extr->list_coord_x);
  flgr_free(extr->list_coord_y);
  flgr_free(extr->list_data_val);
  flgr_free(extr->list_nhb_val);
  flgr_free(extr->size);

  flgr_free(extr);
}
FLGR_Ret flgr2d_destroy_link(FLGR_Data2D *dat) {
  int k;

  

  if(dat==NULL) {
    POST_ERROR("Null objects!\n");
    return FLGR_RET_NULL_OBJECT;
  }

  if(dat->link_overlap == -1) {
    POST_WARNING("Destroying a non-linked data, branching to flgr2d_destroy\n");
    return flgr2d_destroy(dat);
  }

  if(dat->link_position == 0) {
    for(k=dat->size_y-dat->link_overlap ; k<dat->size_y+16 ; k++) {
      flgr1d_destroy(dat->row[k]);
    }

  }else if( dat->link_position == (dat->link_number - 1)  ) {
    for(k=0 ; k<dat->link_overlap ; k++) {
      flgr1d_destroy(dat->row[k]);
    }
    
    for(k=dat->size_y ; k<dat->size_y+16 ; k++) {
      flgr1d_destroy(dat->row[k]);
    }
    
   }else {
    for(k=0 ; k<dat->link_overlap ; k++) {
      flgr1d_destroy(dat->row[k]);
    }
    for(k=dat->size_y-dat->link_overlap ; k<dat->size_y+16 ; k++) {
      flgr1d_destroy(dat->row[k]);
    }
  }

  flgr_free(dat->array);
  flgr_free(dat->row);
  flgr_free(dat);

  return FLGR_RET_OK;
}
FLGR_Ret flgr1d_destroy(FLGR_Data1D *dat) {
  

  if(dat==NULL) {
    POST_ERROR("Null objects!\n");
    return FLGR_RET_NULL_OBJECT;
  }

  flgr_free_align(dat->array_phantom);
  flgr_free(dat);

  return FLGR_RET_OK;
}
FLGR_Ret flgr2d_destroy(FLGR_Data2D *dat) {
  int i;

  

  if(dat==NULL) {
    POST_ERROR("Null objects!\n");
    return FLGR_RET_NULL_OBJECT;
  }

  if(dat->link_overlap > -1) {
    POST_WARNING("Destroying a linked data, branching to flgr2d_destroy_link\n");
    return flgr2d_destroy_link(dat);
  }

  for(i=0 ; i<dat->size_y+16 ; i++) {
    flgr1d_destroy(dat->row[i]);
  }

  flgr_free(dat->array);
  flgr_free(dat->row);
  flgr_free(dat);

  return FLGR_RET_OK;
}
FLGR_Data2D *flgr2d_create_pixmap(int size_y, int size_x, int spp, FLGR_Type type) {
  

  return flgr2d_create(size_y, size_x, spp, type, FLGR_NO_SHAPE, FLGR_NO_CONNEX);
}

FLGR_Data2D *flgr2d_create_neighborhood_from(FLGR_Data2D *nhbsrc) {
  

  if(nhbsrc==NULL) {
    POST_ERROR("Null objects!\n");
    return NULL;
  }
  return flgr2d_create_neighborhood(nhbsrc->size_y, nhbsrc->size_x, nhbsrc->spp,
				    nhbsrc->type, nhbsrc->shape, nhbsrc->connexity);
}

FLGR_Data1D *flgr1d_create_fgUINT16(int length, int spp, FLGR_Shape shape) {
  FLGR_MACRO_CREATE1D(fgUINT16, FLGR_UINT16);
}

FLGR_Data2D *flgr2d_create_fgUINT16(int size_y, int size_x, int spp, FLGR_Shape shape, FLGR_Connexity connexity) {
  FLGR_MACRO_CREATE2D(fgUINT16, FLGR_UINT16);
}

FLGR_Data2D *flgr2d_create(int size_y, int size_x, int spp, FLGR_Type type, FLGR_Shape shape, FLGR_Connexity connexity) {
  

  if(type==FLGR_UINT16)  return flgr2d_create_fgUINT16(size_y, size_x, spp, shape, connexity);
  POST_ERROR("Type unknown!\n");
  return NULL;

}
FLGR_Data2D *flgr2d_create_neighborhood(int size_y, int size_x, int spp, FLGR_Type type,
					FLGR_Shape shape, FLGR_Connexity connexity) {
  FLGR_Data2D *nhb;

  

  if((size_x%2)==0) {
    POST_WARNING("Warning NhbWidth(%d) is even! Changing to the next odd value (%d) \n", size_x, size_x+1);
    size_x++;
  }
  if((size_y%2)==0) {
    POST_WARNING("Warning NhbWidth(%d) is even! Changing to the next odd value (%d) \n", size_y, size_y+1);
    size_y++;
  }

  if( (connexity != FLGR_4_CONNEX) && (connexity != FLGR_6_CONNEX) && (connexity != FLGR_8_CONNEX)) {
    POST_ERROR("bad connexity : %d\n", connexity);
    return NULL;
  }

  if((nhb=flgr2d_create(size_y, size_x, spp, type, shape, connexity))==NULL) return NULL;

  if(flgr2d_fill_neighborhood(nhb, shape, size_x,size_y)!=FLGR_RET_OK) {
    flgr2d_destroy(nhb);
    return NULL;
  }

  return nhb;

}
int flgr_get_sizeof(FLGR_Type type) {
  

  switch(type) {
  case FLGR_BIT     : return sizeof(fgBIT);
  case FLGR_UINT8   : return sizeof(fgUINT8);
  case FLGR_UINT16  : return sizeof(fgUINT16);
  case FLGR_UINT32  : return sizeof(fgUINT32);
  case FLGR_UINT64  : return sizeof(fgUINT64);
  case FLGR_INT8    : return sizeof(fgINT8);
  case FLGR_INT16   : return sizeof(fgINT16);
  case FLGR_INT32   : return sizeof(fgINT32);
  case FLGR_INT64   : return sizeof(fgINT64);
  case FLGR_FLOAT32 : return sizeof(fgFLOAT32);
  case FLGR_FLOAT64 : return sizeof(fgFLOAT64);
  default:
    return FLGR_RET_TYPE_UNKNOWN;
  }
}
FLGR_NhbBox2D *flgr2d_create_neighbor_box(FLGR_Data2D *data) {
  FLGR_NhbBox2D *tmp;
  int sizeMax;
  int typeSize,k;

  

  if(data==NULL) {
    POST_ERROR("Null objects!\n");
    return NULL;
  }
  
  typeSize = flgr_get_sizeof(data->type);
  sizeMax = data->size_x*data->size_y;

  if(typeSize < 1) return NULL;
  if(data->spp < 1) return NULL;



  tmp                   = flgr_malloc(sizeof(FLGR_NhbBox2D));

  tmp->type             = data->type;
  tmp->spp              = data->spp;

  tmp->center_data_val  = flgr_vector_create(data->spp, data->type);
  tmp->center_nhb_val   = flgr_vector_create(data->spp, data->type);
  tmp->center_coord_y   = 0;
  tmp->center_coord_x   = 0;
  tmp->nhb_size_y       = data->size_y;
  tmp->nhb_size_x       = data->size_x;

  tmp->list_coord_x     = (int **) flgr_malloc(sizeof(int*) * data->spp);
  tmp->list_coord_y     = (int **) flgr_malloc(sizeof(int*) * data->spp);
  tmp->list_data_val    = (void **) flgr_malloc(sizeof(void*) *data->spp);
  tmp->list_nhb_val     = (void **) flgr_malloc(sizeof(void*) *data->spp);

  tmp->size             = (int *) flgr_malloc(sizeof(int) * data->spp);

  for(k=0 ; k<data->spp ; k++) {
    tmp->list_coord_y[k] = flgr_malloc(sizeof(int) * sizeMax);
    tmp->list_coord_x[k] = flgr_malloc(sizeof(int) * sizeMax);
    tmp->list_data_val[k] = flgr_malloc(flgr_get_sizeof(tmp->type) * sizeMax);
    tmp->list_nhb_val[k] = flgr_malloc(flgr_get_sizeof(tmp->type) * sizeMax);
    tmp->size[k]=0;
  }


  return tmp;

}
void flgr2d_get_neighborhood_fgUINT16(FLGR_NhbBox2D *extr,
				      FLGR_Data2D *dat, FLGR_Data2D *nhb, int row, int col) {
  FLGR_MACRO_GET_NHB_2D_IN_DATA(fgUINT16);
}

void flgr2d_apply_raster_scan_method_fgUINT16(FLGR_Data2D *nhb) {
  FLGR_APPLY_RASTER_SCAN_METHOD_2D(fgUINT16);
}

void flgr2d_raster_slide_window_fgUINT16(FLGR_Data2D *imgdest, FLGR_Data2D *imgsrc, FLGR_Data2D *nhb, int nhb_sym,
					  const FLGR_ComputeNhb2D computeNhb) {
  FLGR_MACRO_RASTER_SLIDE_WINDOW_2D(fgUINT16,flgr2d_get_neighborhood);
}
void flgr2d_convolution_fgUINT16(FLGR_Data2D *datdest, FLGR_Data2D *datsrc, FLGR_Data2D *nhb) {
  
  flgr2d_raster_slide_window_fgUINT16(datdest, datsrc, nhb, FLGR_NHB_NO_SYM, flgr2d_get_nhb_convolution_fgUINT16);
}


FLGR_Ret flgr2d_convolution(FLGR_Data2D *datdest, FLGR_Data2D *datsrc, FLGR_Data2D *nhb) {
  FLGR_Ret ret;

  

  if((datdest==NULL) || (datsrc==NULL) || (nhb==NULL)){
    POST_ERROR("Null objects!\n");
    return FLGR_RET_NULL_OBJECT;
  }

  if((ret=flgr2d_is_data_same_attributes(datdest,datsrc,__FUNCTION__)) != FLGR_RET_OK) return ret;

  if(datdest->type==FLGR_UINT16) {
    flgr2d_convolution_fgUINT16(datdest,datsrc,nhb);
  }									\
  //FLGR_DISPATCH_PROCEDURE(datdest->type,flgr2d_convolution,datdest,datsrc,nhb);

}
