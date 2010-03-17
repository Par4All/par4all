#include <stdio.h>
//#include <flgrCoreData.h>
//#include <flgrCoreDataIO.h>
//#include <flgrCoreCopy.h>
//#include <flgrCoreNhbManage.h>
//#include <flgrLinearConvolution.h>
//#include <flgrLinearCorrelation.h>

#include "fulguro-included.h"

/* the test file */
int main(int argc, char*argv[])
{

	fgUINT16 rawsrc[]={105,115, 28, 41, 41, 48, 54, 57, 62, 70,		
		72, 76, 75, 76, 76, 78, 75, 77, 78, 76,		
		75, 79, 77, 76, 77, 73, 71, 64, 62, 55,		
		50, 44, 42, 32,123,112,100, 88, 82, 73,		
		73, 76, 76, 81, 85, 86, 90, 90, 93, 92,		
		91, 96, 96,100, 96, 98, 98, 97,102, 99,		
		98, 96, 99,102, 98, 93,100, 99, 94, 96,		
		94, 90, 88, 87, 88, 88, 85, 87, 35,  5};		

	fgUINT16 rawref[]={ 92, 78, 68, 56, 60, 62, 64, 67, 70, 71,		
		87, 78, 71, 63, 65, 65, 66, 66, 66, 67,		
		66, 65, 64, 72, 80, 87, 82, 77, 72, 71,		
		66, 65, 64, 74, 82, 90, 86, 82, 77, 76,		
		71, 71, 71, 81, 90, 98, 95, 93, 90, 90,		
		88, 89, 91, 92, 93, 93, 94, 95, 95, 96,		
		94, 94, 94, 94, 94, 93, 93, 88, 79, 71,		
		94, 94, 93, 93, 92, 92, 92, 83, 69, 57};		

	FLGR_Data2D *imgsrc, *imgref, *img, *nhb;				
	FLGR_Ret ret;								

	imgsrc= flgr2d_create_pixmap(8,10,1,flgr_get_type_from_string("fgUINT16")); 
	imgref= flgr2d_create_pixmap(8,10,1,flgr_get_type_from_string("fgUINT16")); 
	img =   flgr2d_create_pixmap(8,10,1,flgr_get_type_from_string("fgUINT16")); 
	nhb =   flgr2d_create_neighborhood(3, 3,1,				
			flgr_get_type_from_string("fgUINT16"),	
			FLGR_RECT, FLGR_8_CONNEX);		

	flgr2d_import_raw_ptr(imgsrc,rawsrc);					
	flgr2d_import_raw_ptr(imgref,rawref);					


	ret=flgr2d_convolution(img,imgsrc,nhb);				

	/*check_and_display_data2d(imgref,img,ret);				*/

	flgr2d_destroy(imgsrc);						
	flgr2d_destroy(imgref);						
	flgr2d_destroy(img);							
	flgr2d_destroy(nhb);							

	return 1;
}

