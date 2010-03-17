
///////////////////////////////////////////////////////////////////////////////
// 
// *****************************************************************************
// *
// * Project:  H264 / AVC Porting   
// *
// * TITLE:    Test_IQ_IT_Residual_Norm_Imp.c
// *
// * PURPOSE:  Test function for IQ IT Residual process  
// *           Same implementation as required in the H26L norm  
// *
// * HISTORY:
// *    Name         Date        Version        Object
// *  Fraleu S.   08/04/2004      0.00         Creation  
// *
// * COPYRIGHT:   2004, THOMSON 
// *
// *****************************************************************************
// 


#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>

int qP_i;
int qP_mod_6_i;
int qP_div_6_i;
int LeftShift_i;

typedef struct
{
  short p[4][4];
} Block_t;

char LevelScaleBaseMatrix_auc[6][3] =
{ 
  {10, 16, 13},
  {11, 18, 14}, 
  {13, 20, 16},
  {14, 23, 18},
  {16, 25, 20},
  {18, 29, 23},
};

Block_t LevelScale_t;  /*  [LevelScale(ij)] in the standard */
Block_t c_t;  /*  [c(ij)] in the standard */
Block_t d_t;  /*  [d(ij)] in the standard */
Block_t e_t;  /*  [e(ij)] in the standard */
Block_t f_t;  /*  [f(ij)] in the standard */
Block_t g_t;  /*  [g(ij)] in the standard */
Block_t h_t;  /*  [h(ij)] in the standard */
Block_t r_t;  /*  [r(ij)] in the standard */

Block_t MB_t[4][4];  


/* ---------------------------------------------------------- */
/* Display function in Decimal mode for a 4x4 block           */ 
/* ---------------------------------------------------------- */

int DisplayBlock_i(char * BlockName_ac, Block_t * BlockToDisplay_at)
{
  int i = 0, j = 0;
  printf("---- %s ----\n", BlockName_ac);
  for (i=0;i<4; i++) {
    for (j=0;j<4; j++) {
      printf("%08d ", BlockToDisplay_at->p[i][j]);
    }   
    printf("\n");
  }
  printf("------------\n");
  return 1;
}

/* ---------------------------------------------------------- */
/* Display function in Hexa mode for a 4x4 block              */ 
/* ---------------------------------------------------------- */

int DisplayBlockHexa_i(char * BlockName_ac, Block_t * BlockToDisplay_at)
{
  int i = 0, j = 0;
  printf("---- %s ----\n", BlockName_ac);
  for (i=0;i<4; i++) {
    for (j=0;j<4; j++) {
      printf("0x%08X ", (int)(BlockToDisplay_at->p[i][j]));
    }   
    printf("\n");
  }
  printf("------------\n");
  return 1;
}

/* ---------------------------------------------------------- */
/* Init a 4x4 block with PixelValue + 1 for each pixel        */ 
/* ---------------------------------------------------------- */

int FillBlockWithIncValue_i(short PixelValue_s, Block_t * BlockToFill_at)
{
 int i = 0, j = 0;

 printf ("Fill Block with %d for the pixels of this block\n", PixelValue_s);
 for (i=0;i<4; i++) {
   for (j=0;j<4; j++) {
     BlockToFill_at->p[i][j] = PixelValue_s;
     PixelValue_s = PixelValue_s +1;
   }   
 }
 DisplayBlockHexa_i("New Block Content", BlockToFill_at);

 return 1;
}

/* ---------------------------------------------------------- */
/* Compute Level Scale Matrix                                 */ 
/* ---------------------------------------------------------- */

int LevelScaleMatrixDetermination_i(int Scale, Block_t * ComputedLevelScaleMatrix)
{

  ComputedLevelScaleMatrix->p[0][0] = LevelScaleBaseMatrix_auc[Scale][0];
  ComputedLevelScaleMatrix->p[0][2] = LevelScaleBaseMatrix_auc[Scale][0];
  ComputedLevelScaleMatrix->p[2][0] = LevelScaleBaseMatrix_auc[Scale][0];
  ComputedLevelScaleMatrix->p[2][2] = LevelScaleBaseMatrix_auc[Scale][0];
  ComputedLevelScaleMatrix->p[1][1] = LevelScaleBaseMatrix_auc[Scale][1];
  ComputedLevelScaleMatrix->p[1][3] = LevelScaleBaseMatrix_auc[Scale][1];
  ComputedLevelScaleMatrix->p[3][1] = LevelScaleBaseMatrix_auc[Scale][1];
  ComputedLevelScaleMatrix->p[3][3] = LevelScaleBaseMatrix_auc[Scale][1];
  ComputedLevelScaleMatrix->p[0][1] = LevelScaleBaseMatrix_auc[Scale][2];
  ComputedLevelScaleMatrix->p[0][3] = LevelScaleBaseMatrix_auc[Scale][2];
  ComputedLevelScaleMatrix->p[1][0] = LevelScaleBaseMatrix_auc[Scale][2];
  ComputedLevelScaleMatrix->p[1][2] = LevelScaleBaseMatrix_auc[Scale][2];
  ComputedLevelScaleMatrix->p[2][1] = LevelScaleBaseMatrix_auc[Scale][2];
  ComputedLevelScaleMatrix->p[2][3] = LevelScaleBaseMatrix_auc[Scale][2];
  ComputedLevelScaleMatrix->p[3][0] = LevelScaleBaseMatrix_auc[Scale][2];
  ComputedLevelScaleMatrix->p[3][2] = LevelScaleBaseMatrix_auc[Scale][2];
  
  return 1;
}

/* ---------------------------------------------------------- */
/* Init a 4x4 block with PixelValue for each pixel            */ 
/* ---------------------------------------------------------- */

int FillBlockWithOneValue_i(short PixelValue_s, Block_t * BlockToFill_at)
{
 int i = 0, j = 0;

 printf ("Fill Block with %d for the pixels of this block\n", PixelValue_s);
 for (i=0;i<4; i++) {
   for (j=0;j<4; j++) {
     BlockToFill_at->p[i][j] = PixelValue_s;
   }   
 }  

 DisplayBlockHexa_i("New Block Content", BlockToFill_at);

 return 1;
}

/* ---------------------------------------------------------- */
/* Scaling function                                           */ 
/* ---------------------------------------------------------- */

int Scaling_i(Block_t * Scale_at, Block_t * BlockToScale_at, Block_t * ReScaledBlock_at)
{
  int i = 0, j = 0, k = 0;
 
  for (i=0;i<4; i++) {
    for (j=0;j<4; j++) {
     ReScaledBlock_at->p[i][j] = BlockToScale_at->p[i][j] * Scale_at->p[i][j];
    }   
  }
  
  for (i=0;i<4; i++) {
    for (j=0;j<4; j++) {
      for (k=0;k<LeftShift_i; k++)
       ReScaledBlock_at->p[i][j] = ReScaledBlock_at->p[i][j]*2; 
    }   
  }  
  return 1;
}

/* ---------------------------------------------------------- */
/* Horizontal Transform 1                                     */ 
/* ---------------------------------------------------------- */

int HorizontalTransform_1_i(Block_t * BlockToTransform_at, Block_t * TransformedBlock_at)
{
  int i = 0;

  for (i=0;i<4; i++) {
     TransformedBlock_at->p[i][0] = BlockToTransform_at->p[i][0] + BlockToTransform_at->p[i][2];
     TransformedBlock_at->p[i][1] = BlockToTransform_at->p[i][0] - BlockToTransform_at->p[i][2];
     TransformedBlock_at->p[i][2] = ((BlockToTransform_at->p[i][1]) >> 1) - BlockToTransform_at->p[i][3];
     TransformedBlock_at->p[i][3] = BlockToTransform_at->p[i][1] + ((BlockToTransform_at->p[i][3]) >> 1);
  }   

  return 1;
}

/* ---------------------------------------------------------- */
/* Horizontal Transform 2                                     */ 
/* ---------------------------------------------------------- */

int HorizontalTransform_2_i(Block_t * BlockToTransform_at, Block_t * TransformedBlock_at)
{
  int i = 0;

  for (i=0;i<4; i++) {
     TransformedBlock_at->p[i][0] = BlockToTransform_at->p[i][0] + BlockToTransform_at->p[i][3];
     TransformedBlock_at->p[i][1] = BlockToTransform_at->p[i][1] + BlockToTransform_at->p[i][2];
     TransformedBlock_at->p[i][2] = BlockToTransform_at->p[i][1] - BlockToTransform_at->p[i][2];
     TransformedBlock_at->p[i][3] = BlockToTransform_at->p[i][0] - BlockToTransform_at->p[i][3];
  }   

  return 1;
}

/* ---------------------------------------------------------- */
/* Vertical Transform 1                                      */ 
/* ---------------------------------------------------------- */

int VerticalTransform_1_i(Block_t * BlockToTransform_at, Block_t * TransformedBlock_at)
{
  int j = 0;

  for (j=0;j<4; j++) {
     TransformedBlock_at->p[0][j] = BlockToTransform_at->p[0][j] + BlockToTransform_at->p[2][j];
     TransformedBlock_at->p[1][j] = BlockToTransform_at->p[0][j] - BlockToTransform_at->p[2][j];
     TransformedBlock_at->p[2][j] = ((BlockToTransform_at->p[1][j]) >> 1) - BlockToTransform_at->p[3][j];
     TransformedBlock_at->p[3][j] = BlockToTransform_at->p[1][j] + ((BlockToTransform_at->p[3][j]) >> 1);
  }

  return 1;
}

/* ---------------------------------------------------------- */
/* Vertical Transform 2                                       */ 
/* ---------------------------------------------------------- */

int VerticalTransform_2_i(Block_t * BlockToTransform_at, Block_t * TransformedBlock_at)
{
  int j = 0;

  for (j=0;j<4; j++) {
     TransformedBlock_at->p[0][j] = BlockToTransform_at->p[0][j] + BlockToTransform_at->p[3][j];
     TransformedBlock_at->p[1][j] = BlockToTransform_at->p[1][j] + BlockToTransform_at->p[2][j];
     TransformedBlock_at->p[2][j] = BlockToTransform_at->p[1][j] - BlockToTransform_at->p[2][j];
     TransformedBlock_at->p[3][j] = BlockToTransform_at->p[0][j] - BlockToTransform_at->p[3][j];
  }   

  return 1;
}

/* ---------------------------------------------------------- */
/* Addition + Shift                                           */ 
/* ---------------------------------------------------------- */

int AddShift_i(Block_t * BlockToProcess_at, Block_t * ProcessedBlock_at)
{
  int i= 0, j = 0;
 
  for (i=0;i<4; i++) {
    for (j=0;j<4; j++) {
      ProcessedBlock_at->p[i][j] = (BlockToProcess_at->p[i][j] + 2*2*2*2*2) >> 6;
    }
  }
  return 1;
}

/* ---------------------------------------------------------- */
/* Block Processing function                                  */ 
/* ---------------------------------------------------------- */
int BlockProcessing_i(int BlockId_i, int BlockInitialValue_i, Block_t * ResultingBlock_at)
{
 
  printf("\n====== BEGIN Block no %d =========\n", BlockId_i);
  printf("Initialise the residual block\n");
  
  FillBlockWithOneValue_i(BlockInitialValue_i, &c_t);

  printf("======================\n");
  printf("1. Level Scaling operation \n");
  Scaling_i(&LevelScale_t, &c_t, &d_t);
  DisplayBlockHexa_i("Rescaled block", &d_t);

  printf("======================\n");

  printf("2. Horizontal Transform 1\n");
  HorizontalTransform_1_i(&d_t, &e_t);
  DisplayBlockHexa_i("HorizontalTransform_1", &e_t);

  printf("======================\n");

  printf("3. Horizontal Transform 2\n");
  HorizontalTransform_2_i(&e_t, &f_t);
  DisplayBlockHexa_i("HorizontalTransform_2", &f_t);
  
  printf("======================\n");

  printf("4. Vertical Transform 1\n");
  VerticalTransform_1_i(&f_t, &g_t);
  DisplayBlockHexa_i("VerticalTransform_1", &g_t);

  printf("======================\n");

  printf("5. Vertical Transform 2\n");
  VerticalTransform_2_i(&g_t, &h_t);
  DisplayBlockHexa_i("VerticalTransform_2", &h_t);

  printf("======================\n");

  printf("6. Add and Shift\n");
/*   AddShift_i(&h_t, &r_t); */
/*   DisplayBlock_i("IQ-IT Result", &r_t); */
/*   DisplayBlockHexa_i("IQ-IT Result", &r_t);  */

  AddShift_i(&h_t, ResultingBlock_at);
  DisplayBlock_i("IQ-IT Result", ResultingBlock_at);
  DisplayBlockHexa_i("IQ-IT Result", ResultingBlock_at); 

  printf("====== END Block no %d =========\n\n", BlockId_i);
}

/* ---------------------------------------------------------- */
/* Main function                                              */ 
/* ---------------------------------------------------------- */

int main(int argc, char *argv[])
{
  int i = 0, j = 0, k = 0, l = 0;
  Block_t * BlockToDisplay_at = NULL;
  
  printf("\n\nTEST OF IQ-IT RESIDUAL 4x4 Block NORM IMPLEMENTATION %s - %s \n\n", "11:17:40", "Oct 20 2008");

  qP_i = 28;
  qP_div_6_i = (int)(qP_i / 6);
  qP_mod_6_i = qP_i - qP_div_6_i * 6;
  LeftShift_i = qP_div_6_i;

  printf("qP_i %d, qP_div_6_i %d, qP_mod_6_i %d, LeftShift_i %d\n",qP_i, qP_div_6_i, qP_mod_6_i, LeftShift_i);  

  printf("============================================\n");
  LevelScaleMatrixDetermination_i(qP_mod_6_i, &LevelScale_t);  
  DisplayBlock_i("Level Scale Matrix", &LevelScale_t);

  printf("======= START MACRO BLOCK PROCESSING =======\n");

  /* BlockProcessing_i(k, 0x1, &(MB_t[i][j])); */
  k = 0;
  for (i=0; i<4; i++) {
    for (j=0; j<4; j++) {
      BlockProcessing_i(k, k+1, &(MB_t[i][j]));
      k++;
    }
  } 

  k = 0;
  for (i=0; i<4; i++) {
    for (j=0; j<4; j++) {
      printf("0x%08X \n", k);
      DisplayBlockHexa_i("", &MB_t[i][j]);
      k++;
    }
  }
  
  printf("----  ----\n");
  printf("----  ----\n");
  printf("----  ----\n");
  printf("----  ----\n");
  for (i=0;i<4; i++) {
    for (j=0;j<4; j++) {
      printf("\n%d - ", i*4+j);
      for (k=0; k<4; k++) {
	for (l=0; l<4; l++) {
	  BlockToDisplay_at = &MB_t[k][l];
	  printf("0x%08X ", BlockToDisplay_at->p[i][j]);
	}
      }
      printf("\n");
    }   
  }
  
  printf("============================================\n");
  for (i=0;i<4; i++) {
    for (j=0;j<4; j++) {
       for (k=0; k<4; k++) {
	for (l=0; l<4; l++) {
	  BlockToDisplay_at = &MB_t[k][l];
	  printf("%x:%x:%04x: %04x\n", i,j, k*4+l,(short)(BlockToDisplay_at->p[i][j]));
	}
      }
    }   
  }  
  printf("\n\nEND  %s - %s \n\n", "11:17:40", "Oct 20 2008");

}
