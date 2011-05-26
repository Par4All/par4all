#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "steprt_private.h"

#define BETWEEN(val,low,up)  ((val>=low)&&(val<=up))
#define REGION_OFFSET(nbreg)   ((nbreg)*(2*dims)) 


/******************************************************************************/
/*
  R1[][dims][2]
  R2[dims][2]
*/
int 
complement_region(int dims,index_t *R1,index_t *R2, int nb_sub_regions,int *nb_results, index_t *Result)
{
  int i;
  int insert_point= 0;
  int nb_regions,return_nb;

  nb_regions = 0;
  return_nb  = 0;
  display_region(dims,R2, "region 2");
  for (i=0;i<nb_sub_regions;i++)
    {
      diff_region(dims,R2,&(R1[REGION_OFFSET(i)]),&nb_regions,&(Result[REGION_OFFSET(insert_point)]));
      return_nb = return_nb + nb_regions;
      insert_point = return_nb;
    }
  *nb_results = return_nb;
  return (return_nb);
}


/******************************************************************************/
/*
  R1[dims][2]
  R2[dims][2]
*/
int 
is_regions_equal(int dims,index_t *R1,index_t *R2)
{
  int i = 0;
  int equal =true;

  for (i=0;equal && (i<dims);i++)
    {
      equal=((R1[LOW(i)]==R2[LOW(i)]) && (R1[UP(i)]==R2[UP(i)]));
    }
  return equal;
}




/******************************************************************************/
/*
  R[][dims][2]
*/
int 
is_region_empty(int dims,index_t *R)
{
  int i = 0;
  int empty =false;

  for (i=0;(i<dims)&&(empty==false);i++)
    {
      if ((R[LOW(i)]>R[UP(i)])) 
	return (true);
    }
  return (false);
}

/******************************************************************************/
/*
  dest[][dims][2]
  src[dims][2]
*/
void
copy_region(int dims, index_t *src,index_t *dest)   
{
  int i;

  for (i=0;i<dims;i++)
    {
      dest[LOW(i)] = src[LOW(i)];
      dest[UP(i)]  = src[UP(i)];
    }
}

void display_region(int dims,index_t *region, char *reg_name)
{
  int i = 0;
  int empty =false;

  for (i=0;(i<dims)&&(empty==false);i++)
    {
      if ((region[LOW(i)]>region[UP(i)])) 
	empty=true;
    }
  if (empty)
    {
      printf("\n------- REGION %s ---------------------------\n",reg_name);
      printf("\nEMPTY REGION\n");;
      printf("\n--------------------------------------------------\n");;
    }        
  else
    {
      printf("\n------- REGION %s ---------------------------\n",reg_name);
      printf("\n Number of dimensions  = %d\n",dims);
      for (i=0;i<dims;i++)
        {
	  printf("\n dims N°%d \t start =%d \t end =%d",i,region[LOW(i)],region[UP(i)]);
        }
      printf("\n--------------------------------------------------\n");;
    }
}

/******************************************************************************/
/*
  R1[dims][2]
  R2[dims][2]
  Result[dims][2]
*/

int 
intersect_regions(int dims, index_t *R1, index_t *R2, index_t *Result)
{
  int i;
  int empty=false;

  for (i=0; (i<dims); i++)
    {
      if ((R1[LOW(i)]>R1[UP(i)]) || (R2[LOW(i)]>R2[UP(i)])) 
	{ 
	  empty=true;
	}
      else
	{
	  Result[LOW(i)] = MAX(R1[LOW(i)], R2[LOW(i)]);
	  Result[UP(i)]  = MIN(R1[UP(i)], R2[UP(i)]);
          empty |= (Result[LOW(i)] > Result[UP(i)]);
	}
    }

  if (empty==true)
    {
      for (i=0;i<dims;i++)
	{        
	  Result[LOW(i)] = 1;    
	  Result[UP(i)] = 0;
	}  
      return false;
    }
  else 
    {      
      return true;    
    }
}

/******************************************************************************/
/* 
   Result = R1 - R2
   R1[dims][2]
   R2[dims][2]
   Result[][dims][2]
*/

int 
diff_region(int dims, index_t *R1, index_t *R2,int *nb_results, index_t *Result)
{
  int i,nb_regions;
  index_t tmp_region[2*dims];
  index_t r_inter[2*dims];
  index_t R1bis [2*dims];
  int inter_result = intersect_regions(dims,R1,R2,r_inter);

  copy_region(dims,R1,R1bis);
  if (inter_result==false)
    {
      if (!is_region_empty(dims,R1))
	{    
	  *nb_results = 1;
	  copy_region(dims,R1,Result);     
	  return true;       
	}
      else
	{
	  *nb_results = 0;
	  return false;       
	}
    }
  else
    {
      if (is_regions_equal(dims,R1,r_inter))
	{
	  *nb_results = 0;   
	  return false; 
	}
      else
	{
          nb_regions = 0;
          for (i=0;i<dims;i++)
	    {
	      /* AM
		 if (r_inter[UP(i)] < R1bis[UP(i)])
		 {
		 copy_region(dims,R1bis,&(Result[REGION_OFFSET(nb_regions)]));
		 Result[REGION_OFFSET(nb_regions) + LOW(i)] = r_inter[UP(i)]+1;
		 R1bis[UP(i)] = r_inter[UP(i)];
		 nb_regions++;
		 }
		 if (r_inter[LOW(i)] > R1bis[LOW(i)])
		 {
		 copy_region(dims,R1bis,&(Result[REGION_OFFSET(nb_regions)]));
		 Result[REGION_OFFSET(nb_regions) + UP(i)] = r_inter[LOW(i)]-1;
		 R1bis[LOW(i)] = r_inter[LOW(i)];
		 nb_regions++;
		 }
	      */
	      if (BETWEEN(r_inter[UP(i)],R1bis[LOW(i)],R1bis[UP(i)]))
		{
		  copy_region(dims,R1bis,tmp_region);
		  tmp_region[LOW(i)] = r_inter[UP(i)]+1;
		  tmp_region[UP(i)]  = R1bis[UP(i)];
		  R1bis[UP(i)] = r_inter[UP(i)];
		  if (!(is_region_empty(dims,tmp_region)))
		    {
		      copy_region(dims,tmp_region,&(Result[REGION_OFFSET(nb_regions)]));
		      nb_regions++;
		    }
		} 
	      if (BETWEEN(r_inter[LOW(i)],R1bis[LOW(i)],R1bis[UP(i)]))
		{
		  copy_region(dims,R1bis,tmp_region);
		  tmp_region[UP(i)] = r_inter[LOW(i)]-1;
		  R1bis[LOW(i)] = r_inter[LOW(i)];
		  if (!(is_region_empty(dims,tmp_region)))
		    {
		      copy_region(dims,tmp_region,&(Result[REGION_OFFSET(nb_regions)]));
		      nb_regions++;
		    }
		} 
	    }     
          *nb_results = nb_regions;
          return true;
	}
    }
}

/******************************************************************************/
/* 
   Result = Normalize(R1) 
   R1[dims][2]
   Array[dims][2]
*/
/*
  int 
  normalize_region(int dims, index_t *R1, index_t *array,index_t *Result)
  {
  int i;
  for (i=0;i<dims;i++)
  {
                
  }
  }*/
/******************************************************************************/

