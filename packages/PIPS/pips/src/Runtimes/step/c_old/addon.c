#ifdef OLD

#include "steprt.h"
extern MPI_Datatype step_types_table[STEP_MAX_TYPES];
extern struct step_internals step_;


#define bool int
#define true 1
#define false 0

/*####################################################

  Arithmetique de regions STEP

  ####################################################*/

/*
  Pour un tableau T de dimension d dont l'espace d'indice est [low_1;up_1] x ... x [low_d;up_d]

  On definit le tableau B de taille BOUNDS(d) tel que :
  B[LOW(i)] = low_i et
  B[UP(i)] = up_i
  
*/
#define LOW(i) (2*(i))
#define UP(i) (2*(i)+1)
#define BOUNDS(d) (2*d)



#define COPY_1D(I,R)				\
  {						\
    *(R + LOW(0)) = *(I + LOW(0));		\
    *(R + UP(0)) = *(I + UP(0));		\
  }

#define COPY_nD(d,I,R)				\
  {						\
    int i;					\
    for (i=0;i<d;i++)				\
      COPY_1D((I + 2*i),(R + 2*i));		\
  }

#ifdef SAFE
#define EMPTY_nD(d,I)				\
  {						\
    int i;					\
    for (i=0;i<d;i++)				\
      {						\
	*(I + LOW(i))=1;			\
	*(I + UP(i))=0;				\
      }						\
  }
#else
#define EMPTY_nD(d,I) {}
#endif

void printI(int d,int *I)
{
  if(d<=0) printf("[]");
  else
    {
      int i=0;
      printf("[%i,%i]",I[LOW(i)],I[UP(i)]);
      for(i=1;i<d;i++)
	printf("x[%i,%i]",I[LOW(i)],I[UP(i)]);
    }
}


bool empty_nD_p(int d, int *I)
{
  int i;
  bool empty=false;
  for (i=0;i<d && !empty;i++)
    empty |= I[LOW(i)] > I[UP(i)];

  if(empty) EMPTY_nD(d,I); // non indispensable mais initialise I à []^d

  return empty;
}


/*
  I, J et R sont vu comme des produits cartesiens : R[0] x R[1] x ... x R[d-1]
  retourne 0 si R=[], d sinon
*/
int inter_nD(int d, int *I, int *J, int *R)
{
  int i;
  bool empty=false;

  for (i=0; i<d && !empty; i++)
    {
      if (I[LOW(i)]>I[UP(i)] || J[LOW(i)]>J[UP(i)]) // I=[] ou J=[] on retourne []
	empty=true;
      else
	{
	  R[LOW(i)] = MAX(I[LOW(i)], J[LOW(i)]);
	  R[UP(i)] = MIN(I[UP(i)], J[UP(i)]);
	  empty |= R[LOW(i)] > R[UP(i)];
	}
    }

  if (empty)
    {
      EMPTY_nD(d,R);  // non indispensable mais initialise R à []^d
      printI(0,R);
      return 0;
    }
  printI(d,R);
  return d;
}

/*
  R est vu comme une union de i_dd produits cartésiens de d elements de R^2 (avec 0<= i_dd < 2^d)
  retournee la valeur "i_dd" est telle que pour tout i_dd <= i < 2^d => R[i]=[]^d
*/
int diff_nD(int d, int *I, int *J, int R[][d][2])
{
  int i_d,i_dd;
  int B[d][2];
  
  if (!inter_nD(d, I, J, *B)) // (I INTER J)=[] 
    {
      if (empty_nD_p(d, I))   // I=[] => R=[]
	i_dd=0;
      else
	{
	  COPY_nD(d,I,*R[0]); // R=I
	  i_dd=1;
	}
    }
  else
    {
      /*
	On a [B[LOW(i_d)],B[UP(i_d)]] = [I[LOW(i_d)],I[UP(i_d)]] INTER [J[LOW(i_d)],J[UP(i_d)]]

	Pour chaque dimension i_d, on peut décomposer [I[LOW(i_d)],I[UP(i_d)]] en potentiellement 3 intevalles :
	[B[LOW(i_d)], B[UP(i_d)]] -> definisant B(i_d)=[B[LOW(i_d)], B[UP(i_d)]] x ... x [B[LOW(d-1)], B[UP(d-1)]]
	[I[LOW(i_d)], B[LOW(i_d)]-1] -> donnant la sous-region I(i_d) x [I[LOW(i_d)], B[LOW(i_d)]-1] x B(i_d+1)
	[B[UP(i_d)]+1, I[UP(i_d)]] -> donnant la sous-region I(i_d) x [B[UP(i_d)]+1, I[UP(i_d)]] x B(i_d+1)
      */
      i_dd=0;
      for(i_d=d-1; i_d>=0; i_d--)
	{
	  if (I[LOW(i_d)] < B[i_d][LOW(0)]) // on a un intervalle [I[LOW(i_d)], B[LOW(i_d)]-1] non vide
	    {
	      // copie de I(i_d) = I[0] x ... x I[i_d-1]     dans R[i_dd][0] x ... x R[i_dd][i_d-1]
	      COPY_nD(i_d, I, R[i_dd][0]);

	      // copie de [I[LOW(i_d)], B[LOW(i_d)]-1]       dans R[i_dd][i_d]
	      R[i_dd][i_d][LOW(0)] = I[LOW(i_d)];
	      R[i_dd][i_d][UP(0)] = B[i_d][LOW(0)] - 1;
	      
	      // copie de B(i_d+1) = B[i_d+1] x ... x B[d-1] dans R[i_dd][i_d+1] x ... x R[i_dd][d-1]
	      COPY_nD(d-1-i_d, B[i_d+1], R[i_dd][i_d+1]);
	      i_dd+=1;
	    }

	  if (I[UP(i_d)] > B[i_d][UP(0)])   // on a un intervalle [B[UP(i_d)]+1, I[UP(i_d)]] non vide
	    {
	      // copie de I(i_d) = I[0] x ... x I[i_d-1]     dans R[i_dd][0] x ... x R[i_dd][i_d-1]
	      COPY_nD(i_d, I, R[i_dd][0]);

	      // copie de [B[UP(i_d)]+1, I[UP(i_d)]]         dans R[i_dd][i_d]
	      R[i_dd][i_d][LOW(0)] = B[i_d][UP(0)] + 1;
	      R[i_dd][i_d][UP(0)] =  I[UP(i_d)];
	      
	      // copie de B(i_d+1) = B[i_d+1] x ... x B[d-1] dans R[i_dd][i_d+1] x ... x R[i_dd][d-1]
	      COPY_nD(d-1-i_d, B[i_d+1], R[i_dd][i_d+1]);
	      i_dd+=1;
	    }
	}
    }
    
  for (i_d=i_dd;i_d< 1<<d;i_d++) EMPTY_nD(d,*R[i_d]); // non indispensable mais initialise R[i_d] à []^d pour i_dd <= i_d < 2^d

  //affichage du resultat
  printf("\n");
  for(i_d=0;i_d<i_dd;i_d++)
    {
      printf("R(%i)=",i_d);printI(d,R[i_d][0]);printf("\n");
    }
  printf("\n");
  for(i_d=i_dd;i_d< 1<<d ;i_d++)
    {
      printf("R(%i)=",i_d);printI(d,R[i_d][0]);printf("\n");
    }

  return i_dd;
}

/*####################################################

  Conversion region STEP -> type MPI

  ####################################################*/


int region_allArray_Sizes(int d, int *allArray_bounds, int *allArray_sizes)
{
  int i;
  bool empty=false;
  for (i=0; i<d && !empty; i++)
    {
      allArray_sizes[i] = allArray_bounds[UP(i)] - allArray_bounds[LOW(i)] + 1;
      empty |= (allArray_sizes[i]<=0);
    }
  if (empty) return 0;
  return d;
}

int region_subArray_BoundsToStartSizes(int d,int *allArray_bounds, int *subArray_bounds,int *subArray_start, int *subArray_sizes)
{
  int i;
  bool empty=false;
  for (i=0; i<d && !empty; i++)
    {
      subArray_start[i] = subArray_bounds[LOW(i)] - allArray_bounds[LOW(i)];
      subArray_sizes[i] = subArray_bounds[UP(i)] - subArray_bounds[LOW(i)] + 1;
      empty |= (subArray_sizes[i]<=0);
    }
  if (empty) return 0;
  return d;
}


void subArray_union(int nb_union,int dim, int subArray_bounds[nb_union][dim][2],int *allArray_bounds,int *allArray_sizes,
		    MPI_Datatype dataType, int order, MPI_Datatype *unionType)
{
  int id_union;
  int subArray_start[nb_union][dim];
  int subArray_sizes[nb_union][dim];
  int array_of_blocklengths[nb_union];
  MPI_Aint array_of_displacements[nb_union];
  MPI_Datatype array_of_type[nb_union];

  for (id_union=0; id_union<nb_union; id_union ++)
    {
      array_of_blocklengths[id_union]=1;
      array_of_displacements[id_union]=0;
      region_subArray_BoundsToStartSizes(dim, allArray_bounds, *subArray_bounds[id_union], subArray_start[id_union], subArray_sizes[id_union]);
      MPI_Type_create_subarray(dim, allArray_sizes, subArray_sizes[id_union], subArray_start[id_union], order, dataType, &array_of_type[id_union]);
    }
  
  MPI_Type_create_struct(nb_union, array_of_blocklengths, array_of_displacements, array_of_type, unionType);
}


void region_Send_Recv_STEP2MPI(int dim, int nb_regions, STEP_Datatype type, int *allArray_bounds,
			       int *stepSendRegion_bounds, int *stepRecvRegion_bounds,
			       MPI_Datatype *SendData, MPI_Datatype *RecvData)
{
  int id_node;
  int subArray_bounds[BOUNDS(dim)];
  int allArray_sizes[dim];
  int union_R[1<<dim][dim][2];

  int Rk=step_.rank_;
  int order=step_.language;
  MPI_Datatype dataType = step_types_table[type];

  assert(nb_regions==step_.size_);
  region_allArray_Sizes(dim, allArray_bounds, allArray_sizes);

  for (id_node=0; id_node < step_.size_; id_node++)
    {
      // On considere le scheduling static associant la region i au processus de rang i
      int id_region=id_node;

      // calcul SEND(Rk) INTER RECV(id_node)
      printf("%i) SENDdata(%i) = SEND(%i) INTER RECV(%i) : ",Rk,id_node,Rk,id_region);
      if(id_node!=Rk && inter_nD(dim, &stepSendRegion_bounds[Rk * BOUNDS(dim)], &stepRecvRegion_bounds[id_region * BOUNDS(dim)], subArray_bounds))
	{
	  // calcul de (SEND(Rk) INTER RECV(id_node)) \ LOCAL(id_node)
	  // LOCAL(id_node) = SEND(id_node)
	  int *local=&stepSendRegion_bounds[id_region * BOUNDS(dim)];
	  printf("\n%i) local=",Rk);printI(dim,local);printf("\n");

	  int nb_union=diff_nD(dim, subArray_bounds, local, union_R);

	  subArray_union(nb_union, dim, union_R, allArray_bounds, allArray_sizes, dataType, order, &SendData[id_node]);
	  MPI_Type_commit (&SendData[id_node]);
	}
      else
	SendData[id_node]=MPI_DATATYPE_NULL;
      printf("\n");

      //calcul SEND(id_region) INTER RECV(Rk)
      printf("%i) RECVdata(%i) = SEND(%i) INTER RECV(%i) : ",Rk,id_node,id_region,Rk);
      if (id_node!=Rk && inter_nD(dim, &stepSendRegion_bounds[id_region * BOUNDS(dim)], &stepRecvRegion_bounds[Rk * BOUNDS(dim)], subArray_bounds))
	{
	  // calcul de (SEND(id_region) INTER RECV(Rk)) \ LOCAL(Rk)
	  // LOCAL(Rk) = SEND(Rk)
	  int *local=&stepSendRegion_bounds[Rk * BOUNDS(dim)];
	  printf("\n%i) local=",Rk);printI(dim,local);printf("\n");

	  int nb_union=diff_nD(dim, subArray_bounds, local, union_R);
	  
	  subArray_union(nb_union, dim, union_R, allArray_bounds, allArray_sizes, dataType, order, &RecvData[id_node]);
	  MPI_Type_commit (&RecvData[id_node]);
	}
      else
	RecvData[id_node]=MPI_DATATYPE_NULL;
      printf("\n");
    }
}


/*####################################################

  Algorithme de communiation

  ####################################################*/
static void step_alltoallregion_NBlocking(void *array, int tag, MPI_Datatype *SendData, MPI_Datatype *RecvData,
					  int *nb_request, MPI_Request *requests)
{
  IN_TRACE("array = %p SendData = %p RecvData = %p nb_request = %d",
	   array, SendData, RecvData, *nb_request);
  int id_node;
  int nb_node=step_.size_;
  int Rk=step_.rank_;

  for (id_node = 0; id_node < nb_node; id_node++)
    if ((id_node != Rk) && (RecvData[id_node] != MPI_DATATYPE_NULL))
      MPI_Irecv((void *) array, 1, RecvData[id_node], id_node, tag, MPI_COMM_WORLD, &requests[(*nb_request)++]);

  for (id_node = 0; id_node < nb_node; id_node++)
    if ((id_node != Rk) && (SendData[id_node] != MPI_DATATYPE_NULL))
      MPI_Isend((void *) array, 1, SendData[id_node], id_node, tag, MPI_COMM_WORLD, &requests[(*nb_request)++]);

  OUT_TRACE("");
}


void step_alltoallregion_opt(int *dim, int *nb_regions, int *regions_send, int *regions_recv,
			     void *data_array, int *tag ,int *max_nb_request, MPI_Request *requests,
			     int *nb_request, int *algorithm, STEP_Datatype *type)
{
  MPI_Datatype SendData[step_.size_];
  MPI_Datatype RecvData[step_.size_];
  int *allArray_bounds = &regions_send[0];
  int *stepSendRegion = &regions_send[BOUNDS(*dim)];
  int *stepRecvRegion = &regions_recv[BOUNDS(*dim)];
 
  region_Send_Recv_STEP2MPI(*dim, *nb_regions, *type, allArray_bounds,
			    stepSendRegion, stepRecvRegion, SendData, RecvData);

  switch (*algorithm)
    {
    case STEP_NBLOCKING_ALG:
      step_alltoallregion_NBlocking(data_array, *tag, SendData, RecvData, nb_request, requests);
      break;
    default:
      assert(0);
    }
}





/*####################################################

  Pour test unitaire

  ####################################################*/
#ifdef MAIN
int main ()
{
  int i;
  int d=2;
  int I[2][2]={{1,6},{1,6}};
  int J[2][2]={{2,4},{3,4}};
  int R[1<<d][d][2];

  printf("I=");printI(d,I[0]);printf("\n");
  printf("J=");printI(d,J[0]);printf("\n");
  printf("\n");

  diff_nD(d,I[0],J[0],R);

  return 0;
}
#endif
#endif
