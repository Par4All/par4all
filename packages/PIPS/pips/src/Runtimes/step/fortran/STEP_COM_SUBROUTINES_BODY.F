#ifdef COPYRIGHT
* Copyright 2007, 2008,2009 Alain Muller, Frederique Silber-Chaussumier, Christian Parrot
*
*This file is part of STEP.
*
*The program is distributed under the terms of the GNU General Public
*License.
#endif
#define InitReduction(TYPE_SUFFIX,TYPE_FORTRAN,TYPE_MPI)\
      subroutine STEP_InitReduction/**/TYPE_SUFFIX(variable,                                              !$\
     &     variable_reduc,operator)                                                                       !$\
      implicit none                                                                                       !$\
      include 'mpif.h'                                                                                    !$\
      include 'steprt_f.h'                                                                                !$\
      TYPE_FORTRAN variable,variable_reduc                                                                !$\
      integer operator                                                                                    !$\
                                                                                                          !$\
      variable_reduc=variable                                                                             !$\
      select case(operator)                                                                               !$\
      case(STEP_SUM)                                                                                      !$\
         variable=0                                                                                       !$\
      case(STEP_PROD)                                                                                     !$\
         variable=1                                                                                       !$\
      end select                                                                                          !$\
      end                                                                                                 !$\
                                                                                                          !$\

#define Reduction(TYPE_SUFFIX,TYPE_FORTRAN,TYPE_MPI)\
      subroutine STEP_Reduction/**/TYPE_SUFFIX(variable, variable_reduc,                                  !$\
     &     operator)                                                                                      !$\
      implicit none                                                                                       !$\
      include 'mpif.h'                                                                                    !$\
      include 'steprt_f.h'                                                                                !$\
      TYPE_FORTRAN variable,variable_reduc,tmp_result                                                     !$\
      integer operator                                                                                    !$\
      integer*4 internalOperator,iErr                                                                     !$\
                                                                                                          !$\
      internalOperator=operator                                                                           !$\
      call MPI_ALLREDUCE (variable,tmp_result,1,TYPE_MPI,                                                 !$\
     &     internalOperator,MPI_COMM_WORLD,iErr)                                                          !$\
      select case(operator)                                                                               !$\
      case(STEP_SUM)                                                                                      !$\
         variable=variable_reduc+tmp_result                                                               !$\
      case(STEP_PROD)                                                                                     !$\
         variable=variable_reduc*tmp_result                                                               !$\
      end select                                                                                          !$\
      end                                                                                                 !$\
                                                                                                          !$\

#define STEP_MasterToAllScalar(TYPE_SUFFIX,TYPE_FORTRAN,TYPE_MPI)\
      subroutine STEP_MasterToAllScalar/**/TYPE_SUFFIX(scalar,                                            !$\
     &     max_nb_request,requests,nb_request,                                                            !$\
     &     algorithm)                                                                                     !$\
      implicit none                                                                                       !$\
      include 'mpif.h'                                                                                    !$\
      TYPE_FORTRAN scalar                                                                                 !$\
      integer*4 iErr                                                                                      !$\
      integer max_nb_request                                                                              !$\
      integer requests(max_nb_request)                                                                    !$\
      integer nb_request                                                                                  !$\
      integer algorithm                                                                                   !$\
                                                                                                          !$\
      select case (algorithm)                                                                             !$\
      case DEFAULT                                                                                        !$\
         call MPI_BCAST(scalar,1,TYPE_MPI,0,MPI_COMM_WORLD,iErr)                                          !$\
      end select                                                                                          !$\
      end                                                                                                 !$\
                                                                                                          !$\

#define STEP_MasterToAllRegion(TYPE_SUFFIX,TYPE_FORTRAN,TYPE_MPI)\
      subroutine STEP_MasterToAllRegion/**/TYPE_SUFFIX(array,                                             !$\
     &     dim,regions,size,                                                                              !$\
     &     max_nb_request,requests,nb_request,                                                            !$\
     &     algorithm)                                                                                     !$\
      implicit none                                                                                       !$\
      include 'mpif.h'                                                                                    !$\
      include 'STEP.h'                                                                                    !$\
      include 'steprt.h'                                                                                  !$\
      integer dim,size                                                                                    !$\
      TYPE_FORTRAN array(1:size)     !tableau array linearise                                             !$\
      integer L,U,B_                                                                                      !$\
      parameter (L=1,U=2,B_=0)                                                                            !$\
      integer regions(L:U,1:dim,0:1) !description de l'espace d'indice des differentes regions            !$\
                                 !regions(:,:) decrit l'espace d'indice du tableau array non linearise    !$\
      integer*4 iErr                                                                                      !$\
      integer max_nb_request                                                                              !$\
      integer requests(max_nb_request)                                                                    !$\
      integer nb_request                                                                                  !$\
      integer algorithm                                                                                   !$\
      integer origine           !resultat fonction de calcul                                              !$\
      integer type_sub_region   !resultat fonction de calcul                                              !$\
      integer type                                                                                        !$\
      integer*4 internalType                                                                              !$\
                                                                                                          !$\
      call type_subRegion(dim,                                                                            !$\
     &     regions(L,1,B_),                                                                               !$\
     &     regions(L,1,B_),                                                                               !$\
     &     TYPE_MPI,type)                                                                                 !$\
      internalType=type                                                                                   !$\
      if(type.NE.MPI_DATATYPE_NULL) then                                                                  !$\
         call MPI_TYPE_COMMIT(internalType,iErr)                                                          !$\
      endif                                                                                               !$\
      type=internalType                                                                                   !$\
      select case (algorithm)                                                                             !$\
      case DEFAULT                                                                                        !$\
         call MPI_BCAST(array(origine(dim,regions(L,1,B_),                                                !$\
     &        regions(L,1,B_))),1,internalType,0,MPI_COMM_WORLD,iErr)                                     !$\
      type=internalType                                                                                   !$\
      end select                                                                                          !$\
      end                                                                                                 !$\
                                                                                                          !$\

#define STEP_AlltoAllRegion(MERGE,TYPE_SUFFIX,TYPE_FORTRAN,TYPE_MPI)\
#if INTERLACED                                                                                            !$\
#define PARAMETER_MERGE ,dim,nb_regions,regions,initial,buffer                                            !$\
#else                                                                                                     !$\
#define PARAMETER_MERGE                                                                                   !$\
#endif                                                                                                    !$\
      subroutine STEP_AlltoAllRegion/**/MERGE/**/TYPE_SUFFIX(dim,                                         !$\
     &     nb_regions,regions,                                                                            !$\
     &     size,array,                                                                                    !$\
#if INTERLACED                                                                                            !$\
     &     initial,buffer,                                                                                !$\
#endif                                                                                                    !$\
     &     tag,                                                                                           !$\
     &     max_nb_request,requests,nb_request,                                                            !$\
     &     algorithm)                                                                                     !$\
      implicit none                                                                                       !$\
      include 'mpif.h'                                                                                    !$\
      include 'STEP.h'                                                                                    !$\
      include 'steprt.h'                                                                                  !$\
      integer NBLOCKING,BLOCKING1,BLOCKING2                                                               !$\
      parameter (NBLOCKING=0,BLOCKING1=1,BLOCKING2=2)                                                     !$\
      integer dim               !nombre de dimension du tableau array avant linearisation                 !$\
      integer size              !taille du tableau array                                                  !$\
      TYPE_FORTRAN array(1:size)     !tableau array linearise                                             !$\
#if INTERLACED                                                                                            !$\
      TYPE_FORTRAN initial(1:size),buffer(1:size)                                                         !$\
#endif                                                                                                    !$\
      integer L,U,B_                                                                                      !$\
      parameter (L=1,U=2,B_=0)                                                                            !$\
      integer nb_regions        !nombre d'element du tableau region +1                                    !$\
      integer regions(L:U,1:dim,B_:nb_regions) !description de l'espace d'indice des differentes regions  !$\
                                !regions(:,:,B_) decrit l'espace d'indice du tableau array non linearise  !$\
      integer tag               !tag MPI pour le message                                                  !$\
      integer max_nb_request                                                                              !$\
      integer requests(max_nb_request)                                                                    !$\
      integer nb_request                                                                                  !$\
      integer algorithm                                                                                   !$\
      integer origines(STEP_MAX_NBNODE)                                                                   !$\
      integer id_region,idProc                                                                            !$\
      integer origine           !resultat fonction de calcul                                              !$\
      integer type_sub_region   !resultat fonction de calcul                                              !$\
      integer types(STEP_MAX_NBNODE)                                                                      !$\
      integer*4 internalTypes                                                                             !$\
      integer*4 iErr,iBidon                                                                               !$\
                                                                                                          !$\
      if (nb_regions .gt. STEP_MAX_NBNODE) then                                                           !$\
         print *,"STEP_AlltoAllRegion/**/MERGE/**/TYPE_SUFFIX() : STEP_MAX_NBNODE                         !$\
     &trop petit (update STEP.h) nb_regions = ",nb_regions                                                !$\
         call MPI_ABORT(MPI_COMM_WORLD, iBidon, iErr)                                                     !$\
      end if                                                                                              !$\
      if(nb_request.GE.max_nb_request-2*(STEP_Size-1)) then                                               !$\
         print *,"STEP error : requests array too small"                                                  !$\
         stop                                                                                             !$\
      endif                                                                                               !$\
      ! initialisation des types MPI et du tableau des origines des regions                               !$\
      do id_region=1,nb_regions                                                                           !$\
         origines(id_region)=origine(dim,regions(L,1,B_),                                                 !$\
     &        regions(L,1,id_region))                                                                     !$\
         call type_subRegion(dim,                                                                         !$\
     &     regions(L,1,B_),                                                                               !$\
     &     regions(L,1,id_region),                                                                        !$\
     &     TYPE_MPI,types(id_region))                                                                     !$\
         if(types(id_region).NE.MPI_DATATYPE_NULL) then                                                   !$\
            internalTypes=types(id_region)                                                                !$\
            call MPI_TYPE_COMMIT(internalTypes,iErr)                                                      !$\
            types(id_region)=internalTypes                                                                !$\
         endif                                                                                            !$\
      enddo                                                                                               !$\
      SELECT CASE(algorithm)                                                                              !$\
#ifndef INTERLACED                                                                                        !$\
      case (NBLOCKING)                                                                                    !$\
         call AlltoAll_nBlocking/**/TYPE_SUFFIX(STEP_Size,types,origines,                                 !$\
     &        STEP_SizeRegion(dim,regions(L,1,B_)),array,tag,                                             !$\
     &        max_nb_request,requests,nb_request)                                                         !$\
#endif                                                                                                    !$\
      case (BLOCKING1)                                                                                    !$\
         if(mod(STEP_SIZE,2).EQ.0)then                                                                    !$\
            call AlltoAll_even/**/MERGE/**/TYPE_SUFFIX(STEP_Size,types,origines,                          !$\
     &           STEP_SizeRegion(dim,regions(L,1,B_)),array,tag                                           !$\
     &           PARAMETER_MERGE)                                                                         !$\
         else                                                                                             !$\
            call AlltoAll_odd/**/MERGE/**/TYPE_SUFFIX(STEP_Size,types,origines,                           !$\
     &           STEP_SizeRegion(dim,regions(L,1,B_)),array,tag                                           !$\
     &           PARAMETER_MERGE)                                                                         !$\
         endif                                                                                            !$\
      case DEFAULT                                                                                        !$\
         if(mod(STEP_SIZE,2).EQ.0)then                                                                    !$\
c      print *,STEP_Rank,') AlltoAll_even2/**/MERGE/**/TYPE_SUFFIX'                                       !$\
            call AlltoAll_even2/**/MERGE/**/TYPE_SUFFIX(STEP_Size,types,origines,                         !$\
     &           STEP_SizeRegion(dim,regions(L,1,B_)),array,tag                                           !$\
     &           PARAMETER_MERGE)                                                                         !$\
         else                                                                                             !$\
            call AlltoAll_odd2/**/MERGE/**/TYPE_SUFFIX(STEP_Size,types,origines,                          !$\
     &           STEP_SizeRegion(dim,regions(L,1,B_)),array,tag                                           !$\
     &           PARAMETER_MERGE)                                                                         !$\
         endif                                                                                            !$\
      end select                                                                                          !$\
      ! liberation de la memoire pour les type MPI                                                        !$\
      do id_region=1,nb_regions                                                                           !$\
         if (types(id_region).NE.MPI_DATATYPE_NULL) then                                                  !$\
            internalTypes=types(id_region)                                                                !$\
            call MPI_Type_free(internalTypes,iErr);                                                       !$\
            types(id_region)=internalTypes                                                                !$\
         endif                                                                                            !$\
      enddo                                                                                               !$\
      end                                                                                                 !$\
                                                                                                          !$\

#define AlltoAll_nBlocking(TYPE_SUFFIX,TYPE_FORTRAN)\
      subroutine AlltoAll_nBlocking/**/TYPE_SUFFIX(nb_node,types,origines,                                !$\
     &     size,array,tag,max_nb_request,requests,nb_request)                                             !$\
      implicit none                                                                                       !$\
      include 'mpif.h'                                                                                    !$\
      include 'STEP.h'                                                                                    !$\
      integer nb_node                                                                                     !$\
      integer STEP_Rank                                                                                   !$\
      integer origines(0:nb_node-1),size                                                                  !$\
      TYPE_FORTRAN array(size)                                                                            !$\
      integer max_nb_request                                                                              !$\
      integer nb_request                                                                                  !$\
      integer tag                                                                                         !$\
      integer types(0:nb_node-1)                                                                          !$\
      integer requests(max_nb_request)                                                                    !$\
      integer*4 internalTag                                                                               !$\
      integer*4 node,iErr                                                                                 !$\
      integer*4 internalRequests                                                                          !$\
      integer*4 internalTypes                                                                             !$\
                                                                                                          !$\
      internalTag=tag                                                                                     !$\
      CALL STEP_Get_rank(STEP_Rank)                                                                       !$\
      do node=0,nb_node-1                                                                                 !$\
         if ((node.NE.STEP_Rank).AND.                                                                     !$\
     &        (types(node).NE.MPI_DATATYPE_NULL)) then                                                    !$\
            nb_request=nb_request+1                                                                       !$\
            internalTypes=types(node)                                                                     !$\
            internalRequests=requests(nb_request)                                                         !$\
            call MPI_IRECV(array(origines(node)),1,internalTypes,node,                                    !$\
     &           internalTag, MPI_COMM_WORLD,internalRequests,iErr)                                       !$\
            types(node)=internalTypes                                                                     !$\
            requests(nb_request)=internalRequests                                                         !$\
          endif                                                                                           !$\
      enddo                                                                                               !$\
      if(types(STEP_Rank).NE.MPI_DATATYPE_NULL) then                                                      !$\
         do node=0,nb_node-1                                                                              !$\
            if (node.NE.STEP_Rank) then                                                                   !$\
               nb_request=nb_request+1                                                                    !$\
               internalTypes=types(STEP_Rank)                                                             !$\
               internalRequests=requests(nb_request)                                                      !$\
               call MPI_ISEND(array(origines(STEP_Rank)),1,                                               !$\
     &              internalTypes,node,                                                                   !$\
     &              internalTag, MPI_COMM_WORLD,internalRequests,iErr)                                    !$\
               types(STEP_Rank)=internalTypes                                                             !$\
               requests(nb_request)=internalRequests                                                      !$\
            endif                                                                                         !$\
         enddo                                                                                            !$\
      endif                                                                                               !$\
      end                                                                                                 !$\
                                                                                                          !$\

#define  AlltoAll_even(MERGE,TYPE_SUFFIX,TYPE_FORTRAN)\
#if INTERLACED                                                                                            !$\
#define BUFFER buffer                                                                                     !$\
#define PARAMETER_MERGE ,dim,nb_regions,regions,initial,buffer                                            !$\
#else                                                                                                     !$\
#define BUFFER array                                                                                      !$\
#define PARAMETER_MERGE                                                                                   !$\
#endif                                                                                                    !$\
      subroutine AlltoAll_even/**/MERGE/**/TYPE_SUFFIX(nb_node,types,origines,                            !$\
     &     size,array,tag PARAMETER_MERGE)                                                                !$\
      implicit none                                                                                       !$\
      include 'mpif.h'                                                                                    !$\
      include 'STEP.h'                                                                                    !$\
      integer nb_node                                                                                     !$\
      integer origines(0:nb_node-1),size                                                                  !$\
      TYPE_FORTRAN array(size)                                                                            !$\
#if INTERLACED                                                                                            !$\
      integer dim                                                                                         !$\
      integer L,U,B_                                                                                      !$\
      parameter (L=1,U=2,B_=0)                                                                            !$\
      integer nb_regions        !nombre d'élément du tableau région +1                                    !$\
      integer regions(L:U,1:dim,B_:nb_regions) !description de l'espace d'indice des différentes régions  !$\
                                !regions(:,:,B_) decrit l'espace d'indice du tableau array non linéarisé  !$\
      TYPE_FORTRAN initial(1:size),buffer(1:size)                                                         !$\
#endif                                                                                                    !$\
      integer id_data                                                                                     !$\
      integer data_send,data_recv,incr                                                                    !$\
      integer step,m                                                                                      !$\
      logical rank_even,step_even                                                                         !$\
      integer types(0:nb_node-1)                                                                          !$\
      integer tag                                                                                         !$\
      integer STEP_Rank                                                                                   !$\
      integer comm_node(0:1)                                                                              !$\
      integer*4 STATUS(MPI_STATUS_SIZE)                                                                   !$\
      integer*4 internalTag                                                                               !$\
      integer*4 node,iErr                                                                                 !$\
      integer*4 internalRequests                                                                          !$\
      integer*4 internalTypes                                                                             !$\
      integer*4 internalComm_node                                                                         !$\
                                                                                                          !$\
      internalTag=tag                                                                                     !$\
      CALL STEP_Get_rank(STEP_Rank)                                                                       !$\
      step=0                                                                                              !$\
      if (mod(STEP_Rank,2).EQ.0) then !calcule du noeud avec lequel on communique a l'etape step          !$\
         comm_node(0)=STEP_Rank+1                                                                         !$\
         comm_node(1)=STEP_Rank+nb_node-1                                                                 !$\
         rank_even=.TRUE.                                                                                 !$\
      else                                                                                                !$\
         comm_node(0)=STEP_Rank+nb_node-1                                                                 !$\
         comm_node(1)=STEP_Rank+1                                                                         !$\
         rank_even=.FALSE.                                                                                !$\
      endif                                                                                               !$\
      comm_node(0)=mod(comm_node(0),nb_node) ! cas d'une etape pair                                       !$\
      comm_node(1)=mod(comm_node(1),nb_node) ! cas d'une etape impair                                     !$\
      if (comm_node(0).GT.STEP_Rank) then ! cas STEP_Rank pair                                            !$\
         if(types(STEP_Rank).NE.MPI_DATATYPE_NULL) then                                                   !$\
            internalTypes=types(STEP_Rank)                                                                !$\
            internalComm_node=comm_node(0)                                                                !$\
            call MPI_SEND(array(origines(STEP_Rank)),1,                                                   !$\
     &           internalTypes,internalComm_node,internalTag,                                             !$\
     &           MPI_COMM_WORLD,iErr)                                                                     !$\
            types(STEP_Rank)=internalTypes                                                                !$\
            comm_node(0)=internalComm_node                                                                !$\
         endif                                                                                            !$\
         if(types(comm_node(0)).NE.MPI_DATATYPE_NULL) then                                                !$\
            internalTypes=types(comm_node(0))                                                             !$\
            internalComm_node=comm_node(0)                                                                !$\
            call MPI_RECV(BUFFER(origines(comm_node(0))),1,                                               !$\
     &           internalTypes,internalComm_node,internalTag,                                             !$\
     &           MPI_COMM_WORLD,STATUS,iErr)                                                              !$\
            types(comm_node(0))=internalTypes                                                             !$\
            comm_node(0)=internalComm_node                                                                !$\
#if INTERLACED                                                                                            !$\
            call MergeRegion/**/TYPE_SUFFIX(dim,                                                          !$\
     &           nb_regions,regions,comm_node(0)+1,                                                       !$\
     &           size,initial,buffer,array)                                                               !$\
#endif                                                                                                    !$\
         endif                                                                                            !$\
         data_recv=STEP_Rank                                                                              !$\
      else                                                                                                !$\
         if(types(comm_node(0)).NE.MPI_DATATYPE_NULL) then                                                !$\
            internalTypes=types(comm_node(0))                                                             !$\
            internalComm_node=comm_node(0)                                                                !$\
            call MPI_RECV(BUFFER(origines(comm_node(0))),1,                                               !$\
     &           internalTypes,internalComm_node,internalTag,                                             !$\
     &           MPI_COMM_WORLD,STATUS,iErr)                                                              !$\
            types(comm_node(0))=internalTypes                                                             !$\
            comm_node(0)=internalComm_node                                                                !$\
#if INTERLACED                                                                                            !$\
            call MergeRegion/**/TYPE_SUFFIX(dim,                                                          !$\
     &           nb_regions,regions,comm_node(0)+1,                                                       !$\
     &           size,initial,buffer,array)                                                               !$\
#endif                                                                                                    !$\
         endif                                                                                            !$\
         if(types(STEP_Rank).NE.MPI_DATATYPE_NULL) then                                                   !$\
            internalTypes=types(STEP_Rank)                                                                !$\
            internalComm_node=comm_node(0)                                                                !$\
            call MPI_SEND(array(origines(STEP_Rank)),1,                                                   !$\
     &           internalTypes,internalComm_node,internalTag,                                             !$\
     &           MPI_COMM_WORLD,iErr)                                                                     !$\
            types(STEP_Rank)=internalTypes                                                                !$\
            comm_node(0)=internalComm_node                                                                !$\
         endif                                                                                            !$\
         data_recv=STEP_Rank-1                                                                            !$\
      endif                                                                                               !$\
      m=0                                                                                                 !$\
      step_even=.TRUE.                                                                                    !$\
      do step=1,nb_node/2-1                                                                               !$\
         m=1-m                                                                                            !$\
         step_even=.NOT. step_even                                                                        !$\
         data_send=data_recv                                                                              !$\
         if(rank_even.EQV.step_even)then                                                                  !$\
            data_recv=mod(STEP_Rank+step,nb_node)                                                         !$\
         else                                                                                             !$\
            data_recv=mod(STEP_Rank-step-1+nb_node,nb_node)                                               !$\
         endif                                                                                            !$\
         if (comm_node(m).GT.STEP_Rank) then ! cas STEP_Rank pair                                         !$\
            if(types(data_send).NE.MPI_DATATYPE_NULL) then                                                !$\
            internalTypes=types(data_send)                                                                !$\
            internalComm_node=comm_node(m)                                                                !$\
            call MPI_SEND(array(origines(data_send)),1,                                                   !$\
     &           internalTypes,internalComm_node,internalTag,                                             !$\
     &           MPI_COMM_WORLD,iErr)                                                                     !$\
            types(data_send)=internalTypes                                                                !$\
            comm_node(m)=internalComm_node                                                                !$\
         endif                                                                                            !$\
         if(types(data_send+1).NE.MPI_DATATYPE_NULL) then                                                 !$\
            internalTypes=types(data_send+1)                                                              !$\
            internalComm_node=comm_node(m)                                                                !$\
            call MPI_SEND(array(origines(data_send+1)),1,                                                 !$\
     &           internalTypes,internalComm_node,internalTag,                                             !$\
     &           MPI_COMM_WORLD,iErr)                                                                     !$\
            types(data_send+1)=internalTypes                                                              !$\
            comm_node(m)=internalComm_node                                                                !$\
         endif                                                                                            !$\
         if(types(data_recv).NE.MPI_DATATYPE_NULL) then                                                   !$\
            internalTypes=types(data_recv)                                                                !$\
            internalComm_node=comm_node(m)                                                                !$\
            call MPI_RECV(BUFFER(origines(data_recv)),1,                                                  !$\
     &           internalTypes,internalComm_node,internalTag,                                             !$\
     &           MPI_COMM_WORLD,STATUS,iErr)                                                              !$\
            types(data_recv)=internalTypes                                                                !$\
            comm_node(m)=internalComm_node                                                                !$\
#if INTERLACED                                                                                            !$\
            call MergeRegion/**/TYPE_SUFFIX(dim,                                                          !$\
     &     nb_regions,regions,data_recv+1,                                                                !$\
     &     size,initial,buffer,array)                                                                     !$\
#endif                                                                                                    !$\
         endif                                                                                            !$\
         if(types(data_recv+1).NE.MPI_DATATYPE_NULL) then                                                 !$\
            internalTypes=types(data_recv+1)                                                              !$\
            internalComm_node=comm_node(m)                                                                !$\
            call MPI_RECV(BUFFER(origines(data_recv+1)),1,                                                !$\
     &           internalTypes,internalComm_node,internalTag,                                             !$\
     &           MPI_COMM_WORLD,STATUS,iErr)                                                              !$\
            types(data_recv+1)=internalTypes                                                              !$\
            comm_node(m)=internalComm_node                                                                !$\
#if INTERLACED                                                                                            !$\
            call MergeRegion/**/TYPE_SUFFIX(dim,                                                          !$\
     &     nb_regions,regions,data_recv+2,                                                                !$\
     &     size,initial,buffer,array)                                                                     !$\
#endif                                                                                                    !$\
         endif                                                                                            !$\
         else                                                                                             !$\
            if(types(data_recv).NE.MPI_DATATYPE_NULL) then                                                !$\
               internalTypes=types(data_recv)                                                             !$\
               internalComm_node=comm_node(m)                                                             !$\
               call MPI_RECV(BUFFER(origines(data_recv)),1,                                               !$\
     &              internalTypes,internalComm_node,internalTag,                                          !$\
     &              MPI_COMM_WORLD,STATUS,iErr)                                                           !$\
               types(data_recv)=internalTypes                                                             !$\
               comm_node(m)=internalComm_node                                                             !$\
#if INTERLACED                                                                                            !$\
               call MergeRegion/**/TYPE_SUFFIX(dim,                                                       !$\
     &              nb_regions,regions,data_recv+1,                                                       !$\
     &              size,initial,buffer,array)                                                            !$\
#endif                                                                                                    !$\
            endif                                                                                         !$\
            if(types(data_recv+1).NE.MPI_DATATYPE_NULL) then                                              !$\
               internalTypes=types(data_recv+1)                                                           !$\
               internalComm_node=comm_node(m)                                                             !$\
               call MPI_RECV(BUFFER(origines(data_recv+1)),1,                                             !$\
     &              internalTypes,internalComm_node,internalTag,                                          !$\
     &              MPI_COMM_WORLD,STATUS,iErr)                                                           !$\
               types(data_recv+1)=internalTypes                                                           !$\
               comm_node(m)=internalComm_node                                                             !$\
#if INTERLACED                                                                                            !$\
               call MergeRegion/**/TYPE_SUFFIX(dim,                                                       !$\
     &              nb_regions,regions,data_recv+2,                                                       !$\
     &              size,initial,buffer,array)                                                            !$\
#endif                                                                                                    !$\
            endif                                                                                         !$\
            if(types(data_send).NE.MPI_DATATYPE_NULL) then                                                !$\
               internalTypes=types(data_send)                                                             !$\
               internalComm_node=comm_node(m)                                                             !$\
               call MPI_SEND(array(origines(data_send)),1,                                                !$\
     &              internalTypes,internalComm_node,internalTag,                                          !$\
     &              MPI_COMM_WORLD,iErr)                                                                  !$\
               types(data_send)=internalTypes                                                             !$\
               comm_node(m)=internalComm_node                                                             !$\
            endif                                                                                         !$\
            if(types(data_send+1).NE.MPI_DATATYPE_NULL) then                                              !$\
               internalTypes=types(data_send+1)                                                           !$\
               internalComm_node=comm_node(m)                                                             !$\
               call MPI_SEND(array(origines(data_send+1)),1,                                              !$\
     &              internalTypes,internalComm_node,internalTag,                                          !$\
     &              MPI_COMM_WORLD,iErr)                                                                  !$\
               types(data_send+1)=internalTypes                                                           !$\
               comm_node(m)=internalComm_node                                                             !$\
            endif                                                                                         !$\
         endif                                                                                            !$\
      enddo                                                                                               !$\
      end                                                                                                 !$\
                                                                                                          !$\

#define  AlltoAll_even2(MERGE,TYPE_SUFFIX,TYPE_FORTRAN)\
#if INTERLACED                                                                                            !$\
#define BUFFER buffer                                                                                     !$\
#define PARAMETER_MERGE ,dim,nb_regions,regions,initial,buffer                                            !$\
#else                                                                                                     !$\
#define BUFFER array                                                                                      !$\
#define PARAMETER_MERGE                                                                                   !$\
#endif                                                                                                    !$\
      subroutine AlltoAll_even2/**/MERGE/**/TYPE_SUFFIX(nb_node,types,origines,                           !$\
     &     size,array,tag PARAMETER_MERGE)                                                                !$\
      implicit none                                                                                       !$\
      include 'mpif.h'                                                                                    !$\
      include 'STEP.h'                                                                                    !$\
      include 'steprt.h'                                                                                  !$\
      integer nb_node                                                                                     !$\
      integer types(0:STEP_MAX_NBNODE-1)                                                                  !$\
      integer origines(0:STEP_MAX_NBNODE-1),size                                                          !$\
      TYPE_FORTRAN array(size)                                                                            !$\
      integer tag                                                                                         !$\
#if INTERLACED                                                                                            !$\
      integer dim                                                                                         !$\
      integer L,U,B_                                                                                      !$\
      parameter (L=1,U=2,B_=0)                                                                            !$\
      integer nb_regions        !nombre d'élément du tableau région +1                                    !$\
      integer regions(L:U,1:dim,B_:nb_regions) !description de l'espace d'indice des différentes régions  !$\
                                !regions(:,:,B_) decrit l'espace d'indice du tableau array non linéarisé  !$\
      TYPE_FORTRAN initial(1:size),buffer(1:size)                                                         !$\
#endif                                                                                                    !$\
      integer id_data                                                                                     !$\
      integer data_send,data_recv,incr                                                                    !$\
      integer comm                                                                                        !$\
      integer step,m                                                                                      !$\
      integer comm_node(0:1)                                                                              !$\
      logical rank_even,step_even                                                                         !$\
      integer*4 STATUS(MPI_STATUS_SIZE)                                                                   !$\
      integer*4 internalTag                                                                               !$\
      integer*4 node,iErr                                                                                 !$\
      integer*4 internalRequests                                                                          !$\
      integer*4 internalTypes1                                                                            !$\
      integer*4 internalTypes2                                                                            !$\
      integer*4 internalComm_node1                                                                        !$\
      integer*4 internalComm_node2                                                                        !$\
                                                                                                          !$\
      internalTag=tag                                                                                     !$\
      step=0                                                                                              !$\
      if (mod(STEP_Rank,2).EQ.0) then !calcule du noeud avec lequel on communique a l'etape step          !$\
         comm_node(0)=STEP_Rank+1                                                                         !$\
         comm_node(1)=STEP_Rank+nb_node-1                                                                 !$\
         rank_even=.TRUE.                                                                                 !$\
         data_recv=STEP_Rank                                                                              !$\
      else                                                                                                !$\
         comm_node(0)=STEP_Rank+nb_node-1                                                                 !$\
         comm_node(1)=STEP_Rank+1                                                                         !$\
         rank_even=.FALSE.                                                                                !$\
         data_recv=STEP_Rank-1                                                                            !$\
      endif                                                                                               !$\
      comm_node(0)=mod(comm_node(0),nb_node) ! cas d'une etape pair                                       !$\
      comm_node(1)=mod(comm_node(1),nb_node) ! cas d'une etape impair                                     !$\
      comm=0                                                                                              !$\
      if (types(STEP_Rank).NE.MPI_DATATYPE_NULL)then                                                      !$\
         comm=comm+2                                                                                      !$\
      endif                                                                                               !$\
      if (types(comm_node(0)).NE.MPI_DATATYPE_NULL)then                                                   !$\
         comm=comm+1                                                                                      !$\
      endif                                                                                               !$\
      select case(comm)                                                                                   !$\
      case(3)                                                                                             !$\
         internalTypes1=types(STEP_Rank)                                                                  !$\
         internalComm_node1=comm_node(0)                                                                  !$\
         internalTypes2=types(comm_node(0))                                                               !$\
         internalComm_node2=comm_node(0)                                                                  !$\
         call MPI_SENDRECV(array(origines(STEP_Rank)),1,                                                  !$\
     &        internalTypes1,internalComm_node1,internalTag,                                              !$\
     &        BUFFER(origines(comm_node(0))),1,                                                           !$\
     &        internalTypes2,internalComm_node2,internalTag,                                              !$\
     &        MPI_COMM_WORLD,STATUS,iErr)                                                                 !$\
         types(STEP_Rank)=internalTypes1                                                                  !$\
         comm_node(0)=internalComm_node1                                                                  !$\
         types(comm_node(0))=internalTypes2                                                               !$\
#if INTERLACED                                                                                            !$\
         call MergeRegion/**/TYPE_SUFFIX(dim,                                                             !$\
     &        nb_regions,regions,comm_node(0)+1,                                                          !$\
     &        size,initial,buffer,array)                                                                  !$\
#endif                                                                                                    !$\
      case(2)                                                                                             !$\
         internalTypes1=types(STEP_Rank)                                                                  !$\
         internalComm_node1=comm_node(0)                                                                  !$\
         call MPI_SEND(array(origines(STEP_Rank)),1,                                                      !$\
     &        internalTypes1,internalComm_node1,internalTag,                                              !$\
     &        MPI_COMM_WORLD,iErr)                                                                        !$\
         types(STEP_Rank)=internalTypes1                                                                  !$\
         comm_node(0)=internalComm_node1                                                                  !$\
      case(1)                                                                                             !$\
         internalTypes2=types(comm_node(0))                                                               !$\
         internalComm_node2=comm_node(0)                                                                  !$\
         call MPI_RECV(BUFFER(origines(comm_node(0))),1,                                                  !$\
     &        types(comm_node(0)),comm_node(0),tag,                                                       !$\
     &        MPI_COMM_WORLD,STATUS,iErr)                                                                 !$\
         types(comm_node(0))=internalTypes2                                                               !$\
         comm_node(0)=internalComm_node2                                                                  !$\
#if INTERLACED                                                                                            !$\
         call MergeRegion/**/TYPE_SUFFIX(dim,                                                             !$\
     &        nb_regions,regions,comm_node(0)+1,                                                          !$\
     &        size,initial,buffer,array)                                                                  !$\
#endif                                                                                                    !$\
      end select                                                                                          !$\
      m=0                                                                                                 !$\
      step_even=.TRUE.                                                                                    !$\
      do step=1,nb_node/2-1                                                                               !$\
         m=1-m                                                                                            !$\
         step_even=.NOT. step_even                                                                        !$\
         data_send=data_recv                                                                              !$\
         if(rank_even.EQV.step_even)then                                                                  !$\
            data_recv=mod(STEP_Rank+step,nb_node)                                                         !$\
         else                                                                                             !$\
            data_recv=mod(STEP_Rank-step-1+nb_node,nb_node)                                               !$\
         endif                                                                                            !$\
         comm=0                                                                                           !$\
         if (types(data_send).NE.MPI_DATATYPE_NULL)then                                                   !$\
            comm=comm+2                                                                                   !$\
         endif                                                                                            !$\
         if (types(data_recv).NE.MPI_DATATYPE_NULL)then                                                   !$\
            comm=comm+1                                                                                   !$\
         endif                                                                                            !$\
         select case(comm)                                                                                !$\
         case(3)                                                                                          !$\
            internalTypes1=types(data_send)                                                               !$\
            internalComm_node1=comm_node(m)                                                               !$\
            internalTypes2=types(data_recv)                                                               !$\
            internalComm_node2=comm_node(0)                                                               !$\
            call MPI_SENDRECV(array(origines(data_send)),1,                                               !$\
     &           internalTypes1,internalComm_node1,internalTag,                                           !$\
     &           BUFFER(origines(data_recv)),1,                                                           !$\
     &           internalTypes2,internalComm_node2,internalTag,                                           !$\
     &           MPI_COMM_WORLD,STATUS,iErr)                                                              !$\
            types(data_send)=internalTypes1                                                               !$\
            comm_node(m)=internalComm_node1                                                               !$\
            types(data_recv)=internalTypes2                                                               !$\
            comm_node(0)=internalComm_node2                                                               !$\
#if INTERLACED                                                                                            !$\
            call MergeRegion/**/TYPE_SUFFIX(dim,                                                          !$\
     &           nb_regions,regions,data_recv+1,                                                          !$\
     &           size,initial,buffer,array)                                                               !$\
#endif                                                                                                    !$\
         case(2)                                                                                          !$\
            internalTypes1=types(data_send)                                                               !$\
            internalComm_node1=comm_node(m)                                                               !$\
            call MPI_SEND(array(origines(data_send)),1,                                                   !$\
     &           internalTypes1,internalComm_node1,internalTag,                                           !$\
     &           MPI_COMM_WORLD,iErr)                                                                     !$\
            types(data_send)=internalTypes1                                                               !$\
            comm_node(m)=internalComm_node1                                                               !$\
         case(1)                                                                                          !$\
            internalTypes2=types(data_recv)                                                               !$\
            internalComm_node2=comm_node(m)                                                               !$\
            call MPI_RECV(BUFFER(origines(data_recv)),1,                                                  !$\
     &           internalTypes2,internalComm_node2,internalTag,                                           !$\
     &           MPI_COMM_WORLD,STATUS,iErr)                                                              !$\
            types(data_recv)=internalTypes2                                                               !$\
            comm_node(m)=internalComm_node2                                                               !$\
#if INTERLACED                                                                                            !$\
            call MergeRegion/**/TYPE_SUFFIX(dim,                                                          !$\
     &           nb_regions,regions,data_recv+1,                                                          !$\
     &           size,initial,buffer,array)                                                               !$\
#endif                                                                                                    !$\
         end select                                                                                       !$\
         comm=0                                                                                           !$\
         if (types(data_send+1).NE.MPI_DATATYPE_NULL)then                                                 !$\
            comm=comm+2                                                                                   !$\
         endif                                                                                            !$\
         if (types(data_recv+1).NE.MPI_DATATYPE_NULL)then                                                 !$\
            comm=comm+1                                                                                   !$\
         endif                                                                                            !$\
         select case(comm)                                                                                !$\
         case(3)                                                                                          !$\
            internalTypes1=types(data_send+1)                                                             !$\
            internalComm_node1=comm_node(m)                                                               !$\
            internalTypes2=types(data_recv+1)                                                             !$\
            internalComm_node2=comm_node(m)                                                               !$\
            call MPI_SENDRECV(array(origines(data_send+1)),1,                                             !$\
     &           internalTypes1,internalComm_node1,internalTag,                                           !$\
     &           BUFFER(origines(data_recv+1)),1,                                                         !$\
     &           internalTypes2,internalComm_node2,internalTag,                                           !$\
     &           MPI_COMM_WORLD,STATUS,iErr)                                                              !$\
            types(data_send+1)=internalTypes1                                                             !$\
            comm_node(m)=internalComm_node1                                                               !$\
            types(data_recv+1)=internalTypes2                                                             !$\
#if INTERLACED                                                                                            !$\
            call MergeRegion/**/TYPE_SUFFIX(dim,                                                          !$\
     &           nb_regions,regions,data_recv+2,                                                          !$\
     &           size,initial,buffer,array)                                                               !$\
#endif                                                                                                    !$\
         case(2)                                                                                          !$\
            internalTypes1=types(data_send+1)                                                             !$\
            internalComm_node1=comm_node(m)                                                               !$\
            call MPI_SEND(array(origines(data_send+1)),1,                                                 !$\
     &           internalTypes1,internalComm_node1,internalTag,                                           !$\
     &           MPI_COMM_WORLD,iErr)                                                                     !$\
            types(data_send+1)=internalTypes1                                                             !$\
            comm_node(m)=internalComm_node1                                                               !$\
         case(1)                                                                                          !$\
            internalTypes2=types(data_recv+1)                                                             !$\
            internalComm_node2=comm_node(m)                                                               !$\
            call MPI_RECV(BUFFER(origines(data_recv+1)),1,                                                !$\
     &           internalTypes2,internalComm_node2,internalTag,                                           !$\
     &           MPI_COMM_WORLD,STATUS,iErr)                                                              !$\
            types(data_recv+1)=internalTypes2                                                             !$\
            comm_node(m)=internalComm_node2                                                               !$\
#if INTERLACED                                                                                            !$\
            call MergeRegion/**/TYPE_SUFFIX(dim,                                                          !$\
     &           nb_regions,regions,data_recv+2,                                                          !$\
     &           size,initial,buffer,array)                                                               !$\
#endif                                                                                                    !$\
         end select                                                                                       !$\
      enddo                                                                                               !$\
      end                                                                                                 !$\
                                                                                                          !$\

#define AlltoAll_odd(MERGE,TYPE_SUFFIX,TYPE_FORTRAN)\
#if INTERLACED                                                                                            !$\
#define BUFFER buffer                                                                                     !$\
#define PARAMETER_MERGE ,dim,nb_regions,regions,initial,buffer                                            !$\
#else                                                                                                     !$\
#define BUFFER array                                                                                      !$\
#define PARAMETER_MERGE                                                                                   !$\
#endif                                                                                                    !$\
      subroutine AlltoAll_odd/**/MERGE/**/TYPE_SUFFIX(nb_node,types,origines,                             !$\
     &     size,array,tag PARAMETER_MERGE)                                                                !$\
      implicit none                                                                                       !$\
      include 'mpif.h'                                                                                    !$\
      include 'STEP.h'                                                                                    !$\
      include 'steprt.h'                                                                                  !$\
      integer nb_node                                                                                     !$\
      integer types(0:nb_node-1)                                                                          !$\
      integer origines(0:nb_node-1),size                                                                  !$\
      TYPE_FORTRAN array(size)                                                                            !$\
      integer tag                                                                                         !$\
#if INTERLACED                                                                                            !$\
      integer dim                                                                                         !$\
      integer L,U,B_                                                                                      !$\
      parameter (L=1,U=2,B_=0)                                                                            !$\
      integer nb_regions        !nombre d'élément du tableau région +1                                    !$\
      integer regions(L:U,1:dim,B_:nb_regions) !description de l'espace d'indice des différentes régions  !$\
                                !regions(:,:,B_) decrit l'espace d'indice du tableau array non linéarisé  !$\
      TYPE_FORTRAN initial(1:size),buffer(1:size)                                                         !$\
#endif                                                                                                    !$\
      integer comm_node(0:STEP_MAX_NBNODE-1),alias,step                                                   !$\
      integer node                                                                                        !$\
      integer*4 STATUS(MPI_STATUS_SIZE),iErr                                                              !$\
      integer*4 internalTag                                                                               !$\
      integer*4 internalNode                                                                              !$\
      integer*4 internalTypes                                                                             !$\
                                                                                                          !$\
      internalTag=tag                                                                                     !$\
      comm_node(0)=MPI_PROC_NULL                                                                          !$\
      do node=1,nb_node-1                                                                                 !$\
         comm_node(node)=2*nb_node-node                                                                   !$\
      enddo                                                                                               !$\
      do step=0,nb_node-1                                                                                 !$\
         alias=mod(STEP_Rank+step,nb_node)                                                                !$\
         if (alias.NE.0)then                                                                              !$\
            node=mod(comm_node(alias)-step,nb_node)                                                       !$\
            if(node.LT.STEP_Rank) then                                                                    !$\
               if(types(STEP_Rank).NE.MPI_DATATYPE_NULL) then                                             !$\
                  internalTypes=types(STEP_Rank)                                                          !$\
                  internalNode=node                                                                       !$\
                  call MPI_SEND(array(origines(STEP_Rank)),1,                                             !$\
     &                 internalTypes,internalNode,internalTag,                                            !$\
     &                 MPI_COMM_WORLD,iErr)                                                               !$\
                  types(STEP_Rank)=internalTypes                                                          !$\
                  node=internalNode                                                                       !$\
               endif                                                                                      !$\
               if(types(node).NE.MPI_DATATYPE_NULL) then                                                  !$\
                  internalTypes=types(node)                                                               !$\
                  internalNode=node                                                                       !$\
                  call MPI_RECV(BUFFER(origines(node)),1,                                                 !$\
     &                 internalTypes,internalNode,internalTag,                                            !$\
     &                 MPI_COMM_WORLD,STATUS,iErr)                                                        !$\
                  types(node)=internalTypes                                                               !$\
                  node=internalNode                                                                       !$\
#if INTERLACED                                                                                            !$\
                  call MergeRegion/**/TYPE_SUFFIX(dim,                                                    !$\
     &                 nb_regions,regions,node+1,                                                         !$\
     &                 size,initial,buffer,array)                                                         !$\
#endif                                                                                                    !$\
               endif                                                                                      !$\
            else                                                                                          !$\
               if(types(node).NE.MPI_DATATYPE_NULL) then                                                  !$\
                  internalTypes=types(node)                                                               !$\
                  internalNode=node                                                                       !$\
                  call MPI_RECV(BUFFER(origines(node)),1,                                                 !$\
     &                 internalTypes,internalNode,internalTag,                                            !$\
     &                 MPI_COMM_WORLD,STATUS,iErr)                                                        !$\
                  types(node)=internalTypes                                                               !$\
                  node=internalNode                                                                       !$\
#if INTERLACED                                                                                            !$\
                  call MergeRegion/**/TYPE_SUFFIX(dim,                                                    !$\
     &                 nb_regions,regions,node+1,                                                         !$\
     &                 size,initial,buffer,array)                                                         !$\
#endif                                                                                                    !$\
               endif                                                                                      !$\
               if(types(STEP_Rank).NE.MPI_DATATYPE_NULL) then                                             !$\
                  internalTypes=types(STEP_Rank)                                                          !$\
                  internalNode=node                                                                       !$\
                  call MPI_SEND(array(origines(STEP_Rank)),1,                                             !$\
     &                 internalTypes,internalNode,internalTag,                                            !$\
     &                 MPI_COMM_WORLD,iErr)                                                               !$\
                  types(STEP_Rank)=internalTypes                                                          !$\
                  node=internalNode                                                                       !$\
                endif                                                                                     !$\
            endif                                                                                         !$\
         endif                                                                                            !$\
      enddo                                                                                               !$\
      end                                                                                                 !$\
                                                                                                          !$\

#define AlltoAll_odd2(MERGE,TYPE_SUFFIX,TYPE_FORTRAN)\
#if INTERLACED                                                                                            !$\
#define BUFFER buffer                                                                                     !$\
#define PARAMETER_MERGE ,dim,nb_regions,regions,initial,buffer                                            !$\
#else                                                                                                     !$\
#define BUFFER array                                                                                      !$\
#define PARAMETER_MERGE                                                                                   !$\
#endif                                                                                                    !$\
      subroutine AlltoAll_odd2/**/MERGE/**/TYPE_SUFFIX(nb_node,types,origines,                            !$\
     &     size,array,tag PARAMETER_MERGE)                                                                !$\
      implicit none                                                                                       !$\
      include 'mpif.h'                                                                                    !$\
      include 'STEP.h'                                                                                    !$\
      include 'steprt.h'                                                                                  !$\
      integer nb_node                                                                                     !$\
      integer types(0:nb_node-1)                                                                          !$\
      integer origines(0:nb_node-1),size                                                                  !$\
      TYPE_FORTRAN array(size)                                                                            !$\
      integer tag                                                                                         !$\
#if INTERLACED                                                                                            !$\
      integer dim                                                                                         !$\
      integer L,U,B_                                                                                      !$\
      parameter (L=1,U=2,B_=0)                                                                            !$\
      integer nb_regions        !nombre d'élément du tableau région +1                                    !$\
      integer regions(L:U,1:dim,B_:nb_regions) !description de l'espace d'indice des différentes régions  !$\
                                !regions(:,:,B_) decrit l'espace d'indice du tableau array non linéarisé  !$\
      TYPE_FORTRAN initial(1:size),buffer(1:size)                                                         !$\
#endif                                                                                                    !$\
      integer comm_node(0:STEP_MAX_NBNODE-1),alias,step,comm                                              !$\
      integer node                                                                                        !$\
      integer*4 STATUS(MPI_STATUS_SIZE),iErr                                                              !$\
      integer*4 internalTag                                                                               !$\
      integer*4 internalNode1,internalNode2                                                               !$\
      integer*4 internalTypes1,internalTypes2                                                             !$\
                                                                                                          !$\
      internalTag=tag                                                                                     !$\
      comm_node(0)=MPI_PROC_NULL                                                                          !$\
      do node=1,nb_node-1                                                                                 !$\
         comm_node(node)=2*nb_node-node                                                                   !$\
      enddo                                                                                               !$\
      do step=0,nb_node-1                                                                                 !$\
         alias=mod(STEP_Rank+step,nb_node)                                                                !$\
         if (alias.NE.0)then                                                                              !$\
            node=mod(comm_node(alias)-step,nb_node)                                                       !$\
            comm=0                                                                                        !$\
            if (types(STEP_Rank).NE.MPI_DATATYPE_NULL)then                                                !$\
               comm=comm+2                                                                                !$\
            endif                                                                                         !$\
            if (types(node).NE.MPI_DATATYPE_NULL)then                                                     !$\
               comm=comm+1                                                                                !$\
            endif                                                                                         !$\
            select case(comm)                                                                             !$\
            case(3)                                                                                       !$\
               internalTypes1=types(STEP_Rank)                                                            !$\
               internalNode1=node                                                                         !$\
               internalTypes2=types(node)                                                                 !$\
               internalNode2=node                                                                         !$\
               call MPI_SENDRECV(                                                                         !$\
     &              array(origines(STEP_Rank)),1,                                                         !$\
     &              internalTypes1,internalNode1,internalTag,                                             !$\
     &              types(STEP_Rank),node,tag,                                                            !$\
     &              BUFFER(origines(node)),1,                                                             !$\
     &              internalTypes2,internalNode2,internalTag,                                             !$\
     &              MPI_COMM_WORLD,STATUS,iErr)                                                           !$\
               types(STEP_Rank)=internalTypes1                                                            !$\
               node=internalNode1                                                                         !$\
               types(node)=internalTypes2                                                                 !$\
#if INTERLACED                                                                                            !$\
               call MergeRegion/**/TYPE_SUFFIX(dim,                                                       !$\
     &              nb_regions,regions,node+1,                                                            !$\
     &              size,initial,buffer,array)                                                            !$\
#endif                                                                                                    !$\
            case(2)                                                                                       !$\
               internalTypes1=types(STEP_Rank)                                                            !$\
               internalNode1=node                                                                         !$\
               call MPI_SEND(                                                                             !$\
     &              array(origines(STEP_Rank)),1,                                                         !$\
     &              internalTypes1,internalNode1,internalTag,                                             !$\
     &              MPI_COMM_WORLD,iErr)                                                                  !$\
               types(STEP_Rank)=internalTypes1                                                            !$\
               node=internalNode1                                                                         !$\
            case(1)                                                                                       !$\
               internalTypes2=types(node)                                                                 !$\
               internalNode2=node                                                                         !$\
               call MPI_RECV(                                                                             !$\
     &              BUFFER(origines(node)),1,                                                             !$\
     &              internalTypes2,internalNode2,internalTag,                                             !$\
     &              MPI_COMM_WORLD,STATUS,iErr)                                                           !$\
               types(node)=internalTypes2                                                                 !$\
               node=internalNode2                                                                         !$\
#if INTERLACED                                                                                            !$\
               call MergeRegion/**/TYPE_SUFFIX(dim,                                                       !$\
     &              nb_regions,regions,node+1,                                                            !$\
     &              size,initial,buffer,array)                                                            !$\
#endif                                                                                                    !$\
            end select                                                                                    !$\
         endif                                                                                            !$\
      enddo                                                                                               !$\
      end                                                                                                 !$\
                                                                                                          !$\

#define STEP_InitInterlaced(TYPE_SUFFIX,TYPE_FORTRAN)\
      subroutine STEP_InitInterlaced/**/TYPE_SUFFIX(size,array,array_initial,array_buffer)                !$\
      implicit none                                                                                       !$\
      integer size,i                                                                                      !$\
      TYPE_FORTRAN array(size),array_initial(size),array_buffer(size)                                     !$\
      do i=1,size                                                                                         !$\
         array_initial(i)=array(i)                                                                        !$\
         array_buffer(i)=array(i)                                                                         !$\
      end do                                                                                              !$\
      end                                                                                                 !$\
                                                                                                          !$\

#define MergeRegion(TYPE_SUFFIX,TYPE_FORTRAN)\
      subroutine MergeRegion/**/TYPE_SUFFIX(dim,                                                          !$\
     &     nb_regions,regions,id_sub_region,                                                              !$\
     &     size,initial,in,out)                                                                           !$\
      implicit none                                                                                       !$\
      include 'mpif.h'                                                                                    !$\
      integer dim               !nombre de dimension du tableau array avant linéarisation                 !$\
      integer size              !taille du tableau array                                                  !$\
      TYPE_FORTRAN initial(1:size)   !tableau initial linéarisé                                           !$\
      TYPE_FORTRAN in(1:size)        !tableau in linéarisé                                                !$\
      TYPE_FORTRAN out(1:size)       !tableau out linéarisé                                               !$\
      integer L,U,B_                                                                                      !$\
      parameter (L=1,U=2,B_=0)                                                                            !$\
      integer nb_regions        !nombre d'élément du tableau région +1                                    !$\
      integer regions(L:U,1:dim,B_:nb_regions) !description de l'espace d'indice des différentes régions  !$\
                                !regions(:,:,B_) decrit l'espace d'indice du tableau array non linéarisé  !$\
      integer id_sub_region     !identifiant de la région devant être envoyé                              !$\
      integer profil(L:U,1:7)   !profile de la sous-region formate à 7 dimensions                         !$\
      integer d,d1,d2,d3,d4,d5,d6,d7 !indices de parcours des dimensions                                  !$\
      integer coords(1:7),indice,i                                                                        !$\
      integer*4 iErr                                                                                      !$\
                                                                                                          !$\
      !mise a jour du profile                                                                             !$\
      do 10 d=1,dim                                                                                       !$\
         profil(L,d)=regions(L,d,id_sub_region)                                                           !$\
         profil(U,d)=regions(U,d,id_sub_region)                                                           !$\
 10   continue                                                                                            !$\
      do 20 d=dim+1,7                                                                                     !$\
         profil(L,d)=0                                                                                    !$\
         profil(U,d)=0                                                                                    !$\
 20   continue                                                                                            !$\
! parcours de la sous_region                                                                              !$\
      do 30 d7=profil(L,7),profil(U,7)                                                                    !$\
         coords(7)=d7                                                                                     !$\
      do 30 d6=profil(L,6),profil(U,6)                                                                    !$\
         coords(6)=d6                                                                                     !$\
      do 30 d5=profil(L,5),profil(U,5)                                                                    !$\
         coords(5)=d5                                                                                     !$\
      do 30 d4=profil(L,4),profil(U,4)                                                                    !$\
         coords(4)=d4                                                                                     !$\
      do 30 d3=profil(L,3),profil(U,3)                                                                    !$\
         coords(3)=d3                                                                                     !$\
      do 30 d2=profil(L,2),profil(U,2)                                                                    !$\
         coords(2)=d2                                                                                     !$\
      do 30 d1=profil(L,1),profil(U,1)                                                                    !$\
         coords(1)=d1                                                                                     !$\
         !comparaison case à case pour la sous-region                                                     !$\
         i=indice(dim,coords,regions(L,1,B_))                                                             !$\
!         print *,'i',i,' coord :',coords,' comp : ',initial(i),                                          !$\
!     &        in(i),out(i)                                                                               !$\
         if((in(i).NE.initial(i)).AND.(in(i).NE.out(i)) ) then                                            !$\
            if(out(i).NE.initial(i)) then                                                                 !$\
               print *,'\n** Concurrent Access **\n'                                                      !$\
               call MPI_ABORT(MPI_COMM_WORLD,2,iErr)                                                      !$\
            else                                                                                          !$\
               out(i)=in(i)                                                                               !$\
            endif                                                                                         !$\
         endif                                                                                            !$\
 30   continue                                                                                            !$\
      end                                                                                                 !$\
                                                                                                          !$\


#if INTERLACED
#include "COM_SUBROUTINES_INTERLACED.f"
#else
#include "COM_SUBROUTINES.f"
#endif
