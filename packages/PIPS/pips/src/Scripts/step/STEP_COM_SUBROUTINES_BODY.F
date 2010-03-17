#ifdef
* Copyright 2007, 2008 Alain Muller, Frederique Silber-Chaussumier
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
      include 'STEP.h'                                                                                    !$\
      TYPE_FORTRAN variable,variable_reduc                                                                !$\
      integer operator                                                                                    !$\
      variable_reduc=variable                                                                             !$\
      select case(operator)                                                                               !$\
      case(STEP_SUM)                                                                                      !$\
         variable=0                                                                                       !$\
      case(STEP_PROD)                                                                                     !$\
         variable=1                                                                                       !$\
      end select                                                                                          !$\
      end                                                                                                 !$\


#define Reducion(TYPE_SUFFIX,TYPE_FORTRAN,TYPE_MPI)\
      subroutine STEP_Reduction/**/TYPE_SUFFIX(variable, variable_reduc,                                  !$\
     &     operator)                                                                                      !$\
      implicit none                                                                                       !$\
      include 'STEP.h'                                                                                    !$\
      TYPE_FORTRAN variable,variable_reduc,tmp_result                                                     !$\
      integer*4 operator,iErr                                                                               !$\
      call MPI_ALLREDUCE (variable,tmp_result,1,TYPE_MPI,                                                 !$\
     &     operator,MPI_COMM_WORLD,iErr)                                                                  !$\
      select case(operator)                                                                               !$\
      case(STEP_SUM)                                                                                      !$\
         variable=variable_reduc+tmp_result                                                               !$\
      case(STEP_PROD)                                                                                     !$\
         variable=variable_reduc*tmp_result                                                               !$\
      end select                                                                                          !$\
      end                                                                                                 !$\

#define STEP_MasterToAllScalar(TYPE_SUFFIX,TYPE_FORTRAN,TYPE_MPI)\
      subroutine STEP_MasterToAllScalar/**/TYPE_SUFFIX(scalar,                                            !$\
     &     algorithm)                                                                                     !$\
      implicit none                                                                                       !$\
      include 'mpif.h'                                                                                    !$\
      TYPE_FORTRAN scalar                                                                                 !$\
      integer*4 iErr                                                                                        !$\
      integer algorithm                                                                                   !$\
      select case (algorithm)                                                                             !$\
      case DEFAULT                                                                                        !$\
         call MPI_BCAST(scalar,1,TYPE_MPI,0,MPI_COMM_WORLD,iErr)                                          !$\
      end select                                                                                          !$\
      end                                                                                                 !$\


#define STEP_MasterToAllRegion(TYPE_SUFFIX,TYPE_FORTRAN,TYPE_MPI)\
      subroutine STEP_MasterToAllRegion/**/TYPE_SUFFIX(array, !$\
     &     dim,regions,size,                                                                               !$\
     &     algorithm)                                                                                     !$\
      implicit none                                                                                       !$\
      include 'mpif.h'                                                                                    !$\
      integer dim,size                                                                                    !$\
      TYPE_FORTRAN array(1:size)     !tableau array linearise                                             !$\
      integer L,U,B_                                                                                         !$\
      parameter (L=1,U=2,B_=0)                                                                            !$\
      integer regions(L:U,1:dim,0:1) !description de l'espace d'indice des differentes regions                !$\
                                 !regions(:,:) decrit l'espace d'indice du tableau array non linearise    !$\
      integer*4 iErr                                                                                        !$\
      integer algorithm                                                                                   !$\
      integer STEP_Rank, STEP_Size                                                                        !$\
      integer origine           !resultat fonction de calcul                                              !$\
      integer type_sub_region   !resultat fonction de calcul                                              !$\
      integer STEP_SizeRegion   !resultat fonction de calcul                                              !$\
      integer*4 type                                                                                        !$\
      call type_subRegion(dim,                                                                            !$\
     &     regions(L,1,B_),                                                                               !$\
     &     regions(L,1,B_),                                                                               !$\
     &     TYPE_MPI,type)                                                                                 !$\
      if(type.NE.MPI_DATATYPE_NULL) then                                                                  !$\
         call MPI_TYPE_COMMIT(type,iErr)                                                                  !$\
      endif                                                                                               !$\
      select case (algorithm)                                                                             !$\
      case DEFAULT                                                                                        !$\
         call MPI_BCAST(array(origine(dim,regions(L,1,B_),                                                !$\
     &        regions(L,1,B_))),1,type,0,MPI_COMM_WORLD,iErr)                                             !$\
      end select                                                                                          !$\
      end                                                                                                 !$\


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
      include 'STEP.h'                                                                                    !$\
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
                                                                                                          !$\
      integer*4 types(STEP_MAX_NBNODE)                                                                      !$\
      integer origines(STEP_MAX_NBNODE)                                                                   !$\
      integer id_region,idProc                                                                            !$\
      integer*4 iErr,iBidon                                                                                 !$\
      integer origine           !resultat fonction de calcul                                              !$\
      integer type_sub_region   !resultat fonction de calcul                                              !$\
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
                                                                                                          !$\
      ! initialisation des types MPI et du tableau des origines des regions                               !$\
      do id_region=1,nb_regions                                                                           !$\
         origines(id_region)=origine(dim,regions(L,1,B_),                                                 !$\
     &        regions(L,1,id_region))                                                                     !$\
c         print *, STEP_Rank,") id_region=",id_region,                                                    !$\
c     &        "origine(id_region)=",origines(id_region)                                                  !$\
         call type_subRegion(dim,                                                                         !$\
     &     regions(L,1,B_),                                                                               !$\
     &     regions(L,1,id_region),                                                                        !$\
     &     TYPE_MPI,types(id_region))                                                                     !$\
         if(types(id_region).NE.MPI_DATATYPE_NULL) then                                                   !$\
            call MPI_TYPE_COMMIT(types(id_region),iErr)                                                   !$\
         endif                                                                                            !$\
      enddo                                                                                               !$\
c      print *,STEP_Rank,") types=", types                                                                !$\
                                                                                                          !$\
      SELECT CASE(algorithm)                                                                              !$\
#ifndef INTERLACED                                                                                        !$\
      case (NBLOCKING)                                                                                    !$\
c      print *,STEP_Rank,') AlltoAll_nBlocking/**/TYPE_SUFFIX'                                            !$\
         call AlltoAll_nBlocking/**/TYPE_SUFFIX(STEP_Size,types,origines,                                 !$\
     &        STEP_SizeRegion(dim,regions(L,1,B_)),array,tag,                                             !$\
     &        max_nb_request,requests,nb_request)                                                         !$\
#endif                                                                                                    !$\
      case (BLOCKING1)                                                                                    !$\
         if(mod(STEP_SIZE,2).EQ.0)then                                                                    !$\
c      print *,STEP_Rank,') AlltoAll_even/**/MERGE/**/TYPE_SUFFIX'                                        !$\
            call AlltoAll_even/**/MERGE/**/TYPE_SUFFIX(STEP_Size,types,origines,                          !$\
     &           STEP_SizeRegion(dim,regions(L,1,B_)),array,tag                                           !$\
     &           PARAMETER_MERGE)                                                                         !$\
         else                                                                                             !$\
c      print *,STEP_Rank,') AlltoAll_odd/**/MERGE/**/TYPE_SUFFIX'                                         !$\
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
c      print *,STEP_Rank,') AlltoAll_odd2/**/MERGE/**/TYPE_SUFFIX'                                        !$\
            call AlltoAll_odd2/**/MERGE/**/TYPE_SUFFIX(STEP_Size,types,origines,                          !$\
     &           STEP_SizeRegion(dim,regions(L,1,B_)),array,tag                                           !$\
     &           PARAMETER_MERGE)                                                                         !$\
         endif                                                                                            !$\
      end select                                                                                          !$\
                                                                                                          !$\
      ! liberation de la memoire pour les type MPI                                                        !$\
      do id_region=1,nb_regions                                                                           !$\
         if (types(id_region).NE.MPI_DATATYPE_NULL) then                                                  !$\
            call MPI_Type_free(types(id_region),iErr);                                                    !$\
         endif                                                                                            !$\
      enddo                                                                                               !$\
      end                                                                                                 !$\

#define AlltoAll_nBlocking(TYPE_SUFFIX,TYPE_FORTRAN)\
      subroutine AlltoAll_nBlocking/**/TYPE_SUFFIX(nb_node,types,origines,                                !$\
     &     size,array,tag,max_nb_request,requests,nb_request)                                             !$\
      implicit none                                                                                       !$\
      include 'mpif.h'                                                                                    !$\
      integer nb_node                                                                                     !$\
      integer*4 types(0:nb_node-1)                                                                        !$\
      integer origines(0:nb_node-1),size                                                                  !$\
      TYPE_FORTRAN array(size)                                                                            !$\
      integer*4 node,iErr,STEP_Rank                                                                         !$\
      integer max_nb_request                                                                              !$\
      integer*4 requests(max_nb_request)                                                                    !$\
      integer nb_request                                                                                  !$\
      integer*4 tag                                                                                         !$\
                                                                                                          !$\
      CALL STEP_Get_rank(STEP_Rank)                                                                       !$\
                                                                                                          !$\
      do node=0,nb_node-1                                                                                 !$\
         if ((node.NE.STEP_Rank).AND.                                                                     !$\
     &        (types(node).NE.MPI_DATATYPE_NULL)) then                                                    !$\
            nb_request=nb_request+1                                                                       !$\
            call MPI_IRECV(array(origines(node)),1,types(node),node,                                      !$\
     &           tag, MPI_COMM_WORLD,requests(nb_request),iErr)                                           !$\
         endif                                                                                            !$\
      enddo                                                                                               !$\
                                                                                                          !$\
      if(types(STEP_Rank).NE.MPI_DATATYPE_NULL) then                                                      !$\
         do node=0,nb_node-1                                                                              !$\
            if (node.NE.STEP_Rank) then                                                                   !$\
               nb_request=nb_request+1                                                                    !$\
               call MPI_ISEND(array(origines(STEP_Rank)),1,                                               !$\
     &              types(STEP_Rank),node,                                                                !$\
     &              tag, MPI_COMM_WORLD,requests(nb_request),iErr)                                        !$\
            endif                                                                                         !$\
         enddo                                                                                            !$\
      endif                                                                                               !$\
      end                                                                                                 !$\

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
      integer nb_node                                                                                     !$\
      integer*4 types(0:nb_node-1)                                                                        !$\
      integer origines(0:nb_node-1),size                                                                  !$\
      TYPE_FORTRAN array(size)                                                                            !$\
      integer*4 tag                                                                                         !$\
#if INTERLACED                                                                                            !$\
      integer dim                                                                                         !$\
      integer L,U,B_                                                                                      !$\
      parameter (L=1,U=2,B_=0)                                                                            !$\
      integer nb_regions        !nombre d'élément du tableau région +1                                    !$\
      integer regions(L:U,1:dim,B_:nb_regions) !description de l'espace d'indice des différentes régions  !$\
                                !regions(:,:,B_) decrit l'espace d'indice du tableau array non linéarisé  !$\
      TYPE_FORTRAN initial(1:size),buffer(1:size)                                                         !$\
#endif                                                                                                    !$\
      integer STEP_Rank,id_data                                                                           !$\
      integer data_send,data_recv,incr                                                                    !$\
      integer*4 STATUS(MPI_STATUS_SIZE),iErr                                                              !$\
      integer step,m                                                                                      !$\
      integer*4 comm_node(0:1)                                                                            !$\
      logical rank_even,step_even                                                                         !$\
                                                                                                          !$\
      CALL STEP_Get_rank(STEP_Rank)                                                                       !$\
                                                                                                          !$\
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
                                                                                                          !$\
      if (comm_node(0).GT.STEP_Rank) then ! cas STEP_Rank pair                                            !$\
         if(types(STEP_Rank).NE.MPI_DATATYPE_NULL) then                                                   !$\
            call MPI_SEND(array(origines(STEP_Rank)),1,                                                   !$\
     &           types(STEP_Rank),comm_node(0),tag,                                                       !$\
     &           MPI_COMM_WORLD,iErr)                                                                     !$\
         endif                                                                                            !$\
         if(types(comm_node(0)).NE.MPI_DATATYPE_NULL) then                                                !$\
            call MPI_RECV(BUFFER(origines(comm_node(0))),1,                                               !$\
     &           types(comm_node(0)),comm_node(0),tag,                                                    !$\
     &           MPI_COMM_WORLD,STATUS,iErr)                                                              !$\
#if INTERLACED                                                                                            !$\
            call MergeRegion/**/TYPE_SUFFIX(dim,                                                          !$\
     &           nb_regions,regions,comm_node(0)+1,                                                       !$\
     &           size,initial,buffer,array)                                                               !$\
#endif                                                                                                    !$\
         endif                                                                                            !$\
         data_recv=STEP_Rank                                                                              !$\
      else                                                                                                !$\
         if(types(comm_node(0)).NE.MPI_DATATYPE_NULL) then                                                !$\
            call MPI_RECV(BUFFER(origines(comm_node(0))),1,                                               !$\
     &           types(comm_node(0)),comm_node(0),tag,                                                    !$\
     &           MPI_COMM_WORLD,STATUS,iErr)                                                              !$\
#if INTERLACED                                                                                            !$\
            call MergeRegion/**/TYPE_SUFFIX(dim,                                                          !$\
     &           nb_regions,regions,comm_node(0)+1,                                                       !$\
     &           size,initial,buffer,array)                                                               !$\
#endif                                                                                                    !$\
         endif                                                                                            !$\
         if(types(STEP_Rank).NE.MPI_DATATYPE_NULL) then                                                   !$\
            call MPI_SEND(array(origines(STEP_Rank)),1,                                                   !$\
     &           types(STEP_Rank),comm_node(0),tag,                                                       !$\
     &           MPI_COMM_WORLD,iErr)                                                                     !$\
         endif                                                                                            !$\
         data_recv=STEP_Rank-1                                                                            !$\
      endif                                                                                               !$\
                                                                                                          !$\
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
                                                                                                          !$\
         if (comm_node(m).GT.STEP_Rank) then ! cas STEP_Rank pair                                         !$\
            if(types(data_send).NE.MPI_DATATYPE_NULL) then                                                !$\
            call MPI_SEND(array(origines(data_send)),1,                                                   !$\
     &           types(data_send),comm_node(m),tag,                                                       !$\
     &           MPI_COMM_WORLD,iErr)                                                                     !$\
         endif                                                                                            !$\
         if(types(data_send+1).NE.MPI_DATATYPE_NULL) then                                                 !$\
            call MPI_SEND(array(origines(data_send+1)),1,                                                 !$\
     &           types(data_send+1),comm_node(m),tag,                                                     !$\
     &           MPI_COMM_WORLD,iErr)                                                                     !$\
         endif                                                                                            !$\
         if(types(data_recv).NE.MPI_DATATYPE_NULL) then                                                   !$\
            call MPI_RECV(BUFFER(origines(data_recv)),1,                                                  !$\
     &           types(data_recv),comm_node(m),tag,                                                       !$\
     &           MPI_COMM_WORLD,STATUS,iErr)                                                              !$\
#if INTERLACED                                                                                            !$\
            call MergeRegion/**/TYPE_SUFFIX(dim,                                                          !$\
     &     nb_regions,regions,data_recv+1,                                                                !$\
     &     size,initial,buffer,array)                                                                     !$\
#endif                                                                                                    !$\
         endif                                                                                            !$\
         if(types(data_recv+1).NE.MPI_DATATYPE_NULL) then                                                 !$\
            call MPI_RECV(BUFFER(origines(data_recv+1)),1,                                                !$\
     &           types(data_recv+1),comm_node(m),tag,                                                     !$\
     &           MPI_COMM_WORLD,STATUS,iErr)                                                              !$\
#if INTERLACED                                                                                            !$\
            call MergeRegion/**/TYPE_SUFFIX(dim,                                                          !$\
     &     nb_regions,regions,data_recv+2,                                                                !$\
     &     size,initial,buffer,array)                                                                     !$\
#endif                                                                                                    !$\
         endif                                                                                            !$\
         else                                                                                             !$\
            if(types(data_recv).NE.MPI_DATATYPE_NULL) then                                                !$\
               call MPI_RECV(BUFFER(origines(data_recv)),1,                                               !$\
     &              types(data_recv),comm_node(m),tag,                                                    !$\
     &              MPI_COMM_WORLD,STATUS,iErr)                                                           !$\
#if INTERLACED                                                                                            !$\
               call MergeRegion/**/TYPE_SUFFIX(dim,                                                       !$\
     &              nb_regions,regions,data_recv+1,                                                       !$\
     &              size,initial,buffer,array)                                                            !$\
#endif                                                                                                    !$\
            endif                                                                                         !$\
            if(types(data_recv+1).NE.MPI_DATATYPE_NULL) then                                              !$\
               call MPI_RECV(BUFFER(origines(data_recv+1)),1,                                             !$\
     &              types(data_recv+1),comm_node(m),tag,                                                  !$\
     &              MPI_COMM_WORLD,STATUS,iErr)                                                           !$\
#if INTERLACED                                                                                            !$\
               call MergeRegion/**/TYPE_SUFFIX(dim,                                                       !$\
     &              nb_regions,regions,data_recv+2,                                                       !$\
     &              size,initial,buffer,array)                                                            !$\
#endif                                                                                                    !$\
            endif                                                                                         !$\
            if(types(data_send).NE.MPI_DATATYPE_NULL) then                                                !$\
               call MPI_SEND(array(origines(data_send)),1,                                                !$\
     &              types(data_send),comm_node(m),tag,                                                    !$\
     &              MPI_COMM_WORLD,iErr)                                                                  !$\
            endif                                                                                         !$\
            if(types(data_send+1).NE.MPI_DATATYPE_NULL) then                                              !$\
               call MPI_SEND(array(origines(data_send+1)),1,                                              !$\
     &              types(data_send+1),comm_node(m),tag,                                                  !$\
     &              MPI_COMM_WORLD,iErr)                                                                  !$\
            endif                                                                                         !$\
         endif                                                                                            !$\
      enddo                                                                                               !$\
      end                                                                                                 !$\

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
      include 'STEP.h'                                                                                    !$\
      integer nb_node                                                                                     !$\
      integer*4 types(0:STEP_MAX_NBNODE-1)                                                                !$\
      integer origines(0:STEP_MAX_NBNODE-1),size                                                          !$\
      TYPE_FORTRAN array(size)                                                                            !$\
      integer*4 tag                                                                                         !$\
#if INTERLACED                                                                                            !$\
      integer dim                                                                                         !$\
      integer L,U,B_                                                                                      !$\
      parameter (L=1,U=2,B_=0)                                                                            !$\
      integer nb_regions        !nombre d'élément du tableau région +1                                    !$\
      integer regions(L:U,1:dim,B_:nb_regions) !description de l'espace d'indice des différentes régions  !$\
                                !regions(:,:,B_) decrit l'espace d'indice du tableau array non linéarisé  !$\
      TYPE_FORTRAN initial(1:size),buffer(1:size)                                                         !$\
#endif                                                                                                    !$\
      integer id_data                                                                           !$\
      integer data_send,data_recv,incr                                                                    !$\
      integer comm                                                                                        !$\
      integer*4 STATUS(MPI_STATUS_SIZE),iErr                                                                !$\
      integer step,m                                                                                      !$\
      integer*4 comm_node(0:1)                                                                            !$\
      logical rank_even,step_even                                                                         !$\
                                                                                                          !$\
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
                                                                                                          !$\
      comm=0                                                                                              !$\
      if (types(STEP_Rank).NE.MPI_DATATYPE_NULL)then                                                      !$\
         comm=comm+2                                                                                      !$\
      endif                                                                                               !$\
      if (types(comm_node(0)).NE.MPI_DATATYPE_NULL)then                                                   !$\
         comm=comm+1                                                                                      !$\
      endif                                                                                               !$\
      select case(comm)                                                                                   !$\
      case(3)                                                                                             !$\
         call MPI_SENDRECV(array(origines(STEP_Rank)),1,                                                  !$\
     &        types(STEP_Rank),comm_node(0),tag,                                                          !$\
     &        BUFFER(origines(comm_node(0))),1,                                                           !$\
     &        types(comm_node(0)),comm_node(0),tag,                                                       !$\
     &        MPI_COMM_WORLD,STATUS,iErr)                                                                 !$\
#if INTERLACED                                                                                            !$\
         call MergeRegion/**/TYPE_SUFFIX(dim,                                                             !$\
     &        nb_regions,regions,comm_node(0)+1,                                                          !$\
     &        size,initial,buffer,array)                                                                  !$\
#endif                                                                                                    !$\
      case(2)                                                                                             !$\
         call MPI_SEND(array(origines(STEP_Rank)),1,                                                      !$\
     &        types(STEP_Rank),comm_node(0),tag,                                                          !$\
     &        MPI_COMM_WORLD,iErr)                                                                        !$\
      case(1)                                                                                             !$\
         call MPI_RECV(BUFFER(origines(comm_node(0))),1,                                                  !$\
     &        types(comm_node(0)),comm_node(0),tag,                                                       !$\
     &        MPI_COMM_WORLD,STATUS,iErr)                                                                 !$\
#if INTERLACED                                                                                            !$\
         call MergeRegion/**/TYPE_SUFFIX(dim,                                                             !$\
     &        nb_regions,regions,comm_node(0)+1,                                                          !$\
     &        size,initial,buffer,array)                                                                  !$\
#endif                                                                                                    !$\
      end select                                                                                          !$\
                                                                                                          !$\
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
                                                                                                          !$\
         comm=0                                                                                           !$\
         if (types(data_send).NE.MPI_DATATYPE_NULL)then                                                   !$\
            comm=comm+2                                                                                   !$\
         endif                                                                                            !$\
         if (types(data_recv).NE.MPI_DATATYPE_NULL)then                                                   !$\
            comm=comm+1                                                                                   !$\
         endif                                                                                            !$\
         select case(comm)                                                                                !$\
         case(3)                                                                                          !$\
            call MPI_SENDRECV(array(origines(data_send)),1,                                               !$\
     &           types(data_send),comm_node(m),tag,                                                       !$\
     &           BUFFER(origines(data_recv)),1,                                                           !$\
     &           types(data_recv),comm_node(m),tag,                                                       !$\
     &           MPI_COMM_WORLD,STATUS,iErr)                                                              !$\
#if INTERLACED                                                                                            !$\
            call MergeRegion/**/TYPE_SUFFIX(dim,                                                          !$\
     &           nb_regions,regions,data_recv+1,                                                          !$\
     &           size,initial,buffer,array)                                                               !$\
#endif                                                                                                    !$\
         case(2)                                                                                          !$\
            call MPI_SEND(array(origines(data_send)),1,                                                   !$\
     &           types(data_send),comm_node(m),tag,                                                       !$\
     &           MPI_COMM_WORLD,iErr)                                                                     !$\
         case(1)                                                                                          !$\
            call MPI_RECV(BUFFER(origines(data_recv)),1,                                                  !$\
     &           types(data_recv),comm_node(m),tag,                                                       !$\
     &           MPI_COMM_WORLD,STATUS,iErr)                                                              !$\
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
            call MPI_SENDRECV(array(origines(data_send+1)),1,                                             !$\
     &           types(data_send+1),comm_node(m),tag,                                                     !$\
     &           BUFFER(origines(data_recv+1)),1,                                                         !$\
     &           types(data_recv+1),comm_node(m),tag,                                                     !$\
     &           MPI_COMM_WORLD,STATUS,iErr)                                                              !$\
#if INTERLACED                                                                                            !$\
            call MergeRegion/**/TYPE_SUFFIX(dim,                                                          !$\
     &           nb_regions,regions,data_recv+2,                                                          !$\
     &           size,initial,buffer,array)                                                               !$\
#endif                                                                                                    !$\
         case(2)                                                                                          !$\
            call MPI_SEND(array(origines(data_send+1)),1,                                                 !$\
     &           types(data_send+1),comm_node(m),tag,                                                     !$\
     &           MPI_COMM_WORLD,iErr)                                                                     !$\
         case(1)                                                                                          !$\
            call MPI_RECV(BUFFER(origines(data_recv+1)),1,                                                !$\
     &           types(data_recv+1),comm_node(m),tag,                                                     !$\
     &           MPI_COMM_WORLD,STATUS,iErr)                                                              !$\
#if INTERLACED                                                                                            !$\
            call MergeRegion/**/TYPE_SUFFIX(dim,                                                          !$\
     &           nb_regions,regions,data_recv+2,                                                          !$\
     &           size,initial,buffer,array)                                                               !$\
#endif                                                                                                    !$\
         end select                                                                                       !$\
      enddo                                                                                               !$\
      end                                                                                                 !$\

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
      include 'STEP.h'                                                                                    !$\
      integer nb_node                                                                                     !$\
      integer*4 types(0:nb_node-1)                                                                        !$\
      integer origines(0:nb_node-1),size                                                                  !$\
      TYPE_FORTRAN array(size)                                                                            !$\
      integer*4 tag                                                                                         !$\
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
      integer*4 node                                                                                      !$\
      integer*4 STATUS(MPI_STATUS_SIZE),iErr                                                              !$\
                                                                                                          !$\
      comm_node(0)=MPI_PROC_NULL                                                                          !$\
      do node=1,nb_node-1                                                                                 !$\
         comm_node(node)=2*nb_node-node                                                                   !$\
      enddo                                                                                               !$\
                                                                                                          !$\
      do step=0,nb_node-1                                                                                 !$\
         alias=mod(STEP_Rank+step,nb_node)                                                                !$\
         if (alias.NE.0)then                                                                              !$\
            node=mod(comm_node(alias)-step,nb_node)                                                       !$\
            if(node.LT.STEP_Rank) then                                                                    !$\
               if(types(STEP_Rank).NE.MPI_DATATYPE_NULL) then                                             !$\
                  call MPI_SEND(array(origines(STEP_Rank)),1,                                             !$\
     &                 types(STEP_Rank),node,tag,                                                         !$\
     &                 MPI_COMM_WORLD,iErr)                                                               !$\
               endif                                                                                      !$\
               if(types(node).NE.MPI_DATATYPE_NULL) then                                                  !$\
                  call MPI_RECV(BUFFER(origines(node)),1,                                                 !$\
     &                 types(node),node,tag,                                                              !$\
     &                 MPI_COMM_WORLD,STATUS,iErr)                                                        !$\
#if INTERLACED                                                                                            !$\
                  call MergeRegion/**/TYPE_SUFFIX(dim,                                                    !$\
     &                 nb_regions,regions,node+1,                                                         !$\
     &                 size,initial,buffer,array)                                                         !$\
#endif                                                                                                    !$\
               endif                                                                                      !$\
            else                                                                                          !$\
               if(types(node).NE.MPI_DATATYPE_NULL) then                                                  !$\
                  call MPI_RECV(BUFFER(origines(node)),1,                                                 !$\
     &                 types(node),node,tag,                                                              !$\
     &                 MPI_COMM_WORLD,STATUS,iErr)                                                        !$\
#if INTERLACED                                                                                            !$\
                  call MergeRegion/**/TYPE_SUFFIX(dim,                                                    !$\
     &                 nb_regions,regions,node+1,                                                         !$\
     &                 size,initial,buffer,array)                                                         !$\
#endif                                                                                                    !$\
               endif                                                                                      !$\
               if(types(STEP_Rank).NE.MPI_DATATYPE_NULL) then                                             !$\
                  call MPI_SEND(array(origines(STEP_Rank)),1,                                             !$\
     &                 types(STEP_Rank),node,tag,                                                         !$\
     &                 MPI_COMM_WORLD,iErr)                                                               !$\
               endif                                                                                      !$\
            endif                                                                                         !$\
         endif                                                                                            !$\
      enddo                                                                                               !$\
      end                                                                                                 !$\

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
      include 'STEP.h'                                                                                    !$\
      integer nb_node                                                                                     !$\
      integer*4 types(0:nb_node-1)                                                                        !$\
      integer origines(0:nb_node-1),size                                                                  !$\
      TYPE_FORTRAN array(size)                                                                            !$\
      integer*4 tag                                                                                         !$\
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
      integer*4 node                                                                                      !$\
      integer*4 STATUS(MPI_STATUS_SIZE),iErr                                                                !$\
                                                                                                          !$\
      comm_node(0)=MPI_PROC_NULL                                                                          !$\
      do node=1,nb_node-1                                                                                 !$\
         comm_node(node)=2*nb_node-node                                                                   !$\
      enddo                                                                                               !$\
                                                                                                          !$\
      do step=0,nb_node-1                                                                                 !$\
         alias=mod(STEP_Rank+step,nb_node)                                                                !$\
         if (alias.NE.0)then                                                                              !$\
            node=mod(comm_node(alias)-step,nb_node)                                                       !$\
                                                                                                          !$\
            comm=0                                                                                        !$\
            if (types(STEP_Rank).NE.MPI_DATATYPE_NULL)then                                                !$\
               comm=comm+2                                                                                !$\
            endif                                                                                         !$\
            if (types(node).NE.MPI_DATATYPE_NULL)then                                                     !$\
               comm=comm+1                                                                                !$\
            endif                                                                                         !$\
            select case(comm)                                                                             !$\
            case(3)                                                                                       !$\
               call MPI_SENDRECV(                                                                         !$\
     &              array(origines(STEP_Rank)),1,                                                         !$\
     &              types(STEP_Rank),node,tag,                                                            !$\
     &              BUFFER(origines(node)),1,                                                             !$\
     &              types(node),node,tag,                                                                 !$\
     &              MPI_COMM_WORLD,STATUS,iErr)                                                           !$\
#if INTERLACED                                                                                            !$\
               call MergeRegion/**/TYPE_SUFFIX(dim,                                                       !$\
     &              nb_regions,regions,node+1,                                                            !$\
     &              size,initial,buffer,array)                                                            !$\
#endif                                                                                                    !$\
            case(2)                                                                                       !$\
               call MPI_SEND(                                                                             !$\
     &              array(origines(STEP_Rank)),1,                                                         !$\
     &              types(STEP_Rank),node,tag,                                                            !$\
     &              MPI_COMM_WORLD,iErr)                                                                  !$\
            case(1)                                                                                       !$\
               call MPI_RECV(                                                                             !$\
     &              BUFFER(origines(node)),1,                                                             !$\
     &              types(node),node,tag,                                                                 !$\
     &              MPI_COMM_WORLD,STATUS,iErr)                                                           !$\
#if INTERLACED                                                                                            !$\
               call MergeRegion/**/TYPE_SUFFIX(dim,                                                       !$\
     &              nb_regions,regions,node+1,                                                            !$\
     &              size,initial,buffer,array)                                                            !$\
#endif                                                                                                    !$\
            end select                                                                                    !$\
         endif                                                                                            !$\
      enddo                                                                                               !$\
      end



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
      integer*4 ierr                                                                                      !$\
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
                                                                                                          !$\
c      print *,'regions=',regions                                                                         !$\
c      print *,'initial=',initial                                                                         !$\
c      print *,'in=',in                                                                                   !$\
c      print *,'out=',out                                                                                 !$\
c      print *,'profil=',profil                                                                           !$\
                                                                                                          !$\
      !parcours de la sous_region                                                                         !$\
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
                                                                                                          !$\
         !comparaison case à case pour la sous-region                                                     !$\
         i=indice(dim,coords,regions(L,1,B_))                                                             !$\
!         print *,'i',i,' coord :',coords,' comp : ',initial(i),                                          !$\
!     &        in(i),out(i)                                                                               !$\
         if((in(i).NE.initial(i)).AND.(in(i).NE.out(i)) ) then                                            !$\
            if(out(i).NE.initial(i)) then                                                                 !$\
               print *,'\n** Concurrent Access **\n'                                                      !$\
               call MPI_ABORT(MPI_COMM_WORLD,2,ierr)                                                      !$\
            else                                                                                          !$\
               out(i)=in(i)                                                                               !$\
            endif                                                                                         !$\
         endif                                                                                            !$\
 30   continue                                                                                            !$\
      end

#if INTERLACED
#include "COM_SUBROUTINES_INTERLACED.f"
#else
#include "COM_SUBROUTINES.f"
#endif
