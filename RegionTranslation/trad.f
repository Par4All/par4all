c     interprocedural propagation of array regions
      program TRADM
      integer M,N,L,P

      integer A(100,100), B(100,100), D(100,100,100)

      read *, M, N, L, P
      if ((N.gt.1).and.(L.gt.3).and.(M.gt.L)) then 
         read *, A
         read *, B
         read *, D
         call TRAD(A,B,D,M,N,L,P)
      endif

      end


      subroutine TRAD(A,B,D,M,N,L,P)      
      integer M,N,L,P
      integer A(N,M), B(3:N, L:M), C(5,10), D(N,M,L), E(5,10,5), F(4,4)

      read *,B

c     array element / scalar variable
      call TRADS(A)
      print *, A
      call TRADS(A(1,1))
      print *, A
      call TRADS(A(3,L))
      print *, A
      call TRADS(A(3, L*L))
      print *, A

c     array / array with the same number of dimensions

cc       reference with no indices
ccc         same declaration
      call TRAD1(A, N, M)
      print *, A
ccc         linear offset in declaration
      call TRAD2(A, N, M, L)
      print *, A
ccc         non linear offset in declaration
      call TRAD3(A, N, M, L)
      print *, A
ccc         same declaration
      call TRAD1S(A, N, M)
      print *, A
ccc         linear offset in declaration
      call TRAD2S(A, N, M, L)
      print *, A
ccc         non linear offset in declaration
      call TRAD3S(A, N, M, L)
      print *, A

cc       reference with indices equal to lower bounds
ccc         same declaration
      call TRAD4(B(3,L), N, M, L)
      print *, B
ccc         linear offset in declaration
      call TRAD5(B(3,L), N, M, L, P)
      print *, B
ccc         non linear offset in declaration
      call TRAD6(B(3,L), N, M, L)
      print *, B
ccc         same declaration
      call TRAD4S(B(3,L), N, M, L)
      print *, B
ccc         linear offset in declaration
      call TRAD5S(B(3,L), N, M, L, P)
      print *, B
ccc         non linear offset in declaration
      call TRAD6S(B(3,L), N, M, L)
      print *, B

cc       reference with constant indices diff. from lower bounds
cc       1- assumed-size array (A)
ccc         same declaration
      call TRAD7(A(1,2), N, M)
      print *, A
ccc         linear offset in declaration
      call TRAD8(A(1,2), N, M, L)
      print *, A
ccc         non linear offset in declaration
      call TRAD9(A(1,2), N, M, L)
      print *, A
ccc         same declaration
      call TRAD7S(A(1,2), N, M)
      print *, A
ccc         linear offset in declaration
      call TRAD8S(A(1,2), N, M, L)
      print *, A
ccc         non linear offset in declaration
      call TRAD9S(A(1,2), N, M, L)
      print *, A

cc       2- constant size array (C)
ccc         same declaration
      call TRAD7(C(1,2), 5, 10)
      print *, C
ccc         linear offset in declaration
      call TRAD8(C(1,2), 5, 10, L)
      print *, C
ccc         non linear offset in declaration
      call TRAD9(C(1,2), 5, 10, L)
      print *, C
ccc         same declaration
      call TRAD7S(C(1,2), 5, 10)
      print *, C
ccc         linear offset in declaration
      call TRAD8S(C(1,2), 5, 10, L)
      print *, C
ccc         non linear offset in declaration
      call TRAD9S(C(1,2), 5, 10, L)
      print *, C
      

c     array / array with less number of dimensions
c     matrix / vector
      call TRAD10(A, N)
      print *, A
      call TRAD11(A, L, N)
      print *, A
      call TRAD12(A, L, N)
      print *, A
      call TRAD13(A, N, M)
      print *, A
      call TRAD14(A, N)
      print *, A
      call TRAD15(A, N)
      print *, A
      call TRAD16(A, L, N)
      print *, A

      call TRAD10(C, 5)
      print *, C
      call TRAD11(C, L, 5)
      print *, C
      call TRAD12(C, L, 5)
      print *, C
      call TRAD17(C, 50)
      print *, C
      call TRAD17(C, 10)
      print *, C
      call TRAD17(C, 7)
      print *, C
      call TRAD17(C, 9)
      print *, C
      call TRAD17(C, L)
      print *, C

c     cube / matrix
      call TRAD18(D, N, M)
      print *, D
      call TRAD19(D, N, M, P)
      print *, D
      call TRAD20(D, N, M)
      print *, D
      call TRAD21(D, N, M, L)      
      print *, D
      call TRAD22(D, N, M)
      print *, D
      call TRAD23(D, N, M, L)      
      print *, D

      call TRAD24(E, 5, 10, 5)
      print *, E
      call TRAD25(E, 5, 10, 5, P)
      print *, E
      call TRAD26(E, 50)
      print *, E
      call TRAD26(E, 10)
      print *, E
      call TRAD27(E, L)
      print *, E
      call TRAD28(E)
      print *, E
      call TRAD29(E)
      print *, E
      call TRAD30(E)
      print *, E
      call TRAD31(E, L)
      print *, E


C     linear subscript values, but non exact propagation
      read *, F
      call TRAD32(F)
      print *, F
      end


C     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
C     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
C     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
C     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

      subroutine TRADS(S)
      integer S
      S = 0
      end

     

C     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
C     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

      subroutine TRAD1(AA, N, M)
      integer N, M, AA(N,M)

      do i = 1, N
         do j = 1, M
            AA(i,j) = i + j
         enddo
      enddo
      end
      
      subroutine TRAD2(AA, N, M, L)
      integer N, M, L, AA(5:N+4,L:M+L-1)

      do i = 5, N+4
         do j = L, M+L-1
            AA(i,j) = i + j
         enddo
      enddo
      end
      
      subroutine TRAD3(AA, N, M, L)
      integer N, M, L, AA(5:N+4,L*L:M+L*L-1)

      do i = 5, N+4
         AA(i,M) = i + M
      enddo
      end
      
C     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

      subroutine TRAD1S(AA, N, M)
      integer N, M, AA(N,*)

      do i = 1, N
         do j = 1, M
            AA(i,j) = i + j
         enddo
      enddo
      end
      
      subroutine TRAD2S(AA, N, M, L)
      integer N, M, L, AA(5:N+4,L:*)

      do i = 5, N+4
         do j = L, M+L-1
            AA(i,j) = i + j
         enddo
      enddo
      end
      
      subroutine TRAD3S(AA, N, M, L)
      integer N, M, L, AA(5:N+4,L*L:*)

      do i = 5, N+4
         AA(i,M) = i + M
      enddo
      end
      

C     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
C     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

      subroutine TRAD4(AA, N, M, L)
      integer N, M, L, AA(3:N,L:M)

      do i = 3, N
         do j = L, M
            AA(i,j) = i + j
         enddo
      enddo
      end
      
      subroutine TRAD5(AA, N, M, L, P)
      integer N, M, L, P, AA(5:N+2,L+P:M+P)

      do i = 5, N+2
         do j = L+P, M+P
            AA(i,j) = i + j
         enddo
      enddo
      end
      
      subroutine TRAD6(AA, N, M, L)
      integer N, M, L, AA(5:N+2,L+L*L:M+L*L)

      do i = 5, N+2
         AA(i,M) = i + M
      enddo
      end


C     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

      subroutine TRAD4S(AA, N, M, L)
      integer N, M, L, AA(3:N,L:*)

      do i = 3, N
         do j = L, M
            AA(i,j) = i + j
         enddo
      enddo
      end
      
      subroutine TRAD5S(AA, N, M, L, P)
      integer N, M, L, P, AA(5:N+2,L+P:*)

      do i = 5, N+2
         do j = L+P, M+P
            AA(i,j) = i + j
         enddo
      enddo
      end
      
      subroutine TRAD6S(AA, N, M, L)
      integer N, M, L, AA(5:N+2,L+L*L:*)

      do i = 5, N+2
         AA(i,M) = i + M
      enddo
      end


C     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
C     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

      subroutine TRAD7(AA, N, M)
      integer N, M, AA(N,M-1)

      do i = 1, N
         do j = 1, M-1
            AA(i,j) = i + j 
         enddo
      enddo
      end
      
      subroutine TRAD8(AA, N, M, L)
      integer N, M, L, AA(3:N+2,L:M+L-2)

      do i = 3, N+2
         do j = L, M+L-2
            AA(i,j) = i + j
         enddo
      enddo
      end
      
      subroutine TRAD9(AA, N, M, L)
      integer N, M, L, AA(3:N+2,L+L*L:M+L+L*L-2)

      do i = 3, N+2
         AA(i,M) = i + M 
      enddo
      end


C     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

      subroutine TRAD7S(AA, N, M)
      integer N, M, AA(N,*)

      do i = 1, N
         do j = 1, M-1
            AA(i,j) = i + j
         enddo
      enddo
      end
      
      subroutine TRAD8S(AA, N, M, L)
      integer N, M, L, AA(3:N+2,L:*)

      do i = 3, N+2
         do j = L, M+L-2
            AA(i,j) = i + j
         enddo
      enddo
      end
      
      subroutine TRAD9S(AA, N, M, L)
      integer N, M, L, AA(3:N+2,L+L*L:*)

      do i = 3, N+2
         AA(i,M) = i + M 
      enddo
      end

C     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
C     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

      subroutine TRAD10(V, N)
      integer V(*), N

      do i = 1, 2 * N
         V(i) = i 
      enddo
      end

      subroutine TRAD11(V, L, N)
      integer L, V(L:*), N

      do i = L, 2 * N + L - 1
         V(i) = i 
      enddo
      end

      subroutine TRAD12(V, L, N)
      integer L, V(L*L:*), N

      V(N + L) = N + L 
      end

      subroutine TRAD13(V, N, M)
      integer N, M, V(N*M)

      do i = 1, 2 * N
         V(i) = i 
      enddo
      end

      subroutine TRAD14(V, N)
      integer N, V(2*N)

      do i = 1, 2 * N
         V(i) = i 
      enddo
      end

         
      subroutine TRAD15(V, N)
      integer N, V(2*N+10)

      do i = 1, 2 * N + 10
         V(i) = i 
      enddo
      end

      subroutine TRAD16(V, L, N)
      integer N, L, V(L*N)

      do i = 1, 2 * N 
         V(i) = i 
      enddo
      end
      

      subroutine TRAD17(V, N)
      integer N, V(N)
      
      do i = 1, N
         V(i) = i
      enddo
      end

C     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
C     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

      subroutine TRAD18(DD, N, M)
      integer N, M, DD(N, *)
      
      do i = 1, N
         do j = 1, 2 * M
            DD(i,j) = i + j
         enddo
      enddo
      end

      subroutine TRAD19(DD, N, M, P)
      integer N, M, P, DD(P:N+P-1, *)
      
      do i = P, N+P-1
         do j = 1, 2 * M
            DD(i,j) = i + j
         enddo
      enddo
      end

      subroutine TRAD20(DD, N, M)
      integer N, M, DD(N, 2*M)
      
      do i = 1, N
         do j = 1, 2 * M
            DD(i,j) = i + j
         enddo
      enddo
      end

      subroutine TRAD21(DD, N, M, L)
      integer N, M, L, DD(N, L*M)
      
      do i = 1, N
         do j = 1, 2 * M
            DD(i,j) = i + j
         enddo
      enddo
      end

      subroutine TRAD22(DD, N, M)
      integer N, M, DD(N*M, *)
      
      do i = 1, 2*N
         do j = 1, 5
            DD(i,j) = i + j
         enddo
      enddo
      end

      subroutine TRAD23(DD, N, M, L)
      integer N, M, DD(N*M, L)
      
      do i = 1, 2*N
         do j = 1, L
            DD(i,j) = i + j
         enddo
      enddo
      end


      subroutine TRAD24(EE, N, M, L)
      integer N, M, L, EE(5, *)
      
      do i = 1, N
         do j = 1, M
            do k = 1, L
               EE(i,j+k) = i + j
            enddo
         enddo
      enddo
      end      

      subroutine TRAD25(EE, N, M, L, P)
      integer N, M, L, P, EE(P:5+P-1, *)
      
      do i = P, N+P-1
         do j = 1, M
            do k = 1, L
               EE(i,j+k) = i + j
            enddo
         enddo
      enddo
      end      

      subroutine TRAD26(EE, N)
      integer N, EE(5, N)
      
      do i = 1, 5
         do j = 1, N
            EE(i,j) = i + j
         enddo
      enddo
      end      

      subroutine TRAD27(EE, L)
      integer L, EE(5, L * 10)
      
      do i = 1, 5
         do j = 1, L*10
            EE(i,j) = i + j
         enddo
      enddo
      end      

      subroutine TRAD28(EE)
      integer EE(50, *)
      
      do i = 1, 50
         do j = 1, 5
            EE(i,j) = i + j
         enddo
      enddo
      end      

       
      subroutine TRAD29(EE)
      integer EE(50, 5)
      
      do i = 1, 50
         do j = 1, 5
            EE(i,j) = i + j
         enddo
      enddo
      end      

      subroutine TRAD30(EE)
      integer EE(50, 2)
      
      do i = 1, 50
         do j = 1, 2
            EE(i,j) = i + j
         enddo
      enddo
      end      

      subroutine TRAD31(EE, L)
      integer L, EE(50, L)
      
      do i = 1, 50
         do j = 1, L
            EE(i,j) = i + j
         enddo
      enddo
      end      

 
      subroutine TRAD32(FF)
      integer FF(2,8)
      
      do j = 4,5
         FF(1,j) = j
      enddo
      do j = 2,7
         FF(2,j) = j + 1
      enddo
      end      

 
