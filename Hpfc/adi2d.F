!! (c) Fabien COELHO, May 1997
!!
!! 2D ADI kernel inspired by JPDC 38:2 p 161.
!! plus manual invariant code motion and common sub-expressions,
!! plus HPF and FC directives, loop distribution, full updates (__HPFC__)
!!
!! CPP parameters that can be redefined, and their defaults: 
!!  - SIZE [1024]: number of rows and columns
!!  - NITER  [10]: number of iterations
!!  - NPES    [8]: number of processing elements
!
! $Id$
!
! note about (sequential) performances: 
!  - many divisions that are really expensive... (6 cycles?)
!  - the __HPFC__ versions performs better than the initial version.
!
! in seconds        __HPFC__  initial
! Ultra               32.5      40.8
! Power2 66           13.8      16.8
!   reported Mflop/s: 13.9      11.1   [typical fdiv: 10-15 Mflop/s]
!

#if (!defined(SIZE))
#define SIZE 1024
#endif

#if (!defined(NPES))
#define NPES 8
#endif

#if (!defined(NITER))
#define NITER 10
#endif

      program ADI2D

      integer n
      parameter (n=SIZE)

      double precision U(n,n), Uh(n,n), B(n,n)
      double precision alpha, ap2, am2, inv, s

      integer i,j,k

!hpf$ processors P(NPES)
!hpf$ template T(n)
!hpf$ align (i,*) with T(i):: U, Uh, B
!hpf$ distribute T(block) onto P

!hpf$ dynamic U, Uh, B

      print *, '2D ADI running'

! init U
!
! with this mapping, it results in useless communication when
! initializing Uh from U as from the outside loop...
! some code replication and guard could avoid this one?

!hpf$ independent ! gonna fail
      do j=1, n
!fcd$    local
         U(1,j) = 30.0D0
!fcd$    end local
!hpf$    independent
         do i=2, n-1
            U(i,j) = 0.0D0
         enddo
!fcd$    local
         U(n,j) = 30.0D0
!fcd$    end local
      enddo

#if (!defined(__HPFC__))

! init Uh 
!
! as U is copied into Uh and Uh into U, this initialization
! only insures that the borders will have the expected value.
! for HPF, and in order to avoid communication, the arrays
! are fully copied one into the other so as to kill all values, 
! hence avoiding redistribution for the border values that would
! have been live otherwise.

!hpf$ independent
      do j=1, n
!hpf$    independent
         do i=1, n
            Uh(i,j) = U(i,j)
         enddo
      enddo

#endif

      alpha = 8.0D0/DBLE(n)
      ap2 = alpha + 2.0D0
      am2 = alpha - 2.0D0

!fcd$ time

! ITERATIVE LOOP on k:

      do k=1, NITER

         print *, 'k = ', k

#if (!defined(__HPFC__))

! Initial B and U -> Uh

!hpf$    independent
         do j=2, n-1
!hpf$       independent
            do i=2, n-1
               B(i,j) = ap2
               Uh(i,j) = am2 * U(i,j) + U(i,j+1) + U(i,j-1)
            enddo
         enddo

!hpf$    independent
         do j=2, n-1
            Uh(2,j) = Uh(2,j) + U(1,j)
            Uh(n-1,j) = Uh(n-1,j) + U(n,j)
         enddo

#else

! U -> Uh

!hpf$    independent
         do i=1, n
            Uh(i,1) = U(i,1)
         enddo

!hpf$   independent ! this one fails...
         do j=2, n-1
!fcd$       local
            Uh(1,j) = U(1, j)
!fcd$       end local
!hpf$       independent
            do i=2, n-1
               Uh(i,j) = am2 * U(i,j) + U(i,j+1) + U(i,j-1)
            enddo
!fcd$       local
            Uh(n,j) = U(n,j)
!fcd$       end local
         enddo

!hpf$    independent
         do i=1, n
            Uh(i,n) = U(i,n)
         enddo

!hpf$    independent
         do j=2, n-1
            Uh(2,j) = Uh(2,j) + U(1,j)
         enddo

!hpf$    independent
         do j=2, n-1
            Uh(n-1,j) = Uh(n-1,j) + U(n,j)
         enddo

!fcd$    kill U, B ! only Uh is live
!hpf$    realign (*,i) with T(i):: U, Uh, B

! Initialize B

!hpf$ independent
         do j=2, n-1
!hpf$       independent
            do i=2, n-1
               B(i,j) = ap2
            enddo
         enddo

#endif

! forward sweep along columns

!hpf$    independent, new(inv)
         do j=2, n-1
            do i=3, n-1
               inv = 1.0D0 / B(i-1,j)
               B(i,j) = B(i,j) - inv
               Uh(i,j) = Uh(i,j) + Uh(i-1,j) * inv
            enddo
         enddo

!hpf$    independent
         do j=2, n-1
            Uh(n-1,j) = Uh(n-1, j) / B(n-1, j)
         enddo

! backward sweep along columns

!hpf$    independent
         do j=2, n-1
            do i=n-2, 2, -1
               Uh(i,j) = (Uh(i,j) + Uh(i+1,j)) / B(i,j)
            enddo
         enddo

#if (!defined(__HPFC__))

! Initialize B and Uh -> U

!hpf$    independent
         do j=2, n-1
!hpf$       independent
            do i=2, n-1
               B(i,j) = ap2
               U(i,j) = am2 * Uh(i,j) + Uh(i+1,j) + Uh(i-1,j)
            enddo
         enddo

!hpf$    independent
         do i=2, n-1
            U(i,2) = U(i,2) + Uh(i,1)
            U(i,n-1) = U(i,n-1) + Uh(i,n)
         enddo

#else

! Uh -> U

!hpf$    independent
         do i=1, n
            U(i,1) = Uh(i,1)
         enddo

!hpf$    independent
         do j=2, n-1
            U(1,j) = Uh(1,j)
!hpf$       independent ! this one is useless...
            do i=2, n-1
               U(i,j) = am2 * Uh(i,j) + Uh(i+1,j) + Uh(i-1,j)
            enddo
            U(n,j) = Uh(n,j)
         enddo

!hpf$    independent
         do i=1, n
            U(i,n) = Uh(i,n)
         enddo

!hpf$    independent
         do i=2, n-1
            U(i,2) = U(i,2) + Uh(i,1)
         enddo

!hpf$    independent
         do i=2, n-1
            U(i,n-1) = U(i,n-1) + Uh(i,n)
         enddo

!fcd$    kill Uh, B ! only U is live
!hpf$    realign (i,*) with T(i):: U, Uh, B

! Initialize B

!hpf$    independent
         do j=2, n-1
!hpf$       independent
            do i=2, n-1
               B(i,j) = ap2
            enddo
         enddo

#endif

! forward sweep along rows 

         do j=3, n-1
!hpf$       independent, new(inv)
            do i=2, n-1
               inv = 1.0D0 / B(i,j-1)
               B(i,j) = B(i,j) - inv
               U(i,j) = U(i,j) + U(i,j-1) * inv
            enddo
         enddo

!hpf$    independent
         do i=2, n-1
            U(i,n-1) = U(i,n-1) / B(i,n-1)
         enddo

! backward sweep along rows

         do j=n-2, 2, -1
!hpf$       independent
            do i=2, n-1
               U(i,j) = (U(i,j) + U(i,j+1)) / B(i,j)
            enddo
         enddo

      enddo

!fcd$ end time('full computation')

      print *, '2D ADI done'

!
! some output to check the result...
!

!hpf$ independent, reduction(s)
      do j=1, n
!hpf$    independent
         do i=1, n
            s = s + U(i,j)
         enddo
      enddo
      
      print *, 'U(:,:) = ', U(10,10), U(11,10), U(10,11)
      print *, 'SUM(U) = ', s

      end
