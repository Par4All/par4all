!
! PRECISION BUG IN MUST_REGIONS
!
! first test case is tagged MAY
! bug second loop case is tagged EXACT, 
! although it should be the same.
!
      program bugprec
      real a(10)
      integer i, j, n, p

      do i=1, 10
         a(i) = i
      enddo

      read *, n, p, i, j

!ABC if (1.le.p.and.(n.lt.0.or.n.gt.7)) stop
      if (1.le.p) then
         do i=n, n+3
            a(i) = 10
         enddo
      endif

!ABC if (1.le.p.and.(n.lt.0.or.n.gt.7)) stop
      do j=1, p
         do i=n, n+3
            a(i) = 10
         enddo
      enddo

      end
