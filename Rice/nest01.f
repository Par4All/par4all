!     Check nest parallelization: the two loops are parallel, but they
!     must also be interchanged

!     The number of iterations is one of the decision criteria, so we
!     use 10 and 11 to be close to obtain the interchance while
!     different to be able to analyze the results easily

      program nest01

      real a(10,20)

      do i = 1, 10
         do j = 1, 11
            a(i,j) = 0.
         enddo
      enddo

      end
