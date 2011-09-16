      program PI
      implicit none
      integer i, num_iter
      double precision x, my_pi, contrib, iter
      contrib = 0.0
      num_iter = 10
      iter = 1.0/num_iter

!$OMP PARALLEL DO REDUCTION (+:contrib)
      DO i = 1, num_iter
	 x = (i-0.5)*iter
	 contrib = contrib + 4.0/(1.0+x*x)
	 print *, i,x,contrib
      END DO
!$OMP END PARALLEL DO

      my_pi = iter * contrib

      print *,my_pi
      END
