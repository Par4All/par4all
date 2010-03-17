! complex constants
      program complexes
      complex i, j
      complex*16 i2
      integer k

      data j / (1.0,0.0) /
      print *, j
      
      read (5,10) k
      print *, k

! COMPLEX

      i = CMPLX(0.0,1.0)
      print *, i

      i = (0.0,1.0)
      print *, i

      i = (0.0E0,-1.0E0)
      print *, i

! DOUBLE COMPLEX

      i2 = DCMPLX(0.0,1.0)
      print *, i2

      i2 = (0.0D0,-1.0D0)
      print *, i2

      i2 = (+0.428854343225d-31,-1.4564565454326D+12)
      print *, i2

 10   format (i5)

      print *, "continuation...", 
     1         (0., 1.)

! expansion => continuations
      print *, 0.00002D12, i, j, k, 2, i+2*j-4, (0.d0,1.e0), (-0.3,1.1)

      end
