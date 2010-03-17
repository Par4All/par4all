      program aa01

      real x(10), y(10), z(10), u(10,10)
      integer ix(10)

      do i = 1, 10
         x(i) = 0.
      enddo

 100  do i = 1, 10
         x(i) = 0.
      enddo

      do i = 1, 10
C     Reset array X
         x(i) = 0.
      enddo

      do j = 1, 10
         x(j) = 0.
      enddo

      do i = 1, 9
         x(i+1) = 0.
      enddo

      do i = 2, 10
         x(i-1) = 0.
      enddo

      do i = 2, 10
         x(-1+i) = 0.
      enddo

      do i = 1, 10
         x(11-i) = 0.
      enddo

      do i = 1, 4
         x(2*i+1) = 0.
      enddo

C     Cannot be parallelized by PIPS. Hence, the array assignment cannot be prettyprinted
      do i = 1, 3
         x(i**2+1) = 0.
      enddo

      do i = 1, 10
         x(i) = y(i)
      enddo

      do i = 1, 10
C     Detect a scalar in the right hand side
         x(i) = y(k)
      enddo

      do i = 1, 10
C     Use an array contructor with an implicit DO for a non-affine expression
         x(i) = y(i/2)
      enddo

      do i = 1, 10
         x(i) = y(i) + z(i)
      enddo

      do i = 1, 10
         u(i,i) = 0.
      enddo

      do i = 1, 10
         u(i,i**2) = 0.
      enddo

      do i = 1, 10
         u(i,ix(i)) = 0.
      enddo

      do i = 1, 10
         call inc(x(i))
      enddo

      end

      subroutine inc(y)
      y = y + 1.
      end
