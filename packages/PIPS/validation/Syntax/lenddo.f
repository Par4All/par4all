      program lenddo

      real x(100), y(100)

      do i = 1, 100
         if(y(j).lt.0.) go to 1
         x(i) = 0.
 1    enddo

      do 2 i = 1, 99
         x(i) = 1.
 2    enddo

      do i = 1, 98
         x(i) = 2.
      enddo

c      do i = 1, 100
c         x(i) = 3.
c 4    continue

      do 5 j = 1, 99
         do 5 i = 1, 99
            x(i) = 1.+real(j)
 5    enddo

      end
