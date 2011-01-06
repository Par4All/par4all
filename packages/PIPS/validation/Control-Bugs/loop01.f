! Debug the destructuration of a Fortran DO loop by the new controlizer

      program looop01

      do i = 1, n
         if(mod(i,2).eq.0) go to 100
      enddo

      i = i + 10

 100  continue

      end
