      program w17

C     Check refined transformer for while loop with anded condition

      integer i

      i = 0

      if(.TRUE.) then
         do while(i.gt.-5.and.i.lt.5)
            if(y.gt.0.) then
               i = i + 1
            else
               i = i - 1
            endif
         enddo
      endif

      print *, i

      end
