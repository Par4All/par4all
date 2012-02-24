      program funcside4

      external foo4
      integer foo4

      i4 = 10

      i4 = foo4(i4)

      print *, i4

      end

      integer function foo4(j)

      if(mod(j,2).eq.0) then
         foo4 = 2
      else
         foo4 = 0
      endif

      end
