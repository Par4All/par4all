      subroutine non_linear15

C     Check REFINE_TRANSFORMERS

      i = 2
      call foo(i, j, k)
      print *, j

      end

      subroutine foo(i, j, k)
      j = k*i
      if(k.eq.2) then
         read *, i
         l = j*m
         print *, l
      endif

      end
