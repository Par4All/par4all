      program w10

C     Side effects in while loop condition

      integer ms

      ms = 0

      do while(inc(ms).le.2)
         print *, ms
      enddo

      print *, ms

      end

      integer function inc(i)
      i = i + 1
      inc = i
      end
