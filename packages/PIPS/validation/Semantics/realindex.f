      program realindex

C     Check that index with no semantics values are ignored

      real t(10)

      do x = 1, n
         t(x) = 0.
      enddo

      if(n.ge.1) then
         do x = 1, n
            t(x) = 0.
         enddo
      else
         do x = 1, n
            t(x) = 0.
         enddo
      endif

      end
