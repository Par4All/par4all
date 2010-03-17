      program nga06

C     The transformer returned by the inner loop does not include the
C     exit condition because it is used to compute the inner loop body
C     precondition.

C     This inner loop transformer should be updated after storage. This
C     should be done in statement_to_transformer().

      i = 0
c      if(nsp.lt.1) stop
c      if(ntypes.lt.1) stop

      do n = 1, ntypes
         do j = 1, nsp
            i = i + 1
         enddo
         i = i - nsp
      enddo

      print *, i

      end
