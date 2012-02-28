      subroutine xdr_send(msg, len)
      integer msg(len)
      integer buffer_size
      parameter (buffer_size=1024)
      integer buffer(buffer_size)

      call reset(buffer)

      do i = 1, len, buffer_size
         mlen = max(i+buffer_size-1, len)
         do j = i, mlen
            buffer(j-i+1) = msg(j)
         enddo
         print *, (buffer(k), k = 1, mlen-i+1)
      enddo

      end

      subroutine reset(buffer)
      integer buffer_size
      parameter (buffer_size=1024)
      integer buffer(buffer_size)

      do i = 1, buffer_size
         buffer(i) = 0
      enddo

      end

