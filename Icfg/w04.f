      program w03
      do while (.true.)
         call hello
      enddo
      end

      subroutine hello
      do i=1, 10
         call bye
      enddo
      end

      subroutine bye
      print *, 'bye'
      end

