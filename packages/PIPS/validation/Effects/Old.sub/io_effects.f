      program io_effects

      read *, n
      call output(n)

      end

      subroutine output(i)

      write(i,*) 'Hello!'

      end
