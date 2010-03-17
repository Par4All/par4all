      subroutine mask

      call masked_init

      call masked_incr

      print *, 'hello'

      end

      subroutine masked_init
cfirst      common /foo/dummy(4)
csecond      common /foo/i,dummy(3)

      call init

      end

      subroutine init
      common /foo/i,dummy(3)

      i = 0

      end

      subroutine masked_incr
cfirst      common /foo/dummy(4)
csecond      common /foo/i,dummy(3)

      call incr

      end

      subroutine incr
      common /foo/i,dummy(3)

      i = i + 1

      end


      
