      program initial4

C     See what happens with conflicting initial values

      common /foo/i

      print *, i

      end

      blockdata bar1
      common /foo/i
      data i /1/
      end

      blockdata bar2
      common /foo/i
      data i /2/
      end

