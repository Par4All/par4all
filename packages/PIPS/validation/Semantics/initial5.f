      program initial5

C     See what happens with aliases: it should not work
C     but it *does* work. Why?

      common /foo/i

      print *, i

      end

      blockdata bar1
      common /foo/j
      data j /1/
      end

