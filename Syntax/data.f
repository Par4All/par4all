      program 
      common /foo/ i, j
      print *, i, j
      end

      block data bla
      common /foo/ i, j
      data i / 4 /
      end

      block data
      common /foo/ i, j
      data j / 5 /
      end
