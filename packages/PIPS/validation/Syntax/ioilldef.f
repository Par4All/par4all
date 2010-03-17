      program ioilldef

C     Illegal IO control default option should be detected
C     (although PIPS is not supposed to do much to check
C     syntax errors)

      write(*,*,ERR=*) 'Hello'

      end
