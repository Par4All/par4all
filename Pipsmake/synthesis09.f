      program synthesis09

C     Check for overloaded operator

      integer lload
      integer nbr
      parameter (nbr = 2)
      parameter  (lload  = 3*nbr )
      parameter (pi=3.14159)
      real lhiden
      parameter (lhiden = nbr*pi)

      call  hidenh ( max(lhiden, lload), max(lload, n), 9999)

      end
