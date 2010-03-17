      SUBROUTINE SUBSTR02 (TEXTE,I,R,C,ITYPE,LONG)

C     Bug for substring without an upper bound and without a full
C     declaration

      CHARACTER * (*) TEXTE, C
      character*2 formai
      data formai /'i4'/

      jtype = 1

      WRITE (TEXTE(IPOS+1:),FORMAI) JTYPE

      end
