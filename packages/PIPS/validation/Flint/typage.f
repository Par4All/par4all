C     This program is written for testing module TYPING
C
C
      PROGRAM TYPAGE
      LOGICAL L
      INTEGER I
      REAL*4 R
      REAL*8 D
      COMPLEX*8 C
      CHARACTER CH

C     Test for Assignment 
C     ===================
      I = L                     !ERROR
      I = I
      I = R
      I = D
      I = C
      I = CH                    !ERROR

      R = L                     !ERROR
      R = I
      R = R
      R = D
      R = C
      R = CH                    !ERROR

      D = L                     !ERROR
      D = I
      D = R
      D = D
      D = C
      D = CH                    !ERROR
      
      C = L                     !ERROR
      C = I
      C = R
      C = D
      C = C
      C = CH                    !ERROR

      CH = L                    !ERROR
      CH = I                    !ERROR
      CH = R                    !ERROR
      CH = D                    !ERROR
      CH = C                    !ERROR
      CH = CH

      L = L
      L = I                     !ERROR
      L = R                     !ERROR
      L = D                     !ERROR
      L = C                     !ERROR
      L = CH                    !ERROR

C     Test for Arithmetic operators: +, -, *, /
C     =========================================
      I = +5 - I
      I = -I + R
      I = +R / D
      I = C * R

      I = I + R
      I = R + D
      I = D + C                 !ERR - PROBIHITED
      I = R + C
      I = C + R
      
      I = I + R + D + C         !ERR - PROBIHITED
      I = C + D + R + I         !ERR - PROBIHITED
      
C     Test for power operator: **
C     Somes cases are prohibited by FORTRAN itself (just for checking errors)
C     =======================================================================
      I = I**I
      I = I**R
      I = I**D
      I = I**C
      
      I = R**I
      I = R**R
      I = R**D
      I = R**C
      
      I = C**I
      I = C**R
      I = C**D                  !ERR - PROHIBITED
      I = C**C
      
      I = D**I
      I = D**R
      I = D**D
      I = D**C                  !ERR - PROHIBITED
      
C     Relational operators: <, <=, =, !=, >, >=
C     For this, we only have to test the casting of 2 arguments
C     Value of return is always BOOL
C     ==============================
      L = I .LT. R
      L = I .LE. R
      L = I .EQ. R
      L = I .NE. R
      L = I .GT. R
      L = I .GE. R
      
      L = R .LT. I
      L = R .LE. I
      L = R .EQ. I
      L = R .NE. I
      L = R .GT. I
      L = R .GE. I
      
      I = R .GE. I              !ERROR
      L = R .GE. L              !ERROR
      L = L .GE. I              !ERROR
      L = D .GE. C              !ERROR
      
      
C     Logical operators: NOT, AND, OR, EQV, NEQV
C     Value of return is always BOOL
C     ==============================
      L = .NOT. L
      L = L .AND. L
      L = L .OR. L
      L = L .EQV. L
      L = L .NEQV. L
      
      L = I .OR. L              !ERROR
      L = R .AND. L             !ERROR
      L = R .AND. L .OR. 12.4E5 !ERROR
      
C     User-defined functions
C     ======================
      R = USER_FUNC(I, R)
      R = USER_FUNC(R, R)       !ERROR
      R = USER_FUNC(I, I)       !ERROR
      R = USER_FUNC(R, I)       !2 ERRORS
      R = USER_FUNC(R, USER_FUNC(R, R)) !2 ERRORS
      R = USER_FUNC(I)          !ERROR
      R = USER_FUNC(I, R, I, R) !ERROR

      CALL USER_SUB(I, R, C)
      CALL USER_SUB(I, I, C)
      CALL USER_SUB(R, C, I)
      CALL USER_SUB(I, R)
      CALL USER_SUB(I, R, C, L)
      
C     Do-Loop
C     =======
      DO I = 1.5, 5, 1.5
         CONTINUE
      ENDDO
      DO R = 1, 5, 1.5
         CONTINUE
      ENDDO
      DO D = 1, 5, 1.5
         CONTINUE
      ENDDO

      DO D = 1, 5, C            !ERROR
         CONTINUE
      ENDDO
      DO C = 1, 5, 1.5          !ERROR
         CONTINUE
      ENDDO
      DO CH = 1, 5, 1.5         !ERROR
         CONTINUE
      ENDDO

      DO L = .FALSE., .TRUE.    !ERROR
         CONTINUE
      ENDDO

C     While Loop
C     ==========
      DO WHILE (.TRUE.)
         CONTINUE
      ENDDO

      DO WHILE (2+I)            !ERROR
         CONTINUE
      ENDDO

      DO WHILE (USER_FUNC(1,.TRUE.)) !ERROR
         CONTINUE
      ENDDO

C     Condition Test
C     ==============
      IF ( I .LT. R) THEN
         STOP
      ELSE IF (R .GT. D) THEN
         STOP
      ELSE
         STOP
      ENDIF

      IF ( I + R) THEN          !ERROR
         STOP
      ELSE IF (CH) THEN         !EEROR
         STOP
      ELSE
         STOP
      ENDIF

      END

C     User-defined function
C     =====================
      REAL FUNCTION USER_FUNC(A, B)
      INTEGER A
      REAL B
      USER_FUNC = A + B
      END

      SUBROUTINE USER_SUB(I, R, C)
      INTEGER I
      REAL R
      COMPLEX C
      PRINT *, I, R, C
      END
