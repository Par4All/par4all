;;;; Reduction patterns database

(require :init)
(require :ri)
(require :reduction)
(require :util)

(use-package '(:newgen :ri :reduction))

;;; Useful values

(defparameter value[true] (make-value[] gen-true))
(defparameter entity[true] 
    (make-entity[] gen-true "TOP_LEVEL:_TRUE_"))
(defparameter entity[identity]
    (make-entity[] 0 "TOP_LEVEL:.IDENTITY.")
    "Used as the identity operator for reduction detection")
(defparameter entity[fun-xy.x]
    (make-entity[] 0 "TOP_LEVEL:.FUN XY.X.")
    "Used as the operator for auto assignment reduction")
(defparameter entity[fun-xy.y]
    (make-entity[] 0 "TOP_LEVEL:.FUN XY.Y.")
    "Used as the operator for invariant expression detection")
(defparameter entity[gensym]
    (make-entity[] 0 "TOP_LEVEL:.GENSYM.")
    "Used whenever some entity is modified in a non-expressible-way")

(defparameter expression[true] 
    (make-expression[] gen-true "TOP_LEVEL:_TRUE_"))
(defparameter expression[gensym] 
    (make-expression[] 0 "TOP_LEVEL:.GENSYM."))

(defparameter sexpression[true]
    (make-sexpression :gexpressions
		       `(,(make-gexpression :guard expression[true] 
					    :expression expression[true]))))

;;; Useful functions

(defparameter entity[constant]
    (make-entity[] -1 "TOP_LEVEL:CONSTANT")
    "Loop constant")
(defparameter entity[sum] 
    (make-entity[] -1 "TOP_LEVEL:SUM")
    "Arithmetic reduction")
(defparameter entity[product] 
    (make-entity[] -1 "TOP_LEVEL:PRODUCT")
    "Geometric reduction")
(defparameter entity[array-sum] 
    (make-entity[] -1 "TOP_LEVEL:ARRAYSUM")
    "Array sum")
(defparameter entity[array-max] 
    (make-entity[] -1 "TOP_LEVEL:ARRAYMAX")
    "Array max")
(defparameter entity[inner] 
    (make-entity[] -1 "TOP_LEVEL:INNER")
    "Array inner product")
(defparameter entity[minpos] 
    (make-entity[] -1 "TOP_LEVEL:MINPOS")
    "Array minimum index")

;;; Unification variables

(defun make-unification-variable (counter)
    `(*variable* ,counter))

(defun unification-variable-p (var) 
  (and (consp var) 
       (eq (car var) '*variable*)))

;;; Pattern database

(defstruct template
    pattern				;description
    generator				;code generator
    value)				;symbolic value after the loop

;; X=K
;;
(defparameter constant-reduction
    (make-template
     :pattern (make-pattern :variable (make-unification-variable 1)
			    :condition expression[true]
			    :parameter (make-expression-call
					(make-unification-variable 2) '())
			    :operator entity[fun-xy.y]
			    :indices `())
     :generator #'(lambda (var cond param op context loop)
		      (let ((range (loop-range loop)))
			  (make-statement-call 
			   entity[constant]
			   `(,(make-expression-reference var '())
			     ,param
			     ,(range-lower range)
			     ,(range-upper range)
			     ,(range-increment range)))))
     :value (make-expression-call (make-unification-variable 2) '())))

;; X=X
;;
;; We use EXPRESSION[GENSYM], but any thing would work fine.
;;
;; WARNING: This reduction is not detected since it would be detected
;; for all the detected inner-loop reductions (since the
;; STATEMENT-WRITTEN-ENTITIES are not updates correctly).
;;
(defparameter auto-reduction
    (make-template
     :pattern (make-pattern :variable (make-unification-variable 1)
			    :condition expression[true]
			    :parameter 
			    (make-expression[] 0 "TOP-LEVEL:0")
			    :operator entity[fun-xy.x]
			    :indices `())
     :generator #'(lambda (sc cond param op context loop)
		      (make-statement-block '()))
     :value (make-expression-reference (make-unification-variable 1) '())))

;; X=X+K
;;
(defun op-code (var cond param op context loop)
    (let ((range (loop-range loop)))
	(make-statement-call 
	 (cond ((entity-equal-p op (function-entity[] "+"))
		entity[sum])
	       ((entity-equal-p op (function-entity[] "*"))
		entity[product])
	       (t (error "~%Unknown entity ~A" (entity-local-name op))))
	 `(,(make-expression-reference var '())
	   ,param
	   ,(range-lower range)
	   ,(range-upper range)
	   ,(range-increment range)))))

(defparameter sum-reduction
    (make-template
     :pattern (make-pattern :variable (make-unification-variable 1)
			    :condition expression[true]
			    :parameter (make-expression-call
					(make-unification-variable 2) '())
			    :operator (function-entity[] "+")
			    :indices `())
     :generator #'op-code
     :value (make-expression-call (make-unification-variable 2) '())))

;; X=X*K
;;
(defparameter product-reduction
    (make-template
     :pattern (make-pattern :variable (make-unification-variable 1)
			    :condition expression[true]
			    :parameter (make-expression-call
					(make-unification-variable 2) '())
			    :operator (function-entity[] "*")
			    :indices `())
     :generator #'op-code
     :value (make-expression-call (make-unification-variable 2) '())))

;; X=X+T[I]
;;
(defun array-sum-code (var cond param op context loop)
    (let ((range (loop-range loop)))
	(make-statement-call 
	 entity[array-sum]
	 `(,(make-expression-reference var '())
	   ,(make-expression-reference 
	     (expression-reference-variable param)
	     '())
	   ,(range-lower range)
	   ,(range-upper range)
	   ,(range-increment range)))))

(defparameter array-sum-reduction
    (make-template
     :pattern 
     (make-pattern :variable (make-unification-variable 1)
		   :condition expression[true]
		   :parameter (make-expression-reference
			       (make-unification-variable 2)
			       `(,(make-expression-reference
				   (make-unification-variable 3) '())))
		   :operator (function-entity[] "+")
		   :indices `(,(make-unification-variable 3)))
     :generator #'array-sum-code
     :value (make-expression-call (make-unification-variable 2) '())))

;; X=X+T[I]*U[I]
;;
(defun inner-product-code (var cond param op context loop)
    (let ((range (loop-range loop))
	  (arrays (expression-call-arguments param)))
	(make-statement-call 
	 entity[inner]
	 `(,(make-expression-reference var '())
	   ,(first arrays)
	   ,(second arrays)
	   ,(range-lower range)
	   ,(range-upper range)
	   ,(range-increment range)))))

(defparameter inner-product-reduction
    (make-template
     :pattern
     (make-pattern :variable (make-unification-variable 1)
		   :condition expression[true]
		   :parameter
		   (make-expression-call
		    (function-entity[] "*")
		    `(,(make-expression-reference
			(make-unification-variable 2)
			`(,(make-expression-reference
			    (make-unification-variable 3) '())))
		      ,(make-expression-reference
			(make-unification-variable 4)
			`(,(make-expression-reference
			    (make-unification-variable 3) '())))))
		   :operator (function-entity[] "+")
		   :indices `(,(make-unification-variable 3)))
     :generator #'inner-product-code
     :value (make-expression-call
	     (function-entity[] "*")
	     `(,(make-expression-call (make-unification-variable 2) '())
	       ,(make-expression-call (make-unification-variable 4) '())))))

;; IF( T[I].GT.X ) X=T[I]
;;
(defun array-max-code (var cond param op context loop)
    (let ((range (loop-range loop)))
	(make-statement-call 
	 entity[array-max]
	 `(,(make-expression-reference var '())
	   ,(make-expression-reference 
	     (expression-reference-variable param) '())
	   ,(range-lower range)
	   ,(range-upper range)
	   ,(range-increment range)))))

(defparameter array-max-reduction
    (let ((ti (make-expression-reference
	       (make-unification-variable 2)
	       `(,(make-expression-reference
		   (make-unification-variable 3) '())))))
	(make-template
	 :pattern 
	 (make-pattern :variable (make-unification-variable 1)
		       :condition (make-expression-call
				   (function-entity[] ".GT.")
				   `(,ti
				     ,(make-expression-reference
				       (make-unification-variable 1) '())))
		       :parameter ti
		       :operator entity[fun-xy.y]
		       :indices `(,(make-unification-variable 3)))
	 :generator #'array-max-code
	 :value (make-expression-reference
		 (make-unification-variable 2) '()))))

;; IF( T[I].LT.T[M] ) M=I
;;
(defun array-minpos-code (var cond param op context loop)
    (let ((range (loop-range loop)))
	(make-statement-call 
	 entity[minpos]
	 `(,(make-expression-reference var '())
	   ,(make-expression-reference 
	     (expression-reference-variable param) '())
	   ,(range-lower range)
	   ,(range-upper range)
	   ,(range-increment range)))))

(defparameter array-minpos-reduction
    (let* ((m (make-unification-variable 1))
	   (tt (make-unification-variable 2))
	   (i (make-unification-variable 3))
	   (ti (make-expression-reference
		tt`(,(make-expression-reference i '()))))
	   (tm (make-expression-reference
		tt`(,(make-expression-reference m '())))))
	(make-template
	 :pattern 
	 (make-pattern :variable m
		       :condition (make-expression-call
				   (function-entity[] ".LT.") `(,ti ,tm))
		       :parameter (make-expression-reference i '())
		       :operator entity[fun-xy.y]
		       :indices `(,i))
	 :generator #'array-minpos-code
	 :value (make-expression-reference tt'()))))

;;; The global database

(defparameter *reductions*
    `(,constant-reduction
;     ,auto-reduction
      ,sum-reduction
      ,product-reduction
      ,array-sum-reduction
      ,inner-product-reduction
      ,array-max-reduction
      ,array-minpos-reduction
      ))

(defparameter *reduction-entities*
    `(,entity[constant]
      ,entity[sum]
      ,entity[product]
      ,entity[array-sum]
      ,entity[array-max]
      ,entity[inner]
      ,entity[minpos]))

;;;

(provide :patterns)
