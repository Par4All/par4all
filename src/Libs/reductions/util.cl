;;;;
;;;; Utility functions for RI manipulation
;;;;

(require :init)

;;; To delete when NewGen is fixed

(defconstant normalized-undefined :undefined)
(defconstant int-undefined :undefined)

;;; Machine description (well, sort of :-)

(defparameter bytes-per-int 4)
(defparameter bytes-per-float 4)
(defparameter bytes-per-double 8)
(defparameter bytes-per-logical 4)
(defparameter bytes-per-complex 4)

;;; Standard Functions

(defun make-value[] (x)
    (make-value is-value-constant 
		(typecase x
			  (fixnum (make-constant is-constant-int x))
			  (t (make-constant is-constant-litteral :unit)))))

(defun make-entity[] (x name)
    (let* ((b (typecase x
			(fixnum (make-basic is-basic-int 
					    bytes-per-int))
			(list (make-basic is-basic-logical 
					  bytes-per-logical))))
	   (res (if (and (numberp x)
			 (minusp x))
			(make-type is-type-void :unit)
		    (make-type is-type-variable
			       (make-variable
				:basic b
				:dimensions '())))))
	(make-entity 
	 :name name
	 :type (make-type is-type-functional
			  (make-functional :parameters '()
					   :result  res))
	 :storage (make-storage is-storage-rom)
	 :initial (make-value[] x))))
(defun entity-local-name (x)
    (subseq (entity-name x) (1+ (position #\: (entity-name x)))))

(defun make-expression[] (x name)
    (make-expression 
     :syntax (make-syntax is-syntax-call
			  (make-call :function (make-entity[] x name)
				     :arguments '()))))   

(defun expression-equal-p (exp1 exp2)
    (funcall
     (gen-recurse
      exp1
      ((expression syntax) syntax)
      ((syntax tag) tag)
      ((reference indices)
       #'(lambda (e2)
	     (and (expression-reference-p e2)
		  (entity-equal-p (reference-variable reference)
				  (expression-reference-variable e2))
		  (every #'funcall
			 indices
			 (expression-reference-indices e2)))))
      ((call arguments)
       #'(lambda (e2)
	     (and (expression-call-p e2)
		  (entity-equal-p (call-function call)
				  (expression-call-function e2))
		  (every #'funcall
			 arguments
			 (expression-call-arguments e2))))))
     exp2))

(defun pure-expression-p (exp)
    (gen-recurse
      exp
      ((expression syntax) syntax)
      ((syntax tag) tag)
      ((reference indices)
       (or (endp indices)
	   (every #'identity indices)))
      ((call arguments)
       (and (every #'identity arguments)
	    (let ((initial (entity-initial (call-function call))))
		(or (value-intrinsic-p initial)
		    (value-constant-p initial)))))))

(defun free-vars (exp)
    (gen-recurse
      exp
      ((expression syntax) syntax)
      ((syntax tag) tag)
      ((reference indices)
       `(,(reference-variable reference)
	 ,@(apply #'nconc indices)))
      ((call arguments)
       (apply #'nconc arguments))))

(defun entity-equal-p (e1 e2)
    (string= (entity-name e1) (entity-name e2)))

(defun entity-assign-p (e)
    (string= (entity-name e) "TOP-LEVEL:="))

(defun entity-and-p (e)
    (string= (entity-name e) "TOP-LEVEL:.AND."))

(defun boolean-expression-p (which)
    #'(lambda (e)
	  (let ((syntax (expression-syntax e)))
	      (and (syntax-call-p syntax)
		   (let ((call (syntax-call syntax)))
		       (entity-equal-p (call-function call) which))))))
(defun true-expression-p (e) 
    (funcall (boolean-expression-p entity[true]) e))
	  
(defun scalar-entity-p (e)
    (let ((type (entity-type e)))
	(assert (type-variable-p type))
	(endp (variable-dimensions (type-variable type)))))

;;; Shortcut functions

(defun function-entity[] (name)
    (let ((entity-name (concatenate 'string "TOP-LEVEL:" name) ))
	(or (gen-find-tabulated entity-name entity-domain)
	    (error "~%Unknown entity ~S" entity-name))))

(defun make-expression-call (f args)
    (make-expression
     :syntax (make-syntax is-syntax-call 
			  (make-call :function f :arguments args))
     :normalized normalized-undefined))
(defun expression-call-p (exp)
    (syntax-call-p (expression-syntax exp)))
(defun expression-call-function (e)
    (call-function (syntax-call (expression-syntax e))))
(defun expression-call-arguments (e)
    (call-arguments (syntax-call (expression-syntax e))))

(defun make-expression-reference (e inds)
    (make-expression
     :syntax (make-syntax is-syntax-reference
			  (make-reference :variable e :indices inds))
     :normalized normalized-undefined))
(defun expression-reference-variable (e)
    (reference-variable (syntax-reference (expression-syntax e))))
(defun expression-reference-indices (e)
    (reference-indices (syntax-reference (expression-syntax e))))
(defun expression-reference-p (e)
    (syntax-reference-p (expression-syntax e)))

(defun effect-reference-variable (f)
    (reference-variable (effect-reference f)))

(defun statement-written-entities (statement)
    (mapcar #'effect-reference-variable
	    (remove-if-not #'(lambda (x) (action-write-p x))
			   (statement-cumulated-effects statement)
			   :key #'effect-action)))
(defun statement-read-entities (statement)
    (mapcar #'effect-reference-variable
	    (remove-if-not #'(lambda (x) (action-read-p x))
			   (statement-cumulated-effects statement)
			   :key #'effect-action)))
(defun statement-cumulated-effects (statement)
    (effects-effects (gethash (statement-ordering statement)
			      *effects-mapping*
			      :effect-not-found)))

;;; Control Management

(defun get-blocs (c l)
    (if (member c l :test #'eql)
	    l
	(reduce #'get-blocs
		(control-successors c)
		:initial-value (reduce #'get-blocs
				       (control-predecessors c)
				       :initial-value (cons c l)))))

(defmacro control-map (f c blocs)
    (let ((cm-list-init (gensym))
	  (cm-list (gensym)))
	`(let* ((,cm-list-init ,blocs)
		(,cm-list ,cm-list-init))
	     (when (endp ,cm-list)
		 (setf ,cm-list (nreverse (get-blocs ,c '()))))
	     (mapc ,f ,cm-list)
	     (when (endp ,cm-list-init)
		 (setf ,blocs ,cm-list))
	     (values))))
		    

;;; Statement management

(defun empty-label () (function-entity[] "@"))

(defvar *new-statement-ordering* 1)

(defun new-statement-ordering ()
    (cond ((gethash *new-statement-ordering* *effects-mapping*)
	   (incf *new-statement-ordering*)
	   (new-statement-ordering))
	  (t  *new-statement-ordering*)))

;; Incorrect effects !!!
(defun make-statement-instruction (i)
    (let ((st (make-statement :label (empty-label)
			      :number int-undefined
			      :ordering (new-statement-ordering)
			      :comments ""
			      :instruction i)))
	(setf (gethash (statement-ordering st) *effects-mapping*)
	      (make-effects :effects '()))
	st))

(defun make-instruction-call (f args)
    (make-instruction is-instruction-call
		      (make-call :function f
				 :arguments args)))
(defun make-statement-call (name args)
    (make-statement-instruction (make-instruction-call name args)))
    
(defun make-instruction-loop (loop)
    (make-instruction is-instruction-loop loop))
(defun make-statement-loop (loop)
    (make-statement-instruction (make-instruction-loop loop)))

(defun make-instruction-block (stats)
    (make-instruction is-instruction-block stats))
(defun make-statement-block (stats)
    (make-statement-instruction (make-instruction-block stats)))

;;;

(provide :util)

