;;;; The symbolic evaluation of programs.

(require :init)
(require :ri)
(require :reduction)
(require :util)
(require :simplify)

(use-package '(:newgen :ri :reduction))

(defvar *debug-eval* nil)

;;; Evaluation of statement

(defun eval-statement (statement store)
    (funcall
     (gen-recurse 
      statement
      ((statement instruction) instruction)
      ((instruction tag)
       #'(lambda (store guard statement)
	     (if (instruction-block-p instruction)
		     (do ((stats (instruction-block instruction) (cdr stats))
			  (tag tag (cdr tag))
			  (store store 
				 (funcall (car tag) store guard (car stats))))
			 ((endp tag) store))
		 (funcall tag store guard statement))))
      ((call) (eval-call call))
      ((test true false) (eval-test test true false))
      ((loop body) (eval-loop loop body)))
     store
     sexpression[true]
     statement))

(defun eval-call (call)
    #'(lambda (store guard statement)
	  (let ((f (call-function call)))
	      (cond ((entity-assign-p f)
		     (let ((lhs (first (call-arguments call)))
			   (rhs (second (call-arguments call))))
			 (funcall (eval-assign lhs rhs) store guard)))
		    ((member f *reduction-entities*
			     :test #'entity-equal-p)
		     (let ((red (first (call-arguments call))))
			 (update-store store red expression[gensym])))
		    (t
		     (reduce #'(lambda (store mod)
				   (update-store store
						 mod
						 expression[gensym]))
			     (statement-written-entities statement)
			     :initial-value store))))))

(defun eval-assign (lhs rhs)
    #'(lambda (store guard)
	  (let ((lhs-ref (syntax-reference (expression-syntax lhs))))
	      (if (not (endp (reference-indices lhs-ref)))
		      store
		  (update-store store
				(reference-variable lhs-ref)
				(eval-expression rhs store guard))))))

;;; Test management

(defun test-init-store (domain store guard)
    (reduce 
     #'(lambda (store mod)
	   (let ((new  (eval-expression (make-expression-reference mod '())
					store 
					guard)))
	       (update-store store mod new)))
     domain
     :initial-value store))

(defun eval-test (test true false)
    #'(lambda (store guard statement)
	  (let* ((kond (test-condition test))
		 (not-kond
		  (make-expression-call (function-entity[] ".NOT.")
					`(,kond)))
		 (true-statement (test-true test))
		 (false-statement (test-false test))
		 (kond-sexp (eval-expression kond store guard))
		 (not-kond-sexp (eval-expression not-kond store guard))
		 (and-function 
		  #'(lambda (ignore args)
			(make-expression-call
			 (function-entity[] ".AND.") args)))
		 (true-guard 
		  (shuffle-list :ignore 
				`(,kond-sexp ,guard)
				and-function
				#'(lambda (ignore) :ignore)))
		 (false-guard
		  (shuffle-list :ignore 
				`(,not-kond-sexp ,guard)
				and-function
				#'(lambda (ignore) :ignore)))
		 (mods (statement-written-entities statement))
		 (true-init-store (test-init-store mods store kond-sexp))
		 (false-init-store
		  (test-init-store mods store not-kond-sexp))
		 (true-store 
		  (funcall true 
			   true-init-store true-guard true-statement))
		 (false-store 
		  (funcall false
			   false-init-store false-guard false-statement)))
	      (reduce 
	       #'(lambda (store var)
		     (let* ((vt (get-store true-store var))
			    (vf (get-store false-store var))
			    (new (make-sexpression
				  :gexpressions
				  `(,@(sexpression-gexpressions vt)
				      ,@(sexpression-gexpressions vf)))))
			 (update-store store var new)))
	       mods
	       :initial-value store))))

;; We should evaluate the whole range.
;;
(defun eval-loop (loop body)
    #'(lambda (store guard statement)
	  (let* ((vars (statement-written-entities (loop-body loop)))
		 (end-s 
		  (funcall body (init-store vars) guard (loop-body loop)))
		 (upper (eval-expression (range-upper (loop-range loop))
					 store
					 guard))
		 (red-s (reduce
			 #'(lambda (store var)
			       (let* ((val (get-store end-s var))
				      (template (match-pattern
						 (create-instance var val)
					; should barf here!
						 (loop-index loop)))
				      (new-exp
				       (if template
					       (template-value template)
					   expression[gensym]))
				      (new-val (eval-expression
						new-exp store guard)))
				   (update-store store var new-val)))
			 vars
			 :initial-value end-s))
		 (out-s (update-store red-s (loop-index loop) upper)))
	      (when *debug-eval*
		  (format *debug-io* "~%Loop on ~A" 
			  (entity-name (loop-index loop)))
		  (print-store out-s `(,(loop-index loop) ,@vars)))
	      out-s)))

;;; Evaluation of Expressions to Sexpressions.

(defun eval-expression (exp store guard)
    (let* ((guards (mapcar
		    #'(lambda (gexp)
			  (let ((test (gexpression-guard gexp)))
			      (make-gexpression
			       :guard (make-expression-call
				       (function-entity[] ".AND.")
				       `(,test
					 ,(gexpression-expression gexp)))
			       :expression test)))
		    (sexpression-gexpressions guard)))
	   (s-guards (make-sexpression :gexpressions guards)))
	(gen-recurse 
	 exp
	 ((expression syntax) syntax)
	 ((syntax tag) tag)
	 ((reference indices)
	  (let ((id (reference-variable reference)))
	      (if (endp indices)
		      (shuffle-list :ignore
				    `(,s-guards
				      ,(get-store store id))
				    #'(lambda (ignore args)
					  (second args))
				    #'(lambda (ignore) :ignore))
		  (shuffle-list reference indices
				#'make-expression-reference
				#'reference-variable))))
	 ((call arguments)
	  (shuffle-list call arguments
			#'make-expression-call
			#'call-function)))))

;;; Shuffling of symbolic expressions.

(defmacro build (n c h args) `(funcall ,c (funcall ,h ,n) ,args))

(defun shuffle-list (node sexps constructor head)
    (let ((gexps
	   (if (null sexps)
		   `(,(make-gexpression 
		       :guard expression[true]
		       :expression (build node constructor head '())))
	       (mapcar #'(lambda (gexp)
			     (make-gexpression
			      :guard (gexpression-guard gexp)
			      :expression 
			      (build node constructor 
				     head (gexpression-expression gexp))))
		       (shuffle-list-1 sexps)))))
	(make-sexpression 
	 :gexpressions
	 (mapcar #'(lambda (gexp)
		       (make-gexpression
			:guard 
			(simplify-expression (gexpression-guard gexp))
			:expression 
			(simplify-expression (gexpression-expression gexp))))
		 gexps))))

(defun shuffle-list-1 (sexps)
    (reduce 
     #'(lambda (sexp res-gexps)
	   (mapcan 
	    #'(lambda (gexp)
		  (let ((test (gexpression-guard gexp))
			(exp (gexpression-expression gexp)))
		      (mapcar 
		       #'(lambda (res-gexp)
			     (make-gexpression
			      :guard (make-expression-call
				      (function-entity[] ".AND.")
				      `(,(gexpression-guard res-gexp)
					,test))
			      :expression 
			      `(,@(gexpression-expression res-gexp)
				  ,exp)))
		       res-gexps)))
	    (sexpression-gexpressions sexp)))
     (cdr sexps)
     :from-end t
     :initial-value
     (mapcar #'(lambda (gexp)
		    (make-gexpression
		     :guard (gexpression-guard gexp) 
		     :expression `(,(gexpression-expression gexp))))
	     (sexpression-gexpressions (car sexps)))))

;;; Management of stores

(defun entity-to-sexpression (entity)
    (make-sexpression
     :gexpressions
     `(,(make-gexpression :guard expression[true]
			  :expression (make-expression-reference 
				       entity '())))))

(defun update-store (store entity value)
    (acons entity value store))

(defun get-store (store entity)
    (let ((val (assoc (entity-name entity) store 
		      :test #'string= 
		      :key #'entity-name)))
	(if val
		(cdr val)
	    (entity-to-sexpression entity))))

(defun init-store (entities)
    (reduce #'(lambda (store entity)
		  (update-store store 
				entity
				(entity-to-sexpression entity)))
	    entities
	    :initial-value '()))

(defun store-domain (store)
    (mapcar #'car store))

;;; Debug stuff

(defun nice-expression (exp)
    (gen-recurse
     exp
     ((expression syntax) syntax)
     ((syntax tag) tag)
     ((reference indices)
      (format nil "~A(~{~A ~})"
	      (entity-name (reference-variable reference))
	      indices))
     ((call arguments)
      (format nil "~A(~{~A ~})"
	      (entity-name (call-function call))
	      arguments))))

(defun print-store (store vars)
    (mapc #'(lambda (var)
		(format *debug-io* "~%~A ->" (entity-name var))
		(mapc #'(lambda (gexp)
			    (format *debug-io* "~%    ~S,~S"
				    (nice-expression 
				     (gexpression-guard gexp))
				    (nice-expression 
				     (gexpression-expression gexp))))
			(sexpression-gexpressions (get-store store var))))
	  vars)
    :done)

;;;

(provide :eval)

	  
