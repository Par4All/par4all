; Some variables used to store informations about an EPips buffer:
; To store the module name in a buffer:
(make-variable-buffer-local 'epips-local-module-name)
; To store the mode name in a buffer, such as Fortran:
(make-variable-buffer-local 'epips-local-mode)


; The faces used to display various informations :
(if (x-display-color-p)
    (progn
     ; We have a color screen:
     (copy-face 'bold-italic 'epips-face-prompt-user)
     (set-face-foreground 'epips-face-prompt-user "grey")

     (copy-face 'bold-italic 'epips-face-user-error)
     (set-face-foreground 'epips-face-user-error "red")

     (copy-face 'default 'epips-face-user-log)
					;(set-face-foreground 'epips-face-user-log "blue")
     (set-face-foreground 'epips-face-user-log "DodgerBlue1")

     (copy-face 'bold-italic 'epips-face-user-warning)
     (set-face-foreground 'epips-face-user-warning "orange")
     )
  (progn
     ; No...
     (copy-face 'bold-italic 'epips-face-prompt-user)
     (set-face-underline-p 'epips-face-prompt-user t)

     (copy-face 'bold 'epips-face-user-error)
     (invert-face 'epips-face-user-error)

     (copy-face 'bold 'epips-face-user-log)
      
     (copy-face 'bold-italic 'epips-face-user-warning)
   )
  )

; The position of the EPips frames:
(add-to-list
 'special-display-regexps `("Pips-Log" ,@(x-parse-geometry "80x10+0-0")))
(add-to-list
 'special-display-regexps `("Emacs-Pips-[048]" ,@(x-parse-geometry "80x22-0+0")))
(add-to-list
 'special-display-regexps `("Emacs-Pips-[159]" ,@(x-parse-geometry "80x22-0-0")))
(add-to-list
 'special-display-regexps `("Emacs-Pips-[26]" ,@(x-parse-geometry "80x22+0+0")))
(add-to-list
 'special-display-regexps `("Emacs-Pips-[37]" ,@(x-parse-geometry "80x22+0-0")))


; To print some messages during the epips debugging:
(defun epips-debug (something)
  ;(print something (current-buffer))
  )

					; The token to mark the begin
					; and end of command. Use some
					; strings to be usually never
					; used in ISO-8859-1:
(setq
 epips-begin-of-command-token "\200"
 epips-end-of-command-token "\201"
 epips-send-begin-of-command-token "\202"
 epips-send-end-of-command-token "\203"
 epips-command-separator ":"
 )

; All the command string that are used to name commands between Pips
; and emacs:
(setq
 epips-array-data-flow-graph-view-command-name "Array data flow graph View"
 epips-bank-view-command-name "BANK"
 epips-call-graph-view-command-name "Callgraph View"
 epips-dependance-graph-view-command-name "Dependence Graph View"
 epips-distributed-view-command-name "Distributed View"
 epips-emacs-sequential-view-command-name "Emacs Sequential View"
 epips-flint-view-command-name "Flint View"
 epips-hpfc-file-view-command-name "HPFC File"
 epips-ICFG-view-command-name "ICFG View"
 epips-module-command-name "MODULE_NAME"
 epips-parallel-view-command-name "Parallel View"
 epips-placement-view-command-name "Placement View"
 epips-prompt-user-command-name "PROMPT_USER"
 epips-scheduling-view-command-name "Scheduling View"
 epips-sequential-view-command-name "Sequential View"
 epips-user-error-command-name "USER_ERROR"
 epips-user-log-command-name "USER_LOG"
 epips-user-view-command-name "User View"
 epips-user-warning-command-name "USER_WARNING"
 epips-window-number-command-name "WINDOW_NUMBER"
 )


(defun epips-select-and-display-a-buffer (name)
  "The function used to select and display a buffer."
  (let (
        ; Create a new frame if necessary to display the buffer:
	;(pop-up-frames t)
        )
    (pop-to-buffer name)
    )
  )


(defun epips-raw-insert (a-string
			 begin-position
			 end-position
			 a-process)
  "A function that insert the characters begin-position to end-position from a-string."
  (epips-select-and-display-a-buffer (process-buffer a-process))
  (epips-debug 'epips-raw-insert)
  (epips-debug begin-position)
  (epips-debug end-position)
  (epips-debug (substring a-string
			  begin-position
			  end-position))
  (insert (substring a-string
		     begin-position
		     end-position))
  )


; A function that insert some text with a property list :
(defun epips-insert-with-properties (some-text property-list)
  (epips-select-and-display-a-buffer (process-buffer a-process))
  (let ((old-point (point)))
    (insert some-text)
    (add-text-properties old-point
			 (point)
			 property-list))
  )


; Here are defined the various function to deal with the Pips actions:

; Executed from send_module_name_to_emacs() in emacs.c:
(defun epips-module-name-command (epips-command-content)
  (epips-debug 'epips-module-name-command)
  (setq epips-current-module-name epips-command-content)
  )  


; Executed from send_prompt_user_to_emacs() in emacs.c:
(defun epips-prompt-user-command (epips-command-content)
  (epips-debug 'epips-prompt-user-command)
  (epips-insert-with-properties epips-command-content
				'(face epips-face-prompt-user))
  (epips-raw-insert "\n"
		    0
		    nil
		    a-process)
  )  


; Executed from send_user_error_to_emacs() in emacs.c:
(defun epips-user-error-command (epips-command-content)
  (epips-debug 'epips-user-error-command)
  (epips-insert-with-properties epips-command-content
				'(face epips-face-user-error))
  )  


; Executed from send_user_log_to_emacs() in emacs.c:
(defun epips-user-log-command (epips-command-content)
  (epips-debug 'epips-user-log-command)
  (epips-insert-with-properties epips-command-content
				'(face epips-face-user-log))
  )  


(defun epips-user-warning-command (epips-command-content)
  "Executed from send_user_warning_to_emacs() in emacs.c"
  (epips-debug 'epips-user-warning-command)
  (epips-insert-with-properties epips-command-content
				'(face epips-face-user-warning))
  )  


(defun epips-fortran-mode-and-hilit ()
  "Go in Fortran mode and Hilit19"
  (fortran-mode)
  (if (fboundp 'hilit-rehighlight-buffer)
      (hilit-rehighlight-buffer)
    )
  (epips-add-keymaps-and-menu-in-the-current-buffer)
  )


; Initialize the associative list that record the output sent by each
; xtree process:
(setq epips-xtree-list-output '())

(defun epips-xtree-output-filter (a-process an-output-string)
  "Define a filter to interpret the standard output of xtree"
  (let*
      (
					; Allow interruption inside the filter
       (inhibit-quit nil)
					; Get the old output string of
					; this process:
       (epips-this-process-old-output
	(assoc a-process epips-xtree-list-output))
					; Concatenate the new output
					; to the old one
       (full-output-string (concat (cdr epips-this-process-old-output)
				   an-output-string))
       (epips-xtree-output-filter-newline nil)
       )
    					; Each new line is a module
					; name:
    (while (setq epips-xtree-output-filter-newline
		   (string-match "\n" full-output-string))
					; Ok, we received something up
					; a newline from xtree.
      (let* (
	     (one-line-from-xtree
	      (substring full-output-string
			 0
			 epips-xtree-output-filter-newline)
	      )
					; In fact, only the leaf name
					; is needed up to now:
	     (point-place (string-match "[^.]*$" one-line-from-xtree))
	     )
					; Apply a command with the
					; module name clicked with the
					; mouse:
	
					; For exemple, display the
					; code of the module:
      (epips-send-sequential-view-command
       (substring one-line-from-xtree point-place nil))
      )
       
					; Discard the part that is
					; already executed:
      (setq full-output-string
	    (substring full-output-string
		       (1+ epips-xtree-output-filter-newline)
		       nil))
      )
					; To remind the output of this
					; process:
    (setcdr epips-this-process-old-output full-output-string)
    )
  )

  
(defun epips-ICFG-or-graph-view-command (epips-command-name
					 epips-command-content)
  "Display a graph with xtree"
  (let*
      (
					; Do not use a pipe to
					; communicate since it looks
					; like a flush is lacking
					; somewhere and the output is
					; sent to emacs only when
					; xtree exits if so... No
					; (process-connection-type
					; nil)
       (epips-xtree-process
					; Do not use intermediate shell:
	(start-process "xtree"
		       "Pips-Xtree-Log"
		       "xtree"
		       "-name" epips-command-name
		       "-title" epips-current-module-name
		       "-bg" "LightSteelBlue1" "-fg" "purple4"
		       "-separator" "    "
		       "-oformat" "resource"
		       )
	)
					; That mean that we can not a
					; file through stdin and need
					; a temporary buffer:
       (epips-xtree-input-buffer (get-buffer-create "epips-xtree-input"))
       )
    (save-excursion
					; The process to understand
					; the click on nodes in xtree:
      (set-process-filter epips-xtree-process 'epips-xtree-output-filter)
      (set-buffer epips-xtree-input-buffer)
					; Read the ICFG output file:
      (insert-file-contents epips-command-content)
					; Remove the empty lines:
      (perform-replace "^\012" "" nil t nil)
					; Indent the comments lines as
					; the following one to have
					; nodes at the same depth:
      (goto-char (point-min))
      (perform-replace "^\\(C.*\\)\012\\( *\\)" "\\2\\1\012\\2" nil t nil)
					; Send the file content to the
					; xtree process:
      (process-send-region epips-xtree-process (point-min) (point-max))
					; And close stdin to draw the
					; tree:
      (process-send-eof epips-xtree-process)
      (kill-buffer epips-xtree-input-buffer)
					; Reset the output string of
					; this xtree process:
      (setq epips-xtree-list-output
	    (append (` (( (, epips-xtree-process )  . "")) )
		    epips-xtree-list-output))
					; The end of an xtree process
					; would need to clean
					; epips-xtree-list-output...
      )
    )
  )



(defun epips-sequential-view-command (epips-command-name
				      epips-command-content)
  "Executed from send_view_to_emacs() in emacs.c"
  (epips-debug 'epips-sequential-view-command)
  (save-excursion
    (let (
					; Change the window and icon headers:
	  (frame-title-format (list "%b (" 'mode-name ")"))
	  (icon-title-format (list "" 'mode-name ""))
	  )
					; Switch to the buffer number
					; epips-current-window-number:
      (epips-select-and-display-a-buffer
       (aref epips-buffers epips-current-window-number))
					; Erase an old content:
      (delete-region (point-min) (point-max))
					; Insert the file generated by
					; Pips:
      (insert-file-contents epips-command-content)
      (epips-fortran-mode-and-hilit)
      (setq
       epips-local-module-name epips-current-module-name
       epips-local-mode mode-name
       mode-name (concat epips-command-name ": "
			 epips-current-module-name))
					;(setq list-buffers-directory "T")
      )
    )
  )


(defun epips-view-command (epips-command-name epips-command-content)
  "Executed from send_view_to_emacs() in emacs.c"
  (epips-debug 'epips-view-command)
  (if (equal epips-command-name epips-array-data-flow-graph-view-command-name)
      (epips-sequential-view-command epips-command-name epips-command-content)
    (if (equal epips-command-name epips-bank-view-command-name)
	(epips-sequential-view-command epips-command-name epips-command-content)
      (if (equal epips-command-name epips-call-graph-view-command-name)
          (epips-ICFG-or-graph-view-command epips-command-name epips-command-content)
        (if (equal epips-command-name epips-dependance-graph-view-command-name)
            (epips-sequential-view-command epips-command-name epips-command-content)
          (if (equal epips-command-name epips-distributed-view-command-name)
              (epips-sequential-view-command epips-command-name epips-command-content)
            (if (equal epips-command-name epips-emacs-sequential-view-command-name)
                (epips-sequential-view-command epips-command-name epips-command-content)
              (if (equal epips-command-name epips-flint-view-command-name)
                  (epips-sequential-view-command epips-command-name epips-command-content)
		(if (equal epips-command-name epips-hpfc-file-view-command-name)
		    (epips-sequential-view-command epips-command-name epips-command-content)
		  (if (equal epips-command-name epips-ICFG-view-command-name)
		      (epips-ICFG-or-graph-view-command epips-command-name epips-command-content)
		    (if (equal epips-command-name epips-parallel-view-command-name)
			(epips-sequential-view-command epips-command-name epips-command-content)
		      (if (equal epips-command-name epips-placement-view-command-name)
			  (epips-sequential-view-command epips-command-name epips-command-content)
			(if (equal epips-command-name epips-scheduling-view-command-name)
			    (epips-sequential-view-command epips-command-name epips-command-content)
			  (if (equal epips-command-name epips-sequential-view-command-name)
			      (epips-sequential-view-command epips-command-name epips-command-content)
			    (if (equal epips-command-name epips-user-view-command-name)
				(epips-sequential-view-command epips-command-name epips-command-content)
					; Else, command unknown:
			      (epips-user-error-command (concat "\nCommand name \""
								epips-command-name
								"\" with argument \""
								epips-command-content
								"\" not implemented !!!\n\n"))
			      )
			    )
                          )
                        )
		      )
		    )
		  )
		)
	      )
	    )
	  )
	)
      )
    )
  )

; Executed from send_window_number_to_emacs() in emacs.c:
(defun epips-window-number-command (epips-command-content)
  (epips-debug 'epips-window-number-command)
  (setq epips-current-window-number (string-to-number epips-command-content))
  )  


; This function try to execute a command sent by Pips:
(defun epips-execute-output-command (command-string
				     command-begin-position
				     command-end-position
				     a-process)
  (epips-debug 'epips-execute-output-command)
					; Focus on the interesting
					; part of the string:
  (setq command-string (substring command-string
				  (+ command-begin-position
				     (length epips-begin-of-command-token))
				  command-end-position))
  (epips-debug command-string)
					;The command has the format
					;"command_name:command_content":
  (setq epips-command-separator-position
	(string-match epips-command-separator command-string))
  (setq epips-command-name (substring command-string
				      0
				      epips-command-separator-position))
  (epips-debug epips-command-name)
  (setq epips-command-content (substring command-string
					 (+ epips-command-separator-position
					    (length epips-command-separator))
					 nil))
  (epips-debug epips-command-content)
  (if (equal epips-command-name epips-module-command-name)
      (epips-module-name-command epips-command-content)
    (if (equal epips-command-name epips-prompt-user-command-name)
	(epips-prompt-user-command epips-command-content)
      (if (equal epips-command-name epips-user-error-command-name)
	  (epips-user-error-command epips-command-content)
	(if (equal epips-command-name epips-user-log-command-name)
	    (epips-user-log-command epips-command-content)
	  (if (equal epips-command-name epips-user-warning-command-name)
	      (epips-user-warning-command epips-command-content)
	    (if (equal epips-command-name epips-window-number-command-name)
		(epips-window-number-command epips-command-content)
	      ; It may be a view command:
	      (epips-view-command epips-command-name epips-command-content)
	      )
	    )
	  )
	)
      )
    )
  )


; Parse the output of wpips to see if there are some commands inside:
(defun epips-analyse-output (a-process an-output-string)
  (setq epips-packet-begin-position nil
	epips-packet-end-position nil)
  (if
      (equal
       epips-output-automaton-state
       'epips-output-automaton-state-wait-for-begin
       )
					; We are waiting for a begin
					; of packet:
      (progn
					; Search for a begin of packet:
	(setq epips-packet-begin-position
	      (string-match epips-begin-of-command-token an-output-string))
	(epips-debug 'epips-packet-begin-position)
	(epips-debug epips-packet-begin-position)
					; Display all the string up to
					; a potential
					; epips-begin-of-command-token:
	(epips-raw-insert an-output-string
			  0
			  epips-packet-begin-position
			  a-process)
	(if epips-packet-begin-position
					; If we have found a begin,
					; look for an end from the
					; begin position:
	    (progn
					; Raw output up to the command
					; begin :
	      (setq epips-packet-end-position
		    (string-match epips-end-of-command-token an-output-string
				  (+ 1 epips-packet-begin-position)))
	      (epips-debug 'epips-packet-end-position)
	      (epips-debug epips-packet-end-position)
	      (if epips-packet-end-position
					; Ok, we have the end of
					; packet in the same string
					; too
		  (progn
					; Execute the command
		    (epips-execute-output-command
		     an-output-string
		     epips-packet-begin-position
		     epips-packet-end-position
		     a-process)
		    (setq
					; We discard the command from
					; the an-output-string:
		     an-output-string
		     (substring an-output-string
				(+
				 epips-packet-end-position
				 (length epips-end-of-command-token))
				nil))
					; We stay in the
					; wait-for-begin state.
		    )
	  
					; Else we do not have the end
					; of packet yet so we store
					; the command...
		(setq epips-output-command-string
		      (substring
		       an-output-string
		       epips-packet-begin-position
		       nil)
					; We empty the output-string:
		      an-output-string ""
					; And stay in the wait-for-end
					; state:
		      epips-output-automaton-state
		      'epips-output-automaton-state-wait-for-end
		      )
		)
	      )
					; Else, no command found, we
					; stay in the wait-for-begin
					; state and empty the
					; output-string:
	  (setq an-output-string "")

	  )
	)
					; Else:
					; We are waiting for an end of
					; packet:
    (progn
					; Search for an end of packet:
      (setq epips-packet-end-position
	    (string-match epips-end-of-command-token an-output-string))
      (if epips-packet-end-position
					; If we have found an end, we
					; can send the command:
	  (progn
	    (
	     epips-execute-output-command
	     (concat epips-output-command-string
		     (substring an-output-string
				0
				epips-packet-end-position))
	     0
	     nil
	     a-process)

	    (setq
					; We leave the wait-for-end
					; state:
	     epips-output-automaton-state
	     'epips-output-automaton-state-wait-for-begin
					; An leave the rest of the
					; line
	     an-output-string
	     (substring an-output-string
			(+
			 epips-packet-end-position
			 (length epips-end-of-command-token))
			nil)
	     )
	    )
	(setq
					; Else it is a piece of
					; command, thus we just add
					; the string to the command
					; string
	 epips-output-command-string (concat
				      epips-output-command-string
				      an-output-string)
					; End empty the output string
	 an-output-string "")
	)
      )
    )
  (epips-debug epips-packet-begin-position)
  (epips-debug epips-packet-end-position)
  (epips-debug epips-output-automaton-state)
					; Return the remaining
					; an-output-string:
  an-output-string
  )


(defun epips-output-filter (a-process an-output-string)
  "Define a filter to interpret the standard output of wpips:
   The outline come from the E-Lisp manual about \"Process Filter Function\"."
  (let
      (
       (old-buffer (current-buffer))
       (inhibit-quit nil)		; Allow interruption inside the filter
       )
    (unwind-protect
	(let (moving)
					; By default, go to the end of
					; the buffer controling the
					; process:
	  (set-buffer (process-buffer a-process))
	  (setq moving (= (point) (process-mark a-process)))
	  (save-excursion
	    (goto-char (process-mark a-process))
					; Parse the output of wpips to
					; see if there are some
					; commands inside:
	    (while (progn
					; Loop on each semantical
					; piece
		     (setq an-output-string (epips-analyse-output
					     a-process
					     an-output-string))
		     (epips-debug "Return of epips-analyse-output:")
		     (epips-debug an-output-string)
					; Until it returns an empty
					; string:
		     (not (equal an-output-string ""))))

	    (set-marker (process-mark a-process) (point))
	    )
	  (if moving (goto-char (process-mark a-process)))
	  (set-buffer old-buffer)
	  )
      )
    )
  )


; Here are the functions used to send a command to Pips:

(defun epips-send-a-command-to-pips (command-name &optional command-content)
  "Send a command with command-name and command-content to the Pips process"
  (if (not command-content)
      (setq command-content "")
    )
  (process-send-string epips-process
		       (concat epips-send-begin-of-command-token
			       command-name
			       epips-command-separator
			       command-content
			       epips-send-end-of-command-token))
  )


(defun epips-send-module-select-command (module-name)
  "Send a command for choosing a module to the Pips process"
  (epips-send-a-command-to-pips epips-module-command-name module-name)
  )


(defun epips-send-sequential-view-command (&optional module-name)
  "Send a command for displaying the sequential view of the current module"
  (epips-send-a-command-to-pips epips-sequential-view-command-name module-name)
  )


(defun epips-mouse-module-select (event)
  "Select the module with the name under the mouse"
  (interactive "e")
  (mouse-set-point event)
					; Guess that the module name
					; is the word where the user
					; clicked:
  (let ((module-name (thing-at-point 'word)))
	(epips-send-sequential-view-command module-name)
	)
  )


(defun epips-save-to-seminal-file ()
  "Save the current file with a \".f\" file name.
  In this way, PIPS can reparse it and use the modifications done
  by the user or even by PIPS"
  (interactive)
  (mouse-set-point event)
					; Guess that the module name
					; is the word where the user
					; clicked:
;  (let ((module-name (thing-at-point 'word)))
;	(epips-send-sequential-view-command module-name)
;	)
  )

; Various initialization things:

;(setq epips-menu-keymap (cons "Pips" (make-sparse-keymap "Pips")))
;(fset 'epips-menu epips-menu-keymap)
(defun epips-add-keymaps-and-menu-in-the-current-buffer ()
  "This function add the menus and define some keyboard accelerators
 to the current buffer"
  (local-set-key [menu-bar epips] (cons "Pips" (make-sparse-keymap "Pips")))
  (local-set-key [menu-bar epips epips-kill-the-buffers]
		 '("Kill the Pips Log buffer" . epips-kill-the-log-buffer))
  (local-set-key [menu-bar epips epips-kill-the-buffers]
		 '("Kill the Pips buffers" . epips-kill-the-buffers))
  (local-set-key [menu-bar epips epips-clear-log-buffer]
		 '("Clear log buffer" . epips-clear-log-buffer))
  (local-set-key [S-down-mouse-1]
		 '("Go to module" . epips-mouse-module-select))
  (local-set-key [S-mouse-1]
		 '("Nothing..." . ignore))
  (local-set-key "\C-C\C-C"
		 '("Save the file in the seminal .f" . epips-save-to-seminal-file))
  )

(defun epips-add-keymaps-and-menu ()
  "This function add the menus and define some keyboard accelerators"
					; Add the menus on all the
					; Pips windows and log window:
  (mapcar '(lambda (a-buffer)
	     (set-buffer a-buffer)
	     (epips-add-keymaps-and-menu-in-the-current-buffer)
	     )
	  (append epips-buffers (list epips-process-buffer))
	  )
)


(defun epips-clear-log-buffer ()
  "This function clean the log buffer"
  (interactive)
  (save-excursion
					; Switch to the log buffer:
    (set-buffer epips-process-buffer)
					; Erase an old content:
    (delete-region (point-min) (point-max))
    )
  )
  


; The function to created all the EPips buffers
(defun epips-create-the-buffers ()
  (setq epips-buffers (make-vector epips-buffer-number nil))
  (setq i 0)
  (while (< i epips-buffer-number)
					; Create each window:
    (aset epips-buffers i (get-buffer-create
			   (format "Emacs-Pips-%d" i)))
    (setq i (1+ i))
    )		     
  (print epips-buffers)
  )


(defun epips-kill-the-buffers ()
  "The function to kill the EPips Log buffer"
  (interactive)
  (kill-buffer epips-process-buffer)
)


(defun epips-kill-the-buffers ()
  "The function to kill all the EPips buffers"
  (interactive)
  (mapcar 'kill-buffer epips-buffers)
)


; Initialize the buffers to display PIPS stuff:
(setq epips-buffer-number 9)

; Launch the wpips process from Emacs:
(defun epips ()
  "The Emacs-PIPS mode.

The Fortran codes are dealt with the Fortran mode of Emacs.

Some hypertext is added to PIPS:
- by clicking with Shift-mouse-1 (Shift and the mouse left button)
on a function or procedure name, the module is selected and displayed.
The buffers used to display PIPS output are named from Emacs-Pips-0
to Emacs-Pips-9

You can edit the text in a PIPS buffer. After that, you can save the
modified file with \C-C \C-C in the \".f\" file so that PIPS will take
in account your modifications up to now.

The Call Graph and ICFG view are displayed with xtree.
By pressing an xtree node, it is possible to select a module
and display its code as with Shift-mouse-1 in an Emacs-PIPS window.

The log window is a buffer named Pips-Log.

In each PIPS buffer, a menu is available to empty the log buffer and 
even to kill the PIPS buffer when they are no longer useful after 
exiting the PIPS mode.

How does it work? It launches mainly a wpips with the -emacs option...

Killing the Pips-Log buffer kills also the WPips process.
It is useful to interrupt a core dump of 250 MBytes when it appends
for example... :-)

By the way, EPips assumes the use of hilit19...
"
					; Just to have the function
					; for the user :
  (interactive)
  (kill-all-local-variables) ; Clean up the environment.
					; Initialize the automaton
					; that analyses the pips
					; output:
  (setq epips-output-automaton-state
	'epips-output-automaton-state-wait-for-begin)
					; Initialize various variables:
  (setq
					; No module name, yet:
   epips-current-module-name nil
					; No window number yet:
   epips-current-window-number nil
   )
					; Create the display buffers:
  (epips-create-the-buffers)

  (let
      (
       (process-connection-type nil)	; Use a pipe to communicate
       )
    (setq epips-process (start-process "wpips" "Pips-Log" "wpips" "-emacs"))
					;(setq epips-process (start-process "wpips" "epips" "ls" "-lag"))
					;(goto-char (process-mark epips-process))
    (setq epips-process-buffer (process-buffer epips-process))
    (set-process-filter epips-process 'epips-output-filter)
    (epips-select-and-display-a-buffer epips-process-buffer)
					;(switch-to-buffer
					; epips-process-buffer) Hum, I
					; do not know why I need to
					; initialize (process-mark
					; epips-process) if I do not
					; want a rude #<marker in no
					; buffer>. It used to work, but...
    (goto-char (point-max))
    (set-marker (process-mark epips-process) (point))
    (epips-add-keymaps-and-menu)
    )
  )

(defun v-epips-output-filter (a-process an-output-string)
  (let
      (
       (old-buffer (current-buffer))
       (inhibit-quit nil)		; Allow interrupion inside the filter
       )
    (unwind-protect
	(let (moving)
	  (set-buffer (process-buffer a-process))
	  (setq moving (= (point) (process-mark a-process)))
	  (save-excursion
	    ;; Insert the text, moving the process-marker.
	    (goto-char (process-mark a-process))
	    (insert an-output-string)
	    (set-marker (process-mark a-process) (point)))
	  (if moving (goto-char (process-mark a-process)))
	  (set-buffer old-buffer)
	  )
      )
    )
  )
