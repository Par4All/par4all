<DOCTYPE html>

<%inherit file="skeleton.mako"/>

<%def name="head()">
	<title>PAWS</title>
	<link type="text/css" href="/jqueryui/css/cupertino/jquery-ui-1.8.10.custom.css" rel="stylesheet" />
	<link type="text/css" href="/jquery-linedtextarea/jquery-linedtextarea.css" rel="stylesheet" />
	<link type="text/css" href="/pygments/css/pygments.css" rel="stylesheet" />
	<link type="text/css" href="/jqzoom/css/jquery.jqzoom.css" rel="stylesheet" />
	<script type="text/javascript" src="/jqueryui/js/jquery-1.4.4.min.js"></script>
	<script type="text/javascript" src="/jqueryui/js/jquery-ui-1.8.10.custom.min.js"></script>
	<script type="text/javascript" src="/jquery-linedtextarea/jquery-linedtextarea.js"></script>
	<script type="text/javascript" src="/jqzoom/js/jquery.jqzoom-core.js"></script>
	<script type="text/javascript">

	var options = {
		zoomHeight: 400,
		zoomWidth: 400
	};

		function handle_key_down(item, event) {
			catch_tab(item, event);
			modified = true;
		}

                function replaceSelection(input, replaceString) {
                        if (input.setSelectionRange) {
                                var selectionStart = input.selectionStart;
                                var selectionEnd = input.selectionEnd;
                                input.value = input.value.substring(0, selectionStart) + replaceString + input.value.substring(selectionEnd);

                                if (selectionStart != selectionEnd) {
                                        setSelectionRange(input, selectionStart, selectionStart + replaceString.length);
                                } else {
                                        setSelectionRange(input, selectionStart + replaceString.length, selectionStart + replaceString.length);
                                }
                        } else if (document.selection) {
                                var range = document.selection.createRange();
                                if (range.parentElement() == input) {
                                        var isCollapsed = range.text = '';
                                        range.text = replaceString;
                                        if (!isColllapsed) {
                                                range.moveStart('character', -replaceString.length);
                                                range.select();
                                        }
                                }
                        }
                }

                function setSelectionRange(input, selectionStart, selectionEnd) {
                        if (input.setSelectionRange) {
                                input.focus();
                                input.setSelectionRange(selectionStart, selectionEnd);
                        }
                        else if (input.createTextRange) {
                                var range = input.createTextRange();
                                range.collapse(true);
                                range.moveEnd('character', selectionEnd);
                                range.moveStart('character', selectionStart);
                                range.select();
                        }
                }

                function catch_tab(item, e) {
                        c = e.keyCode;
                        if (c == 9) {
                                replaceSelection(item, String.fromCharCode(9));
                                setTimeout("document.getElementById('" + item.id + "').focus();", 0);
                                return false;
                        }
                }

		function initialize_slider(steps) {
			$('#demo_slider').slider('option', 'disabled', false);
			$('#demo_slider').slider('option', 'min', 0);
			$('#demo_slider').slider('option', 'max', parseInt(steps));
			$('#demo_slider').slider('option', 'animate', true);
			$('#demo_slider').slider('option', 'value', '0');
			$('#step').val('0');
			$('#all_steps').val(steps);		
		}

		function source_textarea(data) {
			return '<textarea name="sourcecode" id="sourcecode" class="output" rows="33" cols="85" onkeydown="handle_key_down(this, event)">' + data + '</textarea>'	
		}

		function source_resultarea(data) {
			return '<div id="sourcecode" name="sourcecode" class="output">' + data + '</div>'
		}

		function initialize_demo() {
			modified = false;
			$.ajax({
				type: 'POST',
				data: { name: ${self.file_name()}},
				cache: false,
				async: false,
				url: '/demo/get_steps_number',
				error: function(){
					$('#sourcetpips').html("Web error, demo can't be initialize!");
					return;
				},
				success: function(data){
					initialize_slider(data);
				}
			});					
			$.getJSON("/demo/load_demo", { name: ${self.file_name()}}, function(data){
				$('#sourcetpips').html(data[0]);
				$('#source').html(source_textarea(data[1]));
				$('#sourcecode').linedtextarea();
				$('#sourcecode').attr('spellcheck', false);
			});
			$('#demo_slider').slider('option', 'value', '0');
		}

		$(function(){
			$('#demo_slider').slider({value:0});
			$('#demo_slider').slider('option', 'disabled', true);
			$('#demo_slider').slider('option', 'slide', function(event, ui) {
				$('#step').val(ui.value);
				if (modified) {
					$.ajax({
						type: 'POST',
						data: { 
							code: $('#sourcecode').val(),
							name: ${self.file_name()}
						},
						cache: false,
						async: false,
						url: "/demo/change_source",
						error: function(){
							$('#sourcetpips').html("Web error, can't perform demo.");
						},
						success: function(data){
							modified = false;
						}
					});
				}
				$.ajax({
					type: 'POST',
					data: {
						step: ui.value,
						name: ${self.file_name()}
					},
					cache: false,
					url: "/demo/get_step_output",
					error: function(){
						$('#sourcetpips').html("Web error, can't perform demo.");
					},
					success: function(data){
						if (ui.value != '0') {
							$('#source').html(source_resultarea(data));
							$('.ZOOM_IMAGE').jqzoom(options);
						} else {
							$('#source').html(source_textarea(data));
							$('#sourcecode').linedtextarea();
						}
					}
				});
				$.ajax({
					type: 'POST',
					data: {step: ui.value},
					cache: false,
					url: "/demo/get_step_script",
					error: function(){
						$('#sourcetpips').html("Web error, can't perform demo.");
					},
					success: function(data){
						$('#sourcetpips').html(data);
					}
				});
			});

			$('#step').val($('#demo_slider').slider('value'));
			$('#all_steps').val('0');
			
			$('#sourcecode').linedtextarea();
			$('#sourcecode').attr('spellcheck', false);

			$('#dialog-error-examples').dialog({
				autoOpen: false,
				width: 400,
				buttons: {
					"OK": function(){
						$(this).dialog("close");
					}
				}
			});

			$('#dialog-load-examples').dialog({
				autoOpen: false,
				width: 400
			});

			initialize_demo();

			$("input:submit", ".load_examples").button();
			$("input:submit", ".load_examples").click(function(event){
				
				event.preventDefault();
				$.ajax({
					type: "POST",
					data: {operation: "demo"},
					cache: false,
					url: "/examples/get_examples",
					error: function(){
						$('#dialog-error-examples').dialog('open');
					},
					success: function(data){
						var dialog_content = data;
						$('#select-examples-buttons').html(dialog_content);

						$('#select-examples-buttons input:submit').button();
						$('#select-examples-buttons input:submit').click(function(){
							
							var filename = $(this).attr('value');
							modified = false;
							$.ajax({
								type: 'POST',
								data: { name: ${self.file_name()}},
								cache: false,
								async: false,
								url: '/demo/get_steps_number',
								error: function(){
									$('#sourcetpips').html("Web error, demo can't be initialize!");
								},
								success: function(data){
									initialize_slider(data);
								}
							});
					
							$.ajax({
								type: 'POST',
								data: { name: ${self.file_name()}},
								cache: false,
								url: "/demo/load_demo_tpips",
								error: function(){
									$('#sourcetpips').html("Web error, try again later.");
								},
								success: function(data){
									$('#sourcetpips').html(data);
								}
							});
							$.ajax({
								type: 'POST',
								data: { name: ${self.file_name()}},
								cache: false,
								url: "/demo/load_demo_source",
								error: function(){
									$('#sourcecode').html("Web error, try again later.");
								},
								success: function(data){
									$('#source').html(source_textarea(data));
									$('#sourcecode').linedtextarea();
								}
							});
							$('#dialog-load-examples').dialog('close');
							$('#demo_slider').slider('option', 'value', '0');
						});

						$('#dialog-load-examples').dialog('open');
					}
				});
			});
		})
	</script>
	<style type="text/css">
        	body{ font: 62.5% "Trebuchet MS", sans-serif; margin: 50px;}
                .demoHeaders { margin-top: 2em; }
        	ul#icons {margin: 0; padding: 0;}
                ul#icons li {margin: 2px; position: relative; padding: 4px 0; cursor: pointer; float: left;  list-style: none;}
                ul#icons span.ui-icon {float: left; margin: 0 4px;}
                input.hide { position: absolute; height: 30px; left: -90px; -moz-opacity: 0; filter: alpha(opacity: 0); opacity: 0; z-index: 2;}
                h3 { color: #185172;}
 		h4 { color: #185172;}
		.table_header { color: #185172;}
                .left_side_buttons { margin-left:10px;}
		.slider_label { border:0; font-weight:bold; margin-bottom:10px;}
		.boxpopup {font-family: Arial, sans-serif; font-size:90%; color:black; background:#DEEDF7;width:200px;text-align:center;padding:4px 5px 4px 5px;font-weight:bold;border:1px solid gray;}
		#pdqbox {position:absolute; visibility:hidden; z-index:200;}
		.output {font-size:0.8em;}
	</style>
</%def>

<%def name="header()">
${self.file_name()}
</%def>

<%def name="content()">
	<div id="dialog-error-examples" title="ERROR">
		<p>Error while loading examples, try again!</p>
	</div>

        <div id="dialog-load-examples" title="Select an example.">
        	<div class="select-examples" id="select-examples-buttons">
                	<input value="LOAD" type="submit">
        	</div>
	</div>

	<table class="ui-widget ui-widget-content ui-corner-all"><tr valign="top"><!--<td>
		<div id="left_side">
			<br/><br/><br/><br/><br/><br/>
			<p><h3>CHECK HOW PIPS IS WORKING: </h3></p>
			<br/><br/>
                        <div class="load_examples left_side_buttons">
                                <input value="CHOOSE DEMO" type="submit"/>
                        </div>
			<br/><br/>
			<form action="/" id="buttons">
				<div class="operation left_side_buttons">
					<input value="RUN" type="submit"/>
				</div>
			</form>
		</div>
	</td>--><td width="20">&nbsp;
	</td><td>
        	<div id="demo">
			<br/>
			<label for="step" class="table_header">Current demo step:&nbsp;&nbsp; </label> <input type="text" id="step" class="ui-widget ui-widget-content slider_label"/><br/>
			<label for="all_steps" class="table_header">Number of all the steps:&nbsp;&nbsp; </label> <input type="text" id="all_steps" class="ui-widget ui-widget-content slider_label"/>
			<div id="demo_slider"></div>
			<br/>
			<table><tr><td>
				<div class='table_header'>SOURCE:</div>
			</td><td>
				<div class='table_header'>SCRIPT:</div>
			</td></tr><tr><td valign='top' align='left' width='700px'>
				<div id="source">
					<textarea name="sourcecode" id="sourcecode" class="output" rows="33" cols="85" onkeydown="handle_key_down(this, event)"></textarea>
				</div>
			</td><td valign='top' align='left' width='700px'>
				<div id="sourcetpips" class="output">
					<p align="center"><br/><br/>Loading ...</p>
				</div>
			</td></tr></table>
                </div>
	</td></tr></table>
</%def>

<%def name="operation_id()">
"demo"
</%def>
