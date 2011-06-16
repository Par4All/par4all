<!DOCTYPE html>

<%inherit file="skeleton.mako"/>

<%def name="head()">
	<title>PYPS DEMO PAGE</title>
        <link type="text/css" href="/jqueryui/css/cupertino/jquery-ui-1.8.10.custom.css" rel="stylesheet" />
        <script type="text/javascript" src="/jqueryui/js/jquery-1.4.4.min.js"></script>
        <script type="text/javascript" src="/jqueryui/js/jquery-ui-1.8.10.custom.min.js"></script>
        <script type="text/javascript">

		function initialize_descriptions() {
			main_text();
			functionalities();
		}		

		function main_text() {
			$.ajax({
				type: "GET",
				cache: false,
				url: "/descriptions/paws_text",
				error: function() {
					$('#left').html("Site can't be initialized!");
				},
				success: function(data){
					$('#left').html('<h4>' + data + '</h4>');
				}
			});
		}

		function functionalities() {
			$.getJSON("/descriptions/sections", function(data){
					
					$.ajax({
						type: "GET",
						cache: false,
						async: false,
						url: "/descriptions/accordion",
						error: function(){
							$('#right').html('ERROR');
						},
						success: function(acco) {
							$('#right').append(acco);
						}
					});
					for (i=0; i<data.length; i++) {
						$('#' + data[i]).accordion({
							active: false,
                                			animated: 'bounceslide',
		                        	        collapsible: true,
                			                autoHeight: false
						});
					}
				});
		}

		$(function() {
			$("#basic_tools").accordion({ 
				active: false,
				animated: 'bounceslide',
				collapsible: true,
				autoHeight: false
			});

			$("#tutorial").accordion({ 
                        	active: false,
                                animated: 'bounceslide',
                                collapsible: true,
                                autoHeight: false
                        });
	
			$("#full_control").accordion({ 
                        	active: false,
                                animated: 'bounceslide',
                                collapsible: true,
                                autoHeight: false
                        });

			initialize_descriptions();
		});
	</script>
	<style type="text/css">
        	body{ font: 62.5% "Trebuchet MS", sans-serif; margin: 50px;}
                .demoHeaders { margin-top: 2em; }
                ul#icons {margin: 0; padding: 0;}
                ul#icons li {margin: 2px; position: relative; padding: 4px 0; cursor: pointer; float: left;  list-style: none;}
                ul#icons span.ui-icon {float: left; margin: 0 4px;}
                input.hide { position: absolute; height: 30px; left: -90px; -moz-opacity: 0; filter: alpha(opacity: 0); opacity: 0; z-index: 2;}
		h3 { font-size: 0.9em;}
		h2 { font-size: 1.0em;}
		h4 { font-size: 1.2em;}
		h5 { font-size: 0.7em;}
                .left_side_buttons { margin-left:10px;}
        </style>

</%def>

<%def name="header()">
PYPS AS WEB SERVICE
</%def>

<%def name="content()">
	
	<div id="main" align="center">
		<table class="ui-widget ui-widget-content ui-corner-all"><tr valign="top"><td width="5%">
			<br/>
		</td><td width="20%">
			<div id="left">
			</div>
		</td><td width="5%">
			<br/>
		</td><td>
			<div id="right">
			</div>
		</td></tr></table>
	</div>
</%def>

<%def name="function_accordion(functions)">
${functions}
</%def>	
