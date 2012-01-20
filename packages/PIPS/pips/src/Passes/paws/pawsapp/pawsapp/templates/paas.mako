## paas.mako


<%inherit file="skeleton.mako"/>


<%def name="head()">

<title>PYPS DEMO PAGE</title>

<link type="text/css" href="/css/jq/cupertino/jquery-ui-1.8.10.custom.css" rel="stylesheet" />
<link type="text/css" href="/css/paas.css" rel="stylesheet" />

<script type="text/javascript" src="/jq/jquery-1.4.4.min.js"></script>
<script type="text/javascript" src="/jq/jquery-ui-1.8.10.custom.min.js"></script>
<script type="text/javascript" src="/js/paas.js"></script>

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
