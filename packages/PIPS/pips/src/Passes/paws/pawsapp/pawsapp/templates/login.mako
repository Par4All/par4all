<%doc>
  Page de login
</%doc>


<%inherit file="base.mako"/>

<%namespace name="index" file="index.mako"/>

## LEFT COLUMN

<%def name="left_column()">
${text|n}
</%def>


## MAIN COLUMN

<%def name="main_column()">

${index.site_header()}

<div class="alert alert-message">
  <i class="icon-warning-sign"></i>
  In order to use <b>PAWS</b>, you must first log in.
</div>



${h.form(url, method="post")}	

${h.hidden("came_from", came_from)}

<label for="login">Login</label>
${h.text("login", value=login, size=20)}

<label for="password">Password</label>
${h.password("password", size=20)}

<label>
  <button class="btn btn-primary" name="form.submitted" type="submit">
    Login <i class="icon-lock icon-white"></i></button>
  <button class="btn" type="reset">Cancel</button>
</label>

${h.end_form()}

</%def>

