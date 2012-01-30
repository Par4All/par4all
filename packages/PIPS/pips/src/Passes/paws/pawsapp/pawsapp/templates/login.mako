<%doc>
  Page de login
</%doc>


<%inherit file="base.mako"/>


## LEFT COLUMN

<%def name="left_column()">
In order to use PAWS, you must be logged in.
</%def>


## MAIN COLUMN

<%def name="main_column()">

<div class="hero-unit" style="padding:.5em 1em">
  <h2>PAWS Login form</h2>
</div>

${h.form(url, method="post")}	

<fieldset>

  ${h.hidden("came_from", came_from)}

  <div class="clearfix">
    <label for="login">Login</label>
    <div class="input">
      ${h.text("login", value=login, size=20)}
    </div>
  </div>

  <div class="clearfix">
    <label for="password">Password</label>
    <div class="input">
      ${h.password("password", size=20)}
    </div>
  </div>

  <div class="actions">
    ${h.submit("form.submitted", u"Login", class_="btn primary")}
    <button class="btn" type="reset">Cancel</button>
  </div>
</fieldset>

${h.end_form()}

</%def>

