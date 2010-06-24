#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Authors:
# - Grégoire Péan <gregoire.pean@hpc-project.com>
#

'''
Par4All common option parsing
'''

import string, sys, optparse, smtplib
from email.mime.text import MIMEText
from p4a_util import *
from p4a_version import *

static_options = None
static_args = None


def add_common_options(parser):

    group = optparse.OptionGroup(parser, "General Options")

    group.add_option("--verbose", "-v", action = "count", default = get_verbosity(),
        help = "Run in verbose mode (you can have several -v, such as -vvv which will display the most debugging information).")

    group.add_option("--log", action = "store_true", default = False,
        help = "Enable logging in current directory.")

    group.add_option("--report", action = "store_true", default = False,
        help = "Send a report email to Par4All support email address in case of error. "
            + "This implies --log (it will log to a distinct file every time). "
            + "The report will be anonymous and no file or machine/user data will be collected "
            + "(except the current runtime environment of the script like arguments and environment variables).")

    group.add_option("--no-color", action = "store_true", default = False,
        help = "Disable coloring of terminal output.")

    group.add_option("-V", dest = "script_version", action = "store_true", default = False,
        help = "Display script version and exit.")

    parser.add_option_group(group)


def process_common_options(options, args):

    global static_options, static_args
    static_options = options
    static_args = args

    if options.no_color:
        p4a_term.disabled = True

    set_verbosity(options.verbose)

    if options.script_version:
        main_script = os.path.abspath(sys.argv[0])
        version = guess_file_revision(main_script)
        sys.__stdout__.write(version + "\n")
        return False

    if options.log and not options.report:
        setup_logging()
    elif options.report:
        report = False
        setup_logging(suffix = "_report_" + utc_datetime(), remove = True)
        warn("--report enabled, will send an anonymous email with the program output in case of error")

    return True


def report_enabled():
    global static_options
    return static_options.report


def send_report_email(recipient = "par4all@hpc-project.com"):
    #~ if True:
    try:
        global static_options, static_args

        current_log_file = get_current_log_file()
        if not current_log_file or not os.path.exists(current_log_file):
            raise p4a_error("Current log file is invalid")

        warn("Trying to send an anonymous report email to " + recipient, log = False)
        flush_log()

        server = None
        for serv in [ "mail.hpc-project.com", "mail-int.hpc-project.com" ]:
            if ping(serv):
                server = serv
                break
        if not server:
            raise p4a_error("No SMTP server is reachable")

        is_root = "No"
        if whoami() == "root":
            is_root = "Yes"
        #~ options = ""
        #~ if static_options:
            #~ for o in static_options:
                #~ options += o + ": " + static_options[o] + "\n"
        #~ else:
            #~ options = "{}"
        args = ""
        if static_args:
            for a in static_args:
                args += a + "\n"
        else:
            args = "[]"
        env = ""
        for e in os.environ:
            env += e + ": " + os.environ[e] + "\n"

        msg = MIMEText("This is an automated report email for the following command which failed:\n\n" + " ".join(sys.argv)
            + "\n\nTranslated options:\n\n" + repr(static_options)
            + "\n\nTranslated arguments:\n\n" + args
            + "\nIs root? " + is_root
            + "\n\nEnvironment:\n\n" + env
            + "\nThe full log for this session follow:\n\n" + read_file(current_log_file))
        msg['Subject'] = "[" + get_program_name() + "] Anonymous failure report"
        msg['From'] = "anonymous@par4all.org"
        msg['To'] = recipient

        s = smtplib.SMTP(server)
        s.sendmail(msg['From'], [ msg['To'] ], msg.as_string())
        s.quit()

    except:
        warn("Sending the email failed -- Try sending " + current_log_file + " manually to " + recipient, log = False)

    else:
        done("Report email sent to " + recipient + ", thank you for your feedback", log = False)


def send_report_email_if_enabled():
    if report_enabled():
        send_report_email()
    else:
        error("You may report this error to the Par4All team by running again using --report", log = False)


def suggest_more_verbosity():
    if get_verbosity() < 3:
        v = "v" * (get_verbosity() + 1)
        error("To get more verbose output, try passing -" + v, log = False)


if __name__ == "__main__":
    print(__doc__)
    print("This module is not directly executable")


# Some Emacs stuff:
### Local Variables:
### mode: python
### mode: flyspell
### ispell-local-dictionary: "american"
### tab-width: 4
### End:
