/*

  $Id$

  Copyright 1989-2010 MINES ParisTech

  This file is part of PIPS.

  PIPS is free software: you can redistribute it and/or modify it
  under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  any later version.

  PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.

  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with PIPS.  If not, see <http://www.gnu.org/licenses/>.

*/
/*
 * Directly from the old implementation of pipsdbm.
 * Too many static buffers in my opinion;-) FC.
 */
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>

#include "genC.h"
#include "misc.h"

#include "database.h"
#include "makefile.h"
#include "linear.h"
#include "ri.h"

#include "newgen.h" /* statement_mapping */
#include "pipsdbm.h"

/* pipsmake interruption 
 */
static bool flag_interrupt_pipsmake_asap = FALSE;

void interrupt_pipsmake_asap()
{
    flag_interrupt_pipsmake_asap = TRUE;
}

void dont_interrupt_pipsmake_asap()
{
    flag_interrupt_pipsmake_asap = FALSE;
}

bool interrupt_pipsmake_asap_p()
{
    bool res = flag_interrupt_pipsmake_asap;
    flag_interrupt_pipsmake_asap = FALSE;
    return res;
}

extern int gettimeofday();

/* Timing of one request */
static struct tms request_time;
static struct tms request_phase_time;
static struct tms request_dbm_time;
static double real_request_time;
static double real_request_phase_time;
static double real_request_dbm_time;

/* Timing of one phase */
static struct tms phase_time;
static struct tms dbm_time;
static struct tms total_dbm_time;
static double real_phase_time;
static double real_dbm_time;
static double real_total_dbm_time;


/* Get real time in seconds in a double representation */
static void get_real_timer(double *d)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    *d = (double)tv.tv_sec + (double)tv.tv_usec/1.0e6;
}

/* Functions for timing one request
 */
void init_request_timers()
{
    /* save initial CPU use times */
    times (&request_time);
    /* save initial real time */
    get_real_timer (&real_request_time);

    /* initialize accumulators for dbm and phase times */
    request_dbm_time.tms_utime = 0;
    request_dbm_time.tms_stime = 0;
    real_request_dbm_time = 0.0;

    request_phase_time.tms_utime = 0;
    request_phase_time.tms_stime = 0;
    real_request_phase_time = 0.0;
}

/* Functions for timing one phase 
 */
void init_log_timers()
{
    times (&phase_time);
    get_real_timer (&real_phase_time);
    total_dbm_time.tms_utime = 0;
    total_dbm_time.tms_stime = 0;
    real_total_dbm_time = 0.0;
}

void dbm_start_timer()
{
    times (&dbm_time);
    get_real_timer(&real_dbm_time);
}

/* accumulate dbm related times for one phase */
void dbm_stop_timer()
{
    struct tms current;
    double real_current;

    times (&current);
    total_dbm_time.tms_utime += current.tms_utime - dbm_time.tms_utime;
    total_dbm_time.tms_stime += current.tms_stime - dbm_time.tms_stime;
    get_real_timer(&real_current);
    real_total_dbm_time += real_current - real_dbm_time;
}

#define MAX_TIME_STRING_LENGTH 128
#define HZ 60

/* compute times elapsed since init_log_timers(), i.e. for one phase */
void get_string_timers(string *with_io, string *io)
{
    struct tms total_phase_time;
    double real_total_phase_time;
    static char s1[MAX_TIME_STRING_LENGTH];
    static char s2[MAX_TIME_STRING_LENGTH];

    times (&total_phase_time);
    total_phase_time.tms_utime -= phase_time.tms_utime;
    total_phase_time.tms_stime -= phase_time.tms_stime;
    get_real_timer(&real_total_phase_time);
    real_total_phase_time -= real_phase_time;

    sprintf (s1,"(r%.3f u%.2f s%.2f)\n",
	     real_total_phase_time,
	     (double) total_phase_time.tms_utime / HZ,
	     (double) total_phase_time.tms_stime / HZ);

    sprintf (s2,"(r%.3f u%.2f s%.2f)\n",
	     real_total_dbm_time,
	     (double) total_dbm_time.tms_utime / HZ,
	     (double) total_dbm_time.tms_stime / HZ);

    *with_io = s1; *io = s2;

    /* accumulate phase times in request times */
    request_dbm_time.tms_utime += total_dbm_time.tms_utime;
    request_dbm_time.tms_stime += total_dbm_time.tms_stime;
    real_request_dbm_time +=real_total_dbm_time;

    request_phase_time.tms_utime += total_phase_time.tms_utime;
    request_phase_time.tms_stime += total_phase_time.tms_stime;
    real_request_phase_time += real_total_phase_time;
}

/* compute times elapsed since init_request_log_timers(),
 * i.e. for one request to pipsmake (btw this code is misplaced...)
 */
void get_request_string_timers(string *global, string *phases, string *dbm)
{
    struct tms total_request_time;
    double real_total_request_time;
    static char s1[MAX_TIME_STRING_LENGTH];
    static char s2[MAX_TIME_STRING_LENGTH];
    static char s3[MAX_TIME_STRING_LENGTH];

    times (&total_request_time);
    total_request_time.tms_utime -= request_time.tms_utime;
    total_request_time.tms_stime -= request_time.tms_stime;
    get_real_timer(&real_total_request_time);
    real_total_request_time -= real_request_time;

    sprintf (s1,"(r%.3f u%.2f s%.2f)\n",
	     real_total_request_time,
	     (double) total_request_time.tms_utime / HZ,
	     (double) total_request_time.tms_stime / HZ);

    sprintf (s2,"(r%.3f u%.2f s%.2f)\n",
	     real_request_phase_time,
	     (double) request_phase_time.tms_utime / HZ,
	     (double) request_phase_time.tms_stime / HZ);

    sprintf (s3,"(r%.3f u%.2f s%.2f)\n",
	     real_request_dbm_time,
	     (double) request_dbm_time.tms_utime / HZ,
	     (double) request_dbm_time.tms_stime / HZ);

    *global = s1;
    *phases = s2;
    *dbm = s3;
}

/**************************************************************** OBSOLETE? */

/* Sets of the readwrite resources by pipsdbm */
static set res_read = set_undefined;
static set res_write = set_undefined;

/* init variables */
void init_resource_usage_check()
{
    if (!set_undefined_p(res_read))
	set_clear (res_read);
    res_read = set_make (set_string);
    if (!set_undefined_p(res_write))
	set_clear (res_write);
    res_write = set_make (set_string);
}

/* add an element to the read set */
void add_read_resource(string rname,string oname)
{
    if (!set_undefined_p(res_read))
	set_add_element(res_read,
			res_read,
			strdup(concatenate(oname, ".", rname, NULL)));
}

/* add an element to the write set */
void add_write_resource(string rname,string oname)
{
    if (!set_undefined_p(res_write))
    set_add_element(res_write, res_write,
		    strdup(concatenate(oname, ".", rname, NULL)));
}

/* Get the made sets */
void get_logged_resources(set *sr, set *sw)
{
    *sr = res_read;
    *sw = res_write;
}
