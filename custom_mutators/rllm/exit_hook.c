/*
 * exit_hook.c — redirect child stderr to RLM_STDERR_FILE for validity
 * classification in rllm's post_run hook, and surface AFL timeouts.
 *
 * AFL++ muzzles the child's fd 2 (redirects it to /dev/null) so
 * JerryScript's "Unhandled exception: SyntaxError" messages go nowhere.
 * This library is LD_PRELOAD'd into the target via AFL_PRELOAD; its
 * constructor dup2's fd 2 to RLM_STDERR_FILE before the forkserver wait
 * loop starts, so every forked child inherits the redirect.
 *
 * O_APPEND is load-bearing: the constructor fires once in the forkserver
 * parent; all children share the same open-file-description (same offset).
 * Python's post_run truncates the file after each read. Without O_APPEND
 * the next child would seek to its inherited offset (past EOF) and produce
 * a sparse zero-padded file. With O_APPEND every write seeks to EOF first —
 * after a truncate that is offset 0 — giving clean per-execution content.
 *
 * Python must use os.truncate(path, 0), NOT os.unlink(). Unlinking would
 * orphan the forkserver's open fd in a ghost inode; subsequent child writes
 * would go there and Python's path-based open() would never see them.
 *
 * TIMEOUT marker: a hang produces no stderr, so it is indistinguishable from
 * a clean valid run by text alone. AFL decides the timeout in afl-fuzz and
 * kills the child with `child_kill_signal`; run_rllm.sh sets
 * AFL_KILL_SIGNAL=SIGUSR1 (catchable) so this library can catch that exact
 * moment and append TMOUT_MARK before re-raising the signal to die. AFL only
 * sends that signal on the timeout path, so the handler can't false-fire on
 * normal execs or crashes. rllm's classify_stderr maps the marker → "timeout"
 * (reward -1). Crashes (silent SIGSEGV) are left untouched.
 *
 * Compile:
 *   gcc -shared -fPIC -O2 -o exit_hook.so exit_hook.c
 *
 * Usage (run_rllm.sh sets these before afl-fuzz):
 *   export RLM_STDERR_FILE=<out>/rllm_stderr.txt
 *   export AFL_PRELOAD=/path/to/exit_hook.so
 *   export AFL_KILL_SIGNAL=10            # SIGUSR1, delivered to the child on timeout
 *   export AFL_FORK_SERVER_KILL_SIGNAL=9 # keep the forkserver on SIGKILL
 */

#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

/* NUL-wrapped so it can never collide with JerryScript's ASCII error text.
 * Must match data/validity.py:_TIMEOUT_MARK. */
static const char TMOUT_MARK[] = {0, 'R', 'L', 'M', '_', 'T', 'M', 'O', 'U', 'T', 0};

static void on_timeout_signal(int sig)
{
    /* async-signal-safe: raw write, then die by the same signal so AFL's
     * forkserver waitpid() returns (afl-fuzz already recorded FAULT_TMOUT). */
    (void)!write(STDERR_FILENO, TMOUT_MARK, sizeof(TMOUT_MARK));
    signal(sig, SIG_DFL);
    raise(sig);
}

__attribute__((constructor))
static void exit_hook_init(void)
{
    const char *spath = getenv("RLM_STDERR_FILE");
    if (!spath)
        return;

    int sfd = open(spath, O_WRONLY | O_CREAT | O_TRUNC | O_APPEND, 0600);
    if (sfd < 0)
        return;
    dup2(sfd, STDERR_FILENO);
    close(sfd);

    /* Catch AFL's timeout kill (AFL_KILL_SIGNAL=SIGUSR1). Only fires on the
     * timeout path; normal exits and crashes never deliver this signal. */
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = on_timeout_signal;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGUSR1, &sa, NULL);
}
