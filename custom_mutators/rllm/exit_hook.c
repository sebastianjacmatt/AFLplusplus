/*
 * exit_hook.c — redirect child stderr to RLM_STDERR_FILE for validity
 * classification in rllm's post_run hook.
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
 * Compile:
 *   gcc -shared -fPIC -O2 -o exit_hook.so exit_hook.c
 *
 * Usage (run_rllm.sh sets these before afl-fuzz):
 *   export RLM_STDERR_FILE=<out>/rllm_stderr.txt
 *   export AFL_PRELOAD=/path/to/exit_hook.so
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
}
