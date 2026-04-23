/*
 * exit_hook.c — report child termination status to rlm.py via a tmpfile.
 *
 * AFL++'s post_run() Python hook receives zero arguments; the child's exit
 * status is captured by AFL++'s waitpid() but never forwarded to Python.
 * This library records the termination status in every forked child and
 * writes it to RLM_EXIT_FILE before the process dies.  rlm.py::post_run()
 * then reads that file — AFL++ is synchronous: the child has fully exited
 * before post_run() is called, so there is no race.
 *
 * Coverage:
 *   - exit(code) / return from main()   → code
 *   - _exit(code) / _Exit(code)          → code
 *   - SIGSEGV / SIGABRT / SIGBUS / ...   → 128 + signo (shell convention)
 *   - SIGKILL (AFL timeout)              → uncatchable, file stays absent
 *
 * Compile:
 *   gcc -shared -fPIC -O2 -o exit_hook.so exit_hook.c -ldl
 *
 * Usage (RLM_EXIT_FILE is exported by run_afl.sh before afl-fuzz):
 *   AFL_PRELOAD=/path/to/exit_hook.so  afl-fuzz -i seeds -o out -- ./target @@
 */

#define _GNU_SOURCE
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

static int s_written = 0;

/* Async-signal-safe: uses only open/write/close + snprintf of a small int. */
static void write_exit(int code) {
    if (s_written) return;
    const char *path = getenv("RLM_EXIT_FILE");
    if (!path) return;
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return;
    char buf[32];
    int n = snprintf(buf, sizeof(buf), "%d\n", code);
    if (n > 0) {
        ssize_t w = write(fd, buf, (size_t) n);
        (void) w;
    }
    close(fd);
    s_written = 1;
}

static void on_signal(int sig) {
    write_exit(128 + sig);
    /* Reset to default and re-raise so AFL's waitpid sees the real signal. */
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = SIG_DFL;
    sigaction(sig, &sa, NULL);
    raise(sig);
}

/* glibc's on_exit callback: invoked from within the exit() path, so this
 * fires even when libc calls its internal __GI_exit alias (which is not
 * interposable by LD_PRELOAD). Receives the exit status as first arg. */
static void on_exit_cb(int status, void *arg) {
    (void) arg;
    write_exit(status);
}

extern int on_exit(void (*)(int, void *), void *);

__attribute__((constructor))
static void exit_hook_init(void) {
    if (!getenv("RLM_EXIT_FILE")) {
        const char msg[] =
            "[exit_hook] WARNING: RLM_EXIT_FILE not set — exit codes will not be recorded\n";
        ssize_t w = write(2, msg, sizeof(msg) - 1);
        (void) w;
    }

    on_exit(on_exit_cb, NULL);

    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = on_signal;
    sa.sa_flags   = 0;
    sigaction(SIGSEGV, &sa, NULL);
    sigaction(SIGABRT, &sa, NULL);
    sigaction(SIGBUS,  &sa, NULL);
    sigaction(SIGFPE,  &sa, NULL);
    sigaction(SIGILL,  &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
    sigaction(SIGPIPE, &sa, NULL);
}

void exit(int code) {
    write_exit(code);
    syscall(SYS_exit_group, code);
    __builtin_unreachable();
}

void _exit(int code) {
    write_exit(code);
    syscall(SYS_exit_group, code);
    __builtin_unreachable();
}

void _Exit(int code) {
    write_exit(code);
    syscall(SYS_exit_group, code);
    __builtin_unreachable();
}
