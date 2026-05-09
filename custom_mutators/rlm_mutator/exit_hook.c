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

#include <errno.h>

/* Async-signal-safe: uses only open/write/close + snprintf of a small int. */
static void write_exit(int code) {
    if (s_written) return;
    const char *path = getenv("RLM_EXIT_FILE");

    int lfd = open("/tmp/exit_hook.log", O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (lfd >= 0) {
        char lbuf[256];
        int ln = snprintf(lbuf, sizeof(lbuf),
                          "write_exit pid=%d path=%s code=%d\n",
                          (int) getpid(), path ? path : "(null)", code);
        if (ln > 0) { ssize_t w = write(lfd, lbuf, (size_t) ln); (void) w; }
        close(lfd);
    }

    if (!path) return;
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        int e = errno;
        int elfd = open("/tmp/exit_hook.log", O_WRONLY | O_CREAT | O_APPEND, 0644);
        if (elfd >= 0) {
            char ebuf[256];
            int en = snprintf(ebuf, sizeof(ebuf),
                              "write_exit OPEN FAILED pid=%d path=%s errno=%d\n",
                              (int) getpid(), path, e);
            if (en > 0) { ssize_t w = write(elfd, ebuf, (size_t) en); (void) w; }
            close(elfd);
        }
        return;
    }
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
    int lfd = open("/tmp/exit_hook.log", O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (lfd >= 0) {
        char buf[128];
        int n = snprintf(buf, sizeof(buf),
                         "on_signal pid=%d sig=%d\n",
                         (int) getpid(), sig);
        if (n > 0) { ssize_t w = write(lfd, buf, (size_t) n); (void) w; }
        close(lfd);
    }
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
    int lfd = open("/tmp/exit_hook.log", O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (lfd >= 0) {
        char buf[128];
        int n = snprintf(buf, sizeof(buf),
                         "on_exit_cb pid=%d status=%d\n",
                         (int) getpid(), status);
        if (n > 0) { ssize_t w = write(lfd, buf, (size_t) n); (void) w; }
        close(lfd);
    }
    write_exit(status);
}

extern int on_exit(void (*)(int, void *), void *);

__attribute__((constructor))
static void exit_hook_init(void) {
    const char *path = getenv("RLM_EXIT_FILE");

    /* Diagnostic: AFL redirects child stderr, so log to a file we can inspect.
     * One line per child: pid + whether RLM_EXIT_FILE was visible. */
    int lfd = open("/tmp/exit_hook.log", O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (lfd >= 0) {
        char buf[256];
        int n = snprintf(buf, sizeof(buf), "pid=%d RLM_EXIT_FILE=%s\n",
                         (int) getpid(), path ? path : "(unset)");
        if (n > 0) { ssize_t w = write(lfd, buf, (size_t) n); (void) w; }
        close(lfd);
    }

    if (!path) {
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

    /* Redirect fd 2 to RLM_STDERR_FILE so the Python side can classify
     * Jerry's "Unhandled exception: SyntaxError | ReferenceError | ..."
     * messages and apply CovRL Eq. 2's 3-way validity reward instead of
     * the binary exit-code gate. Done last in the constructor so any
     * earlier WARNING write to fd 2 still reaches AFL's normal stderr. */
    const char *spath = getenv("RLM_STDERR_FILE");
    if (spath) {
        int sfd = open(spath, O_WRONLY | O_CREAT | O_TRUNC, 0600);
        if (sfd >= 0) {
            dup2(sfd, STDERR_FILENO);
            close(sfd);
        }
    }
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
