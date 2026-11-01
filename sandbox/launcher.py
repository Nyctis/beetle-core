"""Locks this process down, then runs the requester's program.

Started by program_runner.py as `python -I -S launcher.py WORKDIR ENTRY`, so
it is a fresh single-threaded process with no site-packages. Everything below
happens before the requester's code runs, and neither layer can be loosened
by the code that follows.

1. Resource limits: 2 GiB address space, 64 MiB files, 64 descriptors, no cores.
2. The program source is read and compiled, and the allowed standard-library
   modules plus the SDK are imported, while files can still be opened.
3. Landlock, if the kernel has it: filesystem access limited to WORKDIR and
   the Python runtime; no TCP; no signals to outside processes.
4. seccomp (always; the launcher refuses to run without it): the syscalls
   that open files, create sockets, start programs or processes, signal or
   trace other processes, read other processes' memory, or change mounts and
   namespaces fail with EPERM. The process keeps only the pipes it was given.

Any failure exits with code 97 before the program runs (fail closed).
Programs are therefore a single file that may use ALLOWED_MODULES and
beetle_sdk.
"""

import ctypes
import os
import platform
import resource
import struct
import sys

FAIL_CLOSED = 97
ALLOWED_MODULES = [
    "math", "random", "json", "re", "itertools", "collections", "functools", "statistics",
    "string", "time", "heapq", "bisect", "dataclasses", "typing", "textwrap", "unicodedata",
    "operator", "copy", "hashlib", "struct", "decimal", "fractions", "enum", "socket",
]

# --- Landlock -----------------------------------------------------------------
SYS_LANDLOCK_CREATE, SYS_LANDLOCK_ADD, SYS_LANDLOCK_RESTRICT = 444, 445, 446  # x86_64 and arm64
EXECUTE, WRITE_FILE, READ_FILE, READ_DIR = 1 << 0, 1 << 1, 1 << 2, 1 << 3
REMOVE_DIR, REMOVE_FILE, MAKE_DIR, MAKE_REG = 1 << 4, 1 << 5, 1 << 7, 1 << 8
REFER, TRUNCATE, IOCTL_DEV = 1 << 13, 1 << 14, 1 << 15
NET_BIND_TCP, NET_CONNECT_TCP = 1 << 0, 1 << 1
SCOPE_ABSTRACT_UNIX_SOCKET, SCOPE_SIGNAL = 1 << 0, 1 << 1

# --- seccomp ------------------------------------------------------------------
PR_SET_NO_NEW_PRIVS, PR_SET_SECCOMP, SECCOMP_MODE_FILTER = 38, 22, 2
BPF_LD_W_ABS, BPF_JEQ_K, BPF_JGE_K, BPF_RET_K = 0x20, 0x15, 0x35, 0x06
RET_ALLOW, RET_KILL_PROCESS = 0x7FFF0000, 0x80000000
RET_EPERM = 0x00050000 | 1
SECCOMP_DATA_NR, SECCOMP_DATA_ARCH = 0, 4

# name: (x86_64 number, arm64 number); None where the syscall doesn't exist.
BLOCKED_SYSCALLS = {
    "open": (2, None), "openat": (257, 56), "openat2": (437, 437), "creat": (85, None),
    "open_by_handle_at": (304, 265), "name_to_handle_at": (303, 264),
    "socket": (41, 198), "socketpair": (53, 199), "connect": (42, 203), "bind": (49, 200),
    "execve": (59, 221), "execveat": (322, 281),
    "fork": (57, None), "vfork": (58, None), "clone": (56, 220), "clone3": (435, 435),
    "kill": (62, 129), "tkill": (200, 130), "tgkill": (234, 131),
    "pidfd_open": (434, 434), "pidfd_getfd": (438, 438), "pidfd_send_signal": (424, 424),
    "ptrace": (101, 117), "process_vm_readv": (310, 270), "process_vm_writev": (311, 271),
    "kcmp": (312, 272), "userfaultfd": (323, 282), "perf_event_open": (298, 241),
    "bpf": (321, 280), "io_uring_setup": (425, 425), "keyctl": (250, 219),
    "add_key": (248, 217), "request_key": (249, 218),
    "mount": (165, 40), "umount2": (166, 39), "setns": (308, 268), "unshare": (272, 97),
    "chroot": (161, 51), "chdir": (80, 49), "fchdir": (81, 50),
    "mknod": (133, None), "mknodat": (259, 33), "mkdir": (83, None), "mkdirat": (258, 34),
    "unlink": (87, None), "unlinkat": (263, 35), "rmdir": (84, None),
    "rename": (82, None), "renameat": (264, None), "renameat2": (316, 276),
    "link": (86, None), "linkat": (265, 37), "symlink": (88, None), "symlinkat": (266, 36),
    "truncate": (76, 45),
}
ARCHES = {"x86_64": (0xC000003E, 0), "aarch64": (0xC00000B7, 1)}  # AUDIT_ARCH_*, column


def die(msg: str) -> None:
    sys.stderr.write(f"launcher: {msg}\n")
    sys.stderr.flush()
    os._exit(FAIL_CLOSED)


def _libc():
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    return libc


L = ctypes.c_long  # pass every integer to variadic syscall()/prctl() as a full 64-bit word


def apply_landlock(workdir: str) -> int:
    """Returns the Landlock ABI applied, or 0 if the kernel doesn't have Landlock."""
    libc = _libc()
    abi = libc.syscall(L(SYS_LANDLOCK_CREATE), None, L(0), L(1))  # 1 = query ABI version
    if abi < 1:
        return 0

    fs = (1 << 13) - 1  # every ABI 1 right
    if abi >= 2:
        fs |= REFER
    if abi >= 3:
        fs |= TRUNCATE
    if abi >= 5:
        fs |= IOCTL_DEV
    if abi >= 6:
        attr = struct.pack("<QQQ", fs, NET_BIND_TCP | NET_CONNECT_TCP,
                           SCOPE_ABSTRACT_UNIX_SOCKET | SCOPE_SIGNAL)
    elif abi >= 4:
        attr = struct.pack("<QQ", fs, NET_BIND_TCP | NET_CONNECT_TCP)
    else:
        attr = struct.pack("<Q", fs)
    buf = ctypes.create_string_buffer(attr, len(attr))
    ruleset = libc.syscall(L(SYS_LANDLOCK_CREATE), buf, L(len(attr)), L(0))
    if ruleset < 0:
        die(f"landlock_create_ruleset failed (errno {ctypes.get_errno()})")

    read_tree = EXECUTE | READ_FILE | READ_DIR
    write_tree = read_tree | WRITE_FILE | REMOVE_DIR | REMOVE_FILE | MAKE_DIR | MAKE_REG
    if abi >= 3:
        write_tree |= TRUNCATE
    rules = [("/usr", read_tree), ("/lib", read_tree), ("/etc", read_tree),
             ("/app/sdk", read_tree), (workdir, write_tree), ("/dev/null", READ_FILE | WRITE_FILE)]
    for path, access in rules:
        if not os.path.exists(path):
            continue
        fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
        rule = ctypes.create_string_buffer(struct.pack("<Qi", access & fs, fd), 12)
        rc = libc.syscall(L(SYS_LANDLOCK_ADD), L(ruleset), L(1), rule, L(0))  # 1 = path beneath
        os.close(fd)
        if rc != 0:
            die(f"landlock_add_rule {path} failed (errno {ctypes.get_errno()})")
    if libc.prctl(L(PR_SET_NO_NEW_PRIVS), L(1), L(0), L(0), L(0)) != 0:
        die("prctl(NO_NEW_PRIVS) failed")
    if libc.syscall(L(SYS_LANDLOCK_RESTRICT), L(ruleset), L(0)) != 0:
        die(f"landlock_restrict_self failed (errno {ctypes.get_errno()})")
    os.close(ruleset)
    return abi


class SockFilter(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint16), ("jt", ctypes.c_uint8),
                ("jf", ctypes.c_uint8), ("k", ctypes.c_uint32)]


class SockFprog(ctypes.Structure):
    _fields_ = [("len", ctypes.c_uint16), ("filter", ctypes.POINTER(SockFilter))]


def seccomp_program(machine: str) -> list[tuple[int, int, int, int]]:
    """BPF: kill on a foreign arch; EPERM for blocked syscalls; allow the rest."""
    if machine not in ARCHES:
        die(f"unsupported architecture {machine}")
    audit_arch, col = ARCHES[machine]
    numbers = sorted({nrs[col] for nrs in BLOCKED_SYSCALLS.values() if nrs[col] is not None})
    prog = [(BPF_LD_W_ABS, 0, 0, SECCOMP_DATA_ARCH),
            (BPF_JEQ_K, 1, 0, audit_arch),          # right arch: skip the kill
            (BPF_RET_K, 0, 0, RET_KILL_PROCESS),
            (BPF_LD_W_ABS, 0, 0, SECCOMP_DATA_NR)]
    if machine == "x86_64":                          # the x32 ABI would bypass the numbers
        prog += [(BPF_JGE_K, 0, 1, 0x40000000), (BPF_RET_K, 0, 0, RET_KILL_PROCESS)]
    n = len(numbers)
    for i, nr in enumerate(numbers):
        # match: jump over the remaining checks and the ALLOW, landing on EPERM
        prog.append((BPF_JEQ_K, n - i, 0, nr))
    prog += [(BPF_RET_K, 0, 0, RET_ALLOW), (BPF_RET_K, 0, 0, RET_EPERM)]
    return prog


def apply_seccomp() -> int:
    prog = seccomp_program(platform.machine())
    filters = (SockFilter * len(prog))(*[SockFilter(*ins) for ins in prog])
    fprog = SockFprog(len(prog), ctypes.cast(filters, ctypes.POINTER(SockFilter)))
    libc = _libc()
    if libc.prctl(L(PR_SET_NO_NEW_PRIVS), L(1), L(0), L(0), L(0)) != 0:
        die("prctl(NO_NEW_PRIVS) failed")
    if libc.prctl(L(PR_SET_SECCOMP), L(SECCOMP_MODE_FILTER), ctypes.byref(fprog), L(0), L(0)) != 0:
        die(f"seccomp filter failed (errno {ctypes.get_errno()})")
    return len(prog)


def main() -> None:
    if len(sys.argv) != 3:
        die("usage: launcher.py WORKDIR ENTRY")
    workdir, entry = os.path.realpath(sys.argv[1]), sys.argv[2]
    path = os.path.join(workdir, entry)
    if "/" in entry or not os.path.isfile(path):
        die("entry must be a file name inside WORKDIR")

    for limit, value in ((resource.RLIMIT_AS, 2 << 30), (resource.RLIMIT_FSIZE, 64 << 20),
                         (resource.RLIMIT_NOFILE, 64), (resource.RLIMIT_CORE, 0)):
        resource.setrlimit(limit, (value, value))

    with open(path, encoding="utf-8") as f:
        code = compile(f.read(), entry, "exec", dont_inherit=True)
    for name in ALLOWED_MODULES:
        __import__(name)
    sys.path[:] = ["/app/sdk"]
    import beetle_sdk  # noqa: F401  (loaded now; opening files is blocked later)

    abi = apply_landlock(workdir)
    n_rules = apply_seccomp()
    # First line of output, written before any requester code runs.
    print(f"launcher: seccomp on ({n_rules} rules), landlock {'abi ' + str(abi) if abi else 'unavailable'}",
          flush=True)
    sys.argv = [entry]
    exec(code, {"__name__": "__main__", "__builtins__": __builtins__})


if __name__ == "__main__":
    main()
