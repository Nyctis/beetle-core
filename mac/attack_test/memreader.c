// Attacker: get the target's task port and scan its memory for the secret.
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CommonCrypto/CommonDigest.h>

int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: memreader PID\n"); return 2; }
    pid_t pid = atoi(argv[1]);
    mach_port_t task;
    kern_return_t kr = task_for_pid(mach_task_self(), pid, &task);
    if (kr != KERN_SUCCESS) { printf("BLOCKED: task_for_pid failed: %s (%d)\n", mach_error_string(kr), kr); return 1; }
    mach_vm_address_t addr = 0; mach_vm_size_t size; vm_region_basic_info_data_64_t info;
    mach_msg_type_number_t cnt; mach_port_t obj;
    const char *marker = "BEETLE-SECRET-";
    while (1) {
        cnt = VM_REGION_BASIC_INFO_COUNT_64;
        if (mach_vm_region(task, &addr, &size, VM_REGION_BASIC_INFO_64, (vm_region_info_t)&info, &cnt, &obj) != KERN_SUCCESS) break;
        if ((info.protection & VM_PROT_READ) && size < (256 << 20)) {
            char *buf = malloc(size); mach_vm_size_t got = 0;
            if (mach_vm_read_overwrite(task, addr, size, (mach_vm_address_t)buf, &got) == KERN_SUCCESS) {
                char *p = memmem(buf, got, marker, strlen(marker));
                if (p && p + 47 <= buf + got) {
                    unsigned char h[CC_SHA256_DIGEST_LENGTH]; CC_SHA256(p, 47, h);
                    printf("LEAKED: read the secret from memory, sha256-prefix %02x%02x%02x%02x\n", h[0], h[1], h[2], h[3]);
                    return 0;
                }
            }
            free(buf);
        }
        addr += size;
    }
    printf("got the task port but did not find the secret\n");
    return 3;
}
