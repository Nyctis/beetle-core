// Injected library: if the loader accepts it, scan the host process for the secret.
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <CommonCrypto/CommonDigest.h>

static void *scan(void *arg) {
    sleep(1);
    mach_vm_address_t addr = 0; mach_vm_size_t size; vm_region_basic_info_data_64_t info;
    mach_msg_type_number_t cnt; mach_port_t obj; const char *marker = "BEETLE-SECRET-";
    while (1) {
        cnt = VM_REGION_BASIC_INFO_COUNT_64;
        if (mach_vm_region(mach_task_self(), &addr, &size, VM_REGION_BASIC_INFO_64, (vm_region_info_t)&info, &cnt, &obj) != KERN_SUCCESS) break;
        if ((info.protection & VM_PROT_READ) && size < (256 << 20)) {
            char *p = memmem((void *)addr, size, marker, strlen(marker));
            if (p && p != marker) {
                unsigned char h[CC_SHA256_DIGEST_LENGTH]; CC_SHA256(p, 47, h);
                fprintf(stderr, "LEAKED-BY-INJECTION: sha256-prefix %02x%02x%02x%02x\n", h[0], h[1], h[2], h[3]);
                return NULL;
            }
        }
        addr += size;
    }
    return NULL;
}
__attribute__((constructor)) static void init(void) {
    fprintf(stderr, "INJECTED: library loaded into the process\n");
    pthread_t t; pthread_create(&t, NULL, scan, NULL); pthread_detach(t);
}
