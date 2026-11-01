// Holds a random secret in heap memory, like the T1 Mac worker holds a job's
// decrypted data. Prints its pid and the secret's fingerprint, then waits.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <CommonCrypto/CommonDigest.h>

int main(void) {
    unsigned char rnd[16];
    arc4random_buf(rnd, sizeof rnd);
    char *secret = malloc(64);
    int n = snprintf(secret, 64, "BEETLE-SECRET-");
    for (int i = 0; i < 16; i++) n += snprintf(secret + n, 64 - n, "%02x", rnd[i]);
    unsigned char h[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(secret, (CC_LONG)strlen(secret), h);
    printf("pid %d secret-sha256-prefix %02x%02x%02x%02x\n", getpid(), h[0], h[1], h[2], h[3]);
    fflush(stdout);
    for (;;) { sleep(1); __asm__ volatile("" :: "r"(secret) : "memory"); }
}
