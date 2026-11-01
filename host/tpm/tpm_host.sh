#!/bin/bash
# Host side of TPM attestation, run inside the swtpm container.
#   serve                      manufacture a TPM (EK + EK cert), start it, wait
#   boot IMAGE_SHA256_HEX      measured boot: extend PCR 23 with the image measurement
#   pcrs                       read PCRs 0-7 and 23 into /work/pcrs.bin
#   enroll                     create EK and AK; write ek.crt, ak.pub, ak.name to /work
#   activate                   answer the verifier's challenge in /work/cred.bin
#   quote NONCE_HEX            quote PCRs 0-7 and 23 with the nonce
set -euo pipefail
W=/work
# No resource manager in front of swtpm: free transient objects and sessions
# so repeated commands don't exhaust the TPM's object slots.
flush() { tpm2_flushcontext -t >/dev/null 2>&1 || true; tpm2_flushcontext -s >/dev/null 2>&1 || true; }
PCRS=sha256:0,1,2,3,4,5,6,7,23

case "${1:-serve}" in
serve)
  mkdir -p /tpm/state
  swtpm_setup --tpm2 --tpmstate /tpm/state --create-ek-cert --create-platform-cert \
    --lock-nvram --overwrite >/tpm/setup.log 2>&1
  swtpm socket --tpm2 --tpmstate dir=/tpm/state --server type=tcp,port=2321 \
    --ctrl type=tcp,port=2322 --flags not-need-init,startup-clear -d
  echo "tpm ready"
  exec sleep infinity ;;
boot)
  tpm2_pcrextend "23:sha256=$2" ;;
pcrs)
  mkdir -p $W && tpm2_pcrread $PCRS -o $W/pcrs.bin >/dev/null ;;
enroll)
  mkdir -p $W && cd $W && flush
  tpm2_createek -c ek.ctx -G rsa -u ek.pub >/dev/null
  tpm2_createak -C ek.ctx -c ak.ctx -G rsa -g sha256 -s rsassa -u ak.pub -n ak.name >/dev/null
  tpm2_nvread 0x01c00002 -o ek.crt 2>/dev/null
  flush
  ls ek.crt ak.pub ak.name ;;
activate)
  cd $W && flush
  tpm2_startauthsession --policy-session -S session.ctx
  tpm2_policysecret -S session.ctx -c e >/dev/null
  tpm2_activatecredential -c ak.ctx -C ek.ctx -i cred.bin -o activated.secret -P "session:session.ctx"
  tpm2_flushcontext session.ctx
  flush
  od -An -tx1 activated.secret | tr -d " \n"; echo ;;
quote)
  cd $W && flush
  tpm2_quote -c ak.ctx -l $PCRS -q "$2" -m quote.msg -s quote.sig -f plain -g sha256 >/dev/null
  tpm2_pcrread $PCRS -o pcrs.bin >/dev/null
  flush
  ls quote.msg quote.sig pcrs.bin ;;
*)
  echo "unknown command $1" >&2; exit 2 ;;
esac
