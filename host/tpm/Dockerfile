# A software TPM (swtpm) plus tpm2-tools, standing in for a contributor PC's
# TPM 2.0 chip in tests. Same protocol and formats as a hardware TPM; its EK
# certificate is signed by a local test CA, so it proves nothing about hardware.
#   docker build -t beetle-swtpm:dev -f host/tpm/Dockerfile host/tpm
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends swtpm swtpm-tools tpm2-tools \
    ca-certificates && rm -rf /var/lib/apt/lists/*
COPY tpm_host.sh /usr/local/bin/tpm_host.sh
ENV TPM2TOOLS_TCTI="swtpm:host=127.0.0.1,port=2321"
ENTRYPOINT ["/usr/local/bin/tpm_host.sh"]
CMD ["serve"]
