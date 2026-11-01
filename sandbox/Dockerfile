# Local mock of the T2 sealed environment. Build from the beetle root:
#   docker build -t beetle-sandbox:dev -f sandbox/Dockerfile .
# GPU hosts add --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu128
FROM python:3.12-slim@sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/tmp/hf \
    HOME=/tmp

COPY sandbox/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir "torch==2.11.0" --index-url "$TORCH_INDEX" \
 && pip install --no-cache-dir --no-deps --require-hashes -r /tmp/requirements.txt \
 && pip freeze > /opt/freeze.txt \
 && useradd --uid 10001 --no-create-home enclave

COPY common/beetle_crypto.py common/beetle_archive.py common/job_defaults.py \
     sandbox/enclave.py sandbox/train_eval.py sandbox/recipe_data.py \
     sandbox/program_runner.py sandbox/launcher.py sandbox/requirements.txt /app/
COPY sandbox/sdk/beetle_sdk.py /app/sdk/
WORKDIR /app
USER 10001
ENTRYPOINT ["python", "/app/enclave.py"]
