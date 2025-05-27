FROM nvcr.io/nvidia/pytorch:25.03-py3
WORKDIR /usr/local/app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src src
COPY scripts scripts

RUN chmod 755 scripts/run.sh

ENTRYPOINT ["./scripts/run.sh"]