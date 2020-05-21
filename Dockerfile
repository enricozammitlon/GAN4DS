FROM tensorflow/tensorflow:latest-gpu AS runner

COPY ./Training/deployment_requirements.txt /tmp/pip-tmp/

RUN apt-get update \
    && apt-get -y install --no-install-recommends apt-utils dialog 2>&1 \
    && apt-get -y install git iproute2 procps lsb-release python3-pip \
    && pip3 install --upgrade pip \
    && pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/deployment_requirements.txt \
    && rm -rf /tmp/pip-tmp \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 16006
CMD ["cd","Training"]
ENTRYPOINT ["python3","Gan4DS.py"]