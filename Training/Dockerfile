FROM tensorflow/tensorflow:2.1.0-gpu-py3 AS runner
COPY . .
COPY ./deployment_requirements.txt /tmp/pip-tmp/
RUN apt-get update -qq && apt-get -y install \
   && apt-get install -y ffmpeg \
   && pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/deployment_requirements.txt
ENV ADDR=0.0.0.0
EXPOSE 6006
# Use  -L 16006:localhost:6006 with ssh to be able to see tensorboard in your browser
ENTRYPOINT ["python3","-u","Gan4DS.py"]
