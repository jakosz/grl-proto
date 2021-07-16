FROM python:3.7.9
USER root

ADD app /app
ADD src /usr/src

RUN python3 -m pip install -r /usr/src/requirements.txt
RUN cd /usr/src/grl && python3 -m pip install -e .

RUN chmod +x /app/ENTRYPOINT
USER u
ENTRYPOINT ["/app/ENTRYPOINT"]
