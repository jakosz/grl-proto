FROM python:3.7.9
USER root

ADD app /app
ADD src /usr/src
ADD ../../../grl /usr/src/grl

RUN python3 -m pip install /usr/src/requirements.txt
RUN cd /usr/src/grl && python3 -m pip install -e .
RUN cd /usr/src/sparse2dense && python3 -m pip install -e .

RUN chmod +x /app/run.py
USER u
