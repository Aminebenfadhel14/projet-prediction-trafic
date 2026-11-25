FROM spark:4.1.4-python3

USER root
RUN python3 -m pip install --no-cache-dir --target /opt/spark/python-deps numpy==2.2.6
ENV PYTHONPATH=/opt/spark/python-deps
USER spark
