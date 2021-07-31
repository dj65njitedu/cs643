FROM protonai/pyspark
WORKDIR /cs643
COPY requirements.txt requirements.txt
COPY model model
COPY predict.py predict.py
RUN pip3 install -r requirements.txt
ENTRYPOINT [ "python3", "predict.py"]