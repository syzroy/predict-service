FROM python:3-slim

COPY src/ .

RUN pip install tensorflow pillow flask

CMD python "app.py"
