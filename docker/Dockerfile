FROM python:3.10.12
WORKDIR /app/
COPY ./docker/requirements.txt /app/
RUN python3 -m pip install -r requirements.txt
CMD ["python3", "bot.py"]
