FROM python:3.13-slim
RUN pip install --no-cache-dir numpy==2.3.4
WORKDIR /bot
COPY bot.py mechfight.py world.py ./
ENTRYPOINT ["python3", "-u", "/bot/bot.py"]
