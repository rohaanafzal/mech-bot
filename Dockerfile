FROM python:3.13-slim
RUN pip install --no-cache-dir numpy==2.3.4
WORKDIR /bot
COPY bot.py practice_brain.py ffa_brain.py world.py mechfight.py nav.bin roofs.json ./
ENTRYPOINT ["python3", "-u", "/bot/bot.py"]
