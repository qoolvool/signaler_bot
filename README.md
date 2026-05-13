# Signaler

Поиск уровней поддержки/сопротивления на MEXC и отправка сигналов в Telegram.

## Установка

```bash
git clone <your-repo-url>
cd signaler

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Настройка

```bash
cp .env.example .env
```

Открой `.env` и заполни:

- `MEXC_API_KEY` / `MEXC_SECRET_KEY` — получить на https://www.mexc.com/user/openapi
- `TELEGRAM_BOT_TOKEN` — токен бота от [@BotFather](https://t.me/BotFather)
- `TELEGRAM_CHAT_ID` — твой chat id (узнать у [@userinfobot](https://t.me/userinfobot))

Опционально меняй параметры анализа (`TRADING_PAIRS`, `TIMEFRAME`, `MIN_TOUCHES` и т.д.).

## Запуск

```bash
python3 signaler.py
```

## Безопасность

- Файл `.env` **никогда** не коммитится — он в `.gitignore`.
- Если ты случайно закоммитил `.env` или ключи — **немедленно**:
  1. Отозви ключи на MEXC и пересоздай Telegram-бота через `/revoke` у [@BotFather](https://t.me/BotFather).
  2. Удали их из истории git (`git filter-repo` или BFG Repo-Cleaner).
- Перед каждым `git push` проверь `git status` — `.env` не должен появляться в списке.
- На MEXC создавай API-ключ **только** с правами на чтение, без вывода средств.
