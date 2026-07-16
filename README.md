# VR AI Assistant (локальный гибридный RAG + OpenAI)

ИИ-помощник для школьного VR-тренажёра (физика, химия, биология). Тонкий
**FastAPI-прокси** между VR-клиентом, локальным ретривал-стеком и OpenAI. Для
follow-up вопросов `/ask` и `/voice_ask` держат ограниченную TTL-память диалога
в процессе API:

- **База знаний локальная** (self-hosted **Qdrant**). Гибридный поиск: плотный
  вектор (`dense`, 1024-d, cosine) + разреженный (`sparse`), две ветки `Prefetch`
  в Qdrant Query API, слитые через **RRF** (Reciprocal Rank Fusion).
- **Эмбеддинги:** локальный GPU-сайдкар `embedder` на модели **BAAI/bge-m3**
  (мультиязычная, сильна на русском и казахском; FlagEmbedding `BGEM3FlagModel`,
  отдаёт dense + learned-sparse). Рассчитан на NVIDIA RTX 3060 12GB (Ampere sm_86).
- **Генерация:** OpenAI **Responses API**. RAG явный: *retrieve → inject →
  generate*. Найденные чанки подставляются в системный промт (а не hosted
  `file_search`). Ссылки на источники собираются из метаданных чанков (ТЗ §4).
- **Контекст сценария** (текущая сцена, шаги, объекты) подставляется в системный
  промт по `scenario_id` (ТЗ §3.2).
- **Голос**: STT и TTS обслуживает встроенный GPU-сайдкар `voice` (Whisper
  ru/kk/auto + Supertonic по умолчанию / Qwen3-TTS 0.6B по явному выбору на русском,
  отдельный OmniVoice с фиксированным молодым мужским голосом на казахском и
  MMS как fallback, каталоги `./voice` и `./voice_omnivoice`). Перед синтезом
  числа детерминированно раскрываются словами на языке соответствующего сайдкара;
  полный конвейер
  `/voice_ask` под целевую задержку ≤5 c (ТЗ §5, §7).

Прокси держит секретный ключ OpenAI у себя. VR-клиент аутентифицируется только
коротким `INTERNAL_API_KEY`, а реальный ключ OpenAI в приложение не попадает.

## Архитектура

```text
VR-клиент (Unity)
   │  Authorization: Bearer <INTERNAL_API_KEY>
   ▼
FastAPI-прокси (этот репозиторий)            ┌─ scenarios/*.json  (контекст сцен)
   │  http://api:8000 (хост :8001)            └─ инъекция в system prompt
   ├─► embedder (bge-m3, GPU)        POST http://embedder:8080/embed
   │       └─ dense + sparse эмбеддинг запроса
   ├─► Qdrant                        http://qdrant:6333 (Query API + RRF)
   │       └─ гибридный поиск (dense + sparse, RRF) → top-k чанки → system prompt
   ├─► voice-сайдкар (GPU)           POST http://voice:8001/stt/recognize
   │       ├─ Whisper (ru/kk/auto)   POST http://voice:8001/tts/synthesize?format=wav
   │       ├─ Supertonic / Qwen / MMS (ru)
   │       └─ MMS fallback (kk)
   ├─► voice-omnivoice (GPU)         POST http://voice-omnivoice:8003/tts/synthesize
   │       └─ OmniVoice fixed young-male default (kk)
   ▼
OpenAI: Responses API (генерация, ключ OpenAI)
```

Хосты вида `http://embedder:8080` являются именами сервисов внутри docker-compose сети.
Для локального запуска прокси вне Docker используйте `localhost` и host-mapped
порты (см. таблицу «Подключения и порты» ниже).

Документы (PDF/DOCX/EPUB/TXT/MD) парсятся локально, нарезаются на чанки,
эмбеддятся через `embedder` и кладутся в Qdrant. Всё выполняется on-prem. Ретривал
изолирован за `app/services/embeddings.py` (HTTP-клиент к эмбеддеру) и
`app/services/vectorstore.py` (обёртка над Qdrant Query API + RRF); генерация
осталась в облаке OpenAI. Форма ответа для VR-клиента (`citations`,
`primary_source`) не изменилась. Ссылки пересобираются из метаданных чанков.

## Быстрый старт

Проще всего поднять весь стек через Docker:

```bash
cp .env.example .env          # заполнить INTERNAL_API_KEY и OPENAI_API_KEY
docker compose up --build     # 5 сервисов (порты ниже)
```

`docker compose` поднимает **пять** сервисов:

| Сервис | Образ / build | Порт (хост → контейнер) | GPU |
|--------|---------------|--------------------------|-----|
| `api` | `.` | `8001 → 8000` | нет |
| `qdrant` | `qdrant/qdrant:latest` | `6333 → 6333`, `6334 → 6334` | нет |
| `embedder` | `./embedder` | `8080 → 8080` | да |
| `voice` | `./voice` | `8002 → 8001` | да |
| `voice-omnivoice` | `./voice_omnivoice` | `8003 → 8003` | да |

`embedder`, `voice` и `voice-omnivoice` являются GPU-сайдкарами, требуют
**NVIDIA Container Toolkit** и делят одну карту; первый запуск скачивает модели
(bge-m3, Whisper, TTS), поэтому
стартует медленно.

Локальная разработка прокси (Qdrant и embedder при этом удобно держать в Docker):

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # INTERNAL_API_KEY, OPENAI_API_KEY,
                              # QDRANT_URL, EMBEDDING_BASE_URL и т.д.

# 1) создать коллекцию в Qdrant (dense + sparse именованные векторы)
python -m scripts.manage_corpus create-collection

# 2) загрузить отдельные общие документы без предметных метаданных
python -m scripts.manage_corpus upload materials/physics_8.pdf
python -m scripts.manage_corpus upload materials/chem_9.docx
python -m scripts.manage_corpus list
python -m scripts.manage_corpus status

# 3) запустить сервис
uvicorn app.main:app --reload --port 8000
# docs: http://localhost:8000/docs
```

Удалить документ из коллекции: `python -m scripts.manage_corpus delete <doc_id>`.

Для полной или массовой загрузки учебников и лабораторных инструкций
предпочтителен структурированный `bulk-ingest`, описанный ниже. Отдельный
тегированный документ можно загрузить через backend API `POST /admin/documents`,
явно передав его предметные метаданные. CLI-команда `upload` по-прежнему
загружает общий документ без предметных метаданных.

## Загрузка учебников, лабораторных инструкций и сценариев

Система использует три независимых источника контекста:

| Источник | Где хранится | Как добавляется |
|----------|--------------|-----------------|
| Учебники и теория | Qdrant, `doc_type=textbook` | `POST /admin/documents` для одного файла или `manage_corpus.py bulk-ingest` для целого корпуса |
| Инструкции к лабораторным работам | Qdrant, `doc_type=lab_instruction` | `POST /admin/documents` для одного файла или `bulk-ingest` из правильной структуры каталогов |
| Статическая логика VR-сцены | JSON в `scenarios/` | добавить или изменить файл вручную |

`scenario_state`, который тренажёр передаёт в каждом запросе, является четвёртым
источником. Это актуальное состояние сцены, а не загружаемый документ. При
расхождении со статическим JSON оно имеет приоритет.

### 1. Подготовить структуру корпуса

`bulk-ingest` определяет метаданные из пути. Поддерживаются файлы `.pdf`, `.docx`,
`.epub`, `.txt` и `.md`. Минимальная структура выглядит так:

```text
Лабораторные физхимбио/
└── Материалы лабок/
    ├── Школьный материал 7-11 класс 3 предмета/
    │   └── Физика/
    │       └── рус/
    │           └── Физика 8 класс.pdf
    └── Лабораторные работы/
        └── Физика/
            └── Физика 10 класс/
                └── рус/
                    └── Лабораторная работа №2.docx
```

Правила распознавания путей:

- каталог типа документа должен начинаться с `Школьный материал` или называться
  `Лабораторные работы`;
- предмет должен быть `Физика`, `Химия` или `Биология`;
- для лабораторной инструкции класс берётся из каталога вида
  `Физика 10 класс`, а для учебника из имени файла;
- язык определяется по каталогу `рус`, `русс`, `каз` или `қаз`;
- номер лабораторной работы берётся из первого номера в имени файла, например
  `Лабораторная работа №2.docx` или `Зертханалық жұмыс № 2.docx`.

Из этих значений формируется стабильный `lab_id`. Например, путь выше получает
`physics-10-ru-02`. Тренажёр должен передать те же значения в объекте `lab`.

Если корпус находится не в каталоге по умолчанию, задайте путь в `.env`:

```dotenv
CORPUS_ROOT=/absolute/path/to/Лабораторные физхимбио
LABS_MANIFEST=./labs.json
```

### 2. Проверить лабораторные работы до загрузки

Команда не обращается к Qdrant или embedder. Она проверяет пути, извлекает текст
и создаёт отчёт `labs.json`:

Перед запуском создайте `.env` из `.env.example` и задайте непустой
`INTERNAL_API_KEY`. Corpus CLI использует общие настройки приложения, хотя для
`gen-manifest` и `bulk-ingest` ключ OpenAI не применяется.

```bash
python -m scripts.manage_corpus gen-manifest
```

В отчёте:

- `complete` означает, что инструкция распознана и содержит достаточно текста;
- `stub` означает, что файл найден, но извлечённого текста слишком мало;
- `missing_metadata` содержит пути, для которых не удалось определить обязательные
  метаданные.

Исправьте `stub` и `missing_metadata` до основной загрузки. Лабораторная работа
без инструкции в Qdrant считается неполной: ассистент отвечает только по теории
и не придумывает последовательность действий.

### 3. Загрузить весь корпус

Qdrant и embedder должны быть доступны по `QDRANT_URL` и
`EMBEDDING_BASE_URL`:

```bash
docker compose up -d qdrant embedder
python -m scripts.manage_corpus create-collection
python -m scripts.manage_corpus bulk-ingest
```

Повторная загрузка безопасно заменяет чанки документа с тем же относительным
путём. Для загрузки только части дерева используйте `--only`, не меняя корневой
каталог. Это сохраняет стабильные идентификаторы документов:

```bash
python -m scripts.manage_corpus bulk-ingest --only 'Биология/рус'
```

Для сканированных PDF или EPUB включите OCR. На машине, где запускается команда,
нужны Tesseract и языковые модели `rus`/`kaz`. Они включены в Docker-образ API,
но для запуска CLI непосредственно на хосте их нужно установить отдельно:

```bash
python -m scripts.manage_corpus bulk-ingest --ocr --only 'Биология/рус'
```

При обработке PDF загрузчик удаляет известные водяные знаки и служебные строки
OKULYK, а также таблицы из одних номеров страниц. Если после очистки текстовый
слой остаётся чрезмерно повторяющимся или малоинформативным, `--ocr` запускает
Tesseract даже при формально большом количестве распознанных кириллических слов.

### 4. Проверить результат

```bash
python -m scripts.manage_corpus status
python -m scripts.manage_corpus list
python -m scripts.manage_corpus gen-manifest
```

После загрузки проверьте тестовый запрос с тем же контекстом, который отправляет
тренажёр:

```bash
curl -s localhost:8000/ask \
  -H "Authorization: Bearer $KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "Что делать дальше?",
    "scenario_id": "physics_lab_02_heating",
    "lab": {
      "subject": "physics",
      "grade": 10,
      "lang": "ru",
      "lab_number": 2
    },
    "scenario_state": {
      "current_step_id": "heat-water",
      "current_step_index": 3,
      "current_step": "Нагреть воду",
      "next_step": "Записать температуру",
      "held_items": ["термометр"],
      "visible_items": ["стакан", "штатив"],
      "allowed_actions": ["включить нагрев"]
    }
  }'
```

Ответ должен содержать цитату на инструкцию с `lab_id=physics-10-ru-02` и, если
понадобилась теория, цитаты на учебник соответствующего предмета и класса.

### 5. Добавить статический сценарий VR

Сценарий не загружается в Qdrant. Создайте JSON-файл, имя которого совпадает с
`scenario_id`:

```text
scenarios/physics_lab_02_heating.json
```

Используйте [существующий пример](scenarios/physics_lab_02_heating.json) как
шаблон. При запуске через Docker каталог `scenarios/` смонтирован read-only, но
изменения файлов подхватываются без пересборки контейнера.

Проверить список доступных сценариев можно через API:

```bash
curl -s localhost:8000/admin/scenarios \
  -H "Authorization: Bearer $KEY"
```

Сценарии через API не загружаются. Учебники, лабораторные инструкции и общие
документы можно загружать по одному через backend API. Для синхронизации всего
корпуса по структуре каталогов используйте `bulk-ingest`.

### Одиночная загрузка через backend API

`POST /admin/documents` принимает `multipart/form-data` со следующими полями:

| Поле | Значения | Назначение |
|------|----------|------------|
| `file` | PDF, DOCX, EPUB, TXT или MD | обязательный загружаемый файл |
| `doc_type` | `textbook` или `lab_instruction` | необязательный тип структурированного документа |
| `subject` | `physics`, `chemistry` или `biology` | предмет структурированного документа |
| `grade` | целое число от 7 до 11 | класс структурированного документа |
| `lang` | `ru` или `kk` | язык структурированного документа |
| `lab_number` | целое число от 1 до 99 | обязателен при `doc_type=lab_instruction` |
| `ocr` | `true` или `false` | необязательно включить OCR, по умолчанию `false` |

Все поля метаданных необязательны для общего документа. Если передано хотя бы
одно из них, требуется `doc_type`; для обоих структурированных типов также
требуются `subject`, `grade` и `lang`. Учебник не принимает `lab_number`.

Если отправить только `file`, документ останется общим и будет участвовать в
поиске без структурированного `lab`-контекста. При передаче `doc_type`,
`subject`, `grade` и `lang` backend сохраняет детерминированные метаданные. Для
лабораторной инструкции он также строит стабильный `lab_id`, например
`physics-10-ru-02`. Повторная загрузка файла с теми же метаданными и тем же
именем заменяет существующий документ и его чанки, а не создаёт дубль.

Пример загрузки учебника на развёрнутый backend:

```bash
API_BASE=https://megroup-b560m-hdv-m-2.tail7dd37a.ts.net

curl -sS -X POST "$API_BASE/admin/documents" \
  -H "Authorization: Bearer $KEY" \
  -F 'file=@materials/physics_8_ru.pdf' \
  -F 'doc_type=textbook' \
  -F 'subject=physics' \
  -F 'grade=8' \
  -F 'lang=ru' \
  -F 'ocr=false'
```

Пример загрузки инструкции к лабораторной работе:

```bash
API_BASE=https://megroup-b560m-hdv-m-2.tail7dd37a.ts.net

curl -sS -X POST "$API_BASE/admin/documents" \
  -H "Authorization: Bearer $KEY" \
  -F 'file=@materials/physics_lab_02_ru.docx' \
  -F 'doc_type=lab_instruction' \
  -F 'subject=physics' \
  -F 'grade=10' \
  -F 'lang=ru' \
  -F 'lab_number=2' \
  -F 'ocr=false'
```

Для общего документа без предметных метаданных можно использовать тот же API
с одним полем `file` или CLI:

```bash
python -m scripts.manage_corpus upload path/to/document.pdf
```

API-вариант без метаданных:

```bash
curl -s localhost:8000/admin/documents \
  -H "Authorization: Bearer $KEY" \
  -F 'file=@path/to/document.pdf'
```

Управление загруженными документами:

```bash
API_BASE=https://megroup-b560m-hdv-m-2.tail7dd37a.ts.net

# Список документов и их file_id
curl -sS "$API_BASE/admin/documents" -H "Authorization: Bearer $KEY"

# Удаление документа и всех его чанков
curl -sS -X DELETE "$API_BASE/admin/documents/<file_id>" \
  -H "Authorization: Bearer $KEY"

# Состояние коллекции Qdrant
curl -sS "$API_BASE/admin/corpus_status" -H "Authorization: Bearer $KEY"
```

## Подключения и порты

Прокси является единственным сервисом, который ходит наружу. К каждому из трёх локальных
сервисов он подключается по HTTP через ленивый `httpx.AsyncClient` (Qdrant через
свой клиент). Под docker-compose адреса по умолчанию указывают на имена сервисов;
для запуска прокси вне Docker переопределите их на `localhost` + host-mapped порт.

| Подключение | Переменная | URL под compose (по умолчанию) | URL для локального dev | Эндпоинты, которые вызывает прокси |
|-------------|-----------|-------------------------------|------------------------|-----------------------------------|
| VR-клиент → API | `INTERNAL_API_KEY` | `Authorization: Bearer` | `Authorization: Bearer` | весь публичный API (см. ниже) |
| API → OpenAI | `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL` | `api.openai.com` (если `OPENAI_BASE_URL` пуст) | то же | Responses API |
| API → Qdrant | `QDRANT_URL` | `http://qdrant:6333` | `http://localhost:6333` | Query API (поиск), upsert/delete |
| API → embedder | `EMBEDDING_BASE_URL` | `http://embedder:8080` | `http://localhost:8080` | `POST /embed` |
| API → voice | `VOICE_BASE_URL` | `http://voice:8001` | `http://localhost:8002` | `POST /stt/recognize`, `POST /tts/synthesize?format=wav` |
| API → voice-omnivoice | `VOICE_KK_OMNIVOICE_BASE_URL` | `http://voice-omnivoice:8003` | `http://localhost:8003` | `POST /tts/synthesize?format=wav` |

> ⚠️ Локальный dev: значения по умолчанию в `.env.example` уже указывают на
> `localhost` (`QDRANT_URL=http://localhost:6333`, `EMBEDDING_BASE_URL=http://localhost:8080`,
> `VOICE_BASE_URL=http://localhost:8002`, `VOICE_KK_OMNIVOICE_BASE_URL=http://localhost:8003`).
> Под docker-compose их нужно переопределить на имена сервисов
> (`qdrant`/`embedder`/`voice`/`voice-omnivoice`). Это уже сделано в `environment:`
> блоке compose-файла, отдельно в `.env` менять не требуется.

### Публичный доступ (Tailscale Funnel, HTTPS)

API выставлен наружу через **Tailscale Funnel**, публичный HTTPS-эндпоинт с
автоматическим TLS, без проброса портов и без Tailscale на стороне клиента.
Наружу выставлен **только API (порт 8001)**; голосовой порт `8002` не публикуется.

| Что | URL | Авторизация |
|-----|-----|-------------|
| API прокси (`/ask`, `/voice_ask`, `/hint`, …) | `https://megroup-b560m-hdv-m-2.tail7dd37a.ts.net` | `Authorization: Bearer <INTERNAL_API_KEY>` |
| Swagger UI | `https://megroup-b560m-hdv-m-2.tail7dd37a.ts.net/docs` | вызовы требуют Bearer-ключ |
| Healthcheck | `https://megroup-b560m-hdv-m-2.tail7dd37a.ts.net/health` | нет |

Включается на сервере: `sudo tailscale funnel --bg 8001` (статус:
`tailscale funnel status`, выключение: `sudo tailscale funnel --https=443 off`).
Каждый вызов `/ask`/`/hint` идёт в OpenAI и стоит денег. Держите
`INTERNAL_API_KEY` в секрете и `RATE_LIMIT_PER_MINUTE > 0`.

### Переменные окружения (генерация и кэш)

| Переменная | Назначение |
|-----------|-----------|
| `OPENAI_API_KEY` | ключ OpenAI, обязательный для генерации ответов |
| `OPENAI_BASE_URL` | необязательный URL Azure или совместимого прокси |
| `OPENAI_MODEL` | модель Responses API, по умолчанию `gpt-4.1-mini` |
| `OPENAI_SERVICE_TIER` | пусто для стандартного режима или `priority` для меньшей задержки и большей стоимости |
| `REQUEST_TIMEOUT_S` | таймаут запроса генерации |
| `ANSWER_CACHE_SIZE` / `ANSWER_CACHE_TTL_S` | размер и TTL кэша повторяющихся ответов; ≤0 отключает кэш |
| `TTS_CACHE_SIZE` | размер кэша синтезированного аудио; ≤0 отключает кэш |
| `CHAT_MEMORY_MAX_CONVERSATIONS` / `CHAT_MEMORY_TTL_S` | размер и TTL эфемерной памяти VR-диалогов; ≤0 отключает серверную память |
| `CHAT_MEMORY_MAX_MESSAGES` / `CHAT_MEMORY_HISTORY_CHARS` | лимиты истории одного диалога |
| `CHAT_MEMORY_RETRIEVAL_CONTEXT_CHARS` | объём предыдущего диалога, добавляемый к ретривалу follow-up вопроса |
| `ALLOW_GENERAL_KNOWLEDGE_FALLBACK` | разрешить ответ из общеизвестных научных знаний без цитат только вне структурированного контекста лабораторной работы, если найденные материалы непригодны и нет авторитетного контекста сцены/процедуры |

### Переменные окружения (ретривал)

| Переменная | Назначение |
|-----------|-----------|
| `QDRANT_URL` | адрес Qdrant (compose: `http://qdrant:6333`, local: `http://localhost:6333`) |
| `QDRANT_COLLECTION` | имя коллекции (по умолчанию `school_kb`) |
| `EMBEDDING_BASE_URL` | адрес сайдкара-эмбеддера, вызов `POST /embed` |
| `EMBEDDING_DIM` | размерность плотного вектора (1024 для bge-m3) |
| `EMBED_BATCH_SIZE` | размер батча эмбеддинга (≤0 означает одним запросом) |
| `RETRIEVAL_TOP_K` | сколько чанков подставить в промт |
| `RETRIEVAL_CANDIDATES` | сколько кандидатов тянуть из каждой ветки до RRF |
| `RETRIEVAL_SCORE_THRESHOLD` | минимальный скор после слияния |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | параметры нарезки при загрузке |
| `CORPUS_ROOT` | корень структурированного корпуса для `bulk-ingest` |
| `LABS_MANIFEST` | путь для отчёта `gen-manifest` |
| `OCR_ENABLED` / `OCR_DPI` / `OCR_MAX_PAGES` | OCR для сканированных материалов при загрузке |

### Переменные окружения (голос)

| Переменная | Назначение |
|-----------|-----------|
| `VOICE_BASE_URL` | адрес основного voice-сайдкара (compose: `http://voice:8001`, local: `http://localhost:8002`) |
| `VOICE_KK_OMNIVOICE_BASE_URL` | адрес отдельного сайдкара Kazakh OmniVoice (compose: `http://voice-omnivoice:8003`, local: `http://localhost:8003`) |
| `VOICE_VERIFY_SSL` | проверять ли TLS (по умолчанию `false`, внутри compose plain HTTP) |
| `VOICE_TIMEOUT_S` | таймаут STT/TTS (по умолчанию `120`, покрывает GPU cold start) |
| `VOICE_TTS_RU_DEFAULT_BACKEND` | backend, который API явно отправляет для русского TTS без поля `backend`; по умолчанию `supertonic` |
| `VOICE_TTS_KK_DEFAULT_BACKEND` | backend по умолчанию для казахского TTS: `omnivoice` (фиксированный молодой мужской голос) или `mms` (fallback) |
| `OMNIVOICE_MODEL` / `OMNIVOICE_INSTRUCT` / `OMNIVOICE_STEPS` | checkpoint, фиксированный голосовой профиль и число diffusion-шагов отдельного OmniVoice-сайдкара |
| `OMNIVOICE_AUDIO_TOKENIZER_PATH` | смонтированный локальный путь к `eustlb/higgs-audio-v2-tokenizer`, чтобы запуск OmniVoice не зависел от Hugging Face |
| `OMNIVOICE_NORMALIZE_KK_NUMBERS` | раскрывать числа казахскими словами непосредственно перед OmniVoice; по умолчанию `true`, `false` отключает для диагностики |
| `TTS_RU_BACKEND` / `TTS_RU_BACKENDS` | default и список загружаемых русских backend-ов voice-сайдкара; по умолчанию `supertonic` и `supertonic,qwen` |
| `TTS_RU_QWEN_MODEL` | Hugging Face ID Qwen3-TTS; по умолчанию `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` |
| `TTS_RU_QWEN_SPEAKER` | мужской голос Qwen по умолчанию; `Aiden` |
| `TTS_RU_QWEN_DTYPE` / `TTS_RU_QWEN_ATTENTION` | тип весов и attention implementation; `bfloat16` / `sdpa` |
| `TTS_RU_SUPERTONIC_VOICE_STYLE` | стиль Supertonic по умолчанию; `M3` |
| `TTS_NORMALIZE_RU_NUMBERS` | раскрывать числа русскими словами перед Supertonic, Qwen и MMS; по умолчанию `true`, `false` отключает для диагностики |
| `DEFAULT_LANGUAGE` | резервный язык ответа для неоднозначного текста и язык TTS по умолчанию (`ru`/`kk`); STT без `language` использует `auto` |

### Нормализация чисел для TTS

Русский и казахский сайдкары используют отдельные детерминированные нормализаторы
из репозитория. Они обрабатывают целые и отрицательные числа, десятичные дроби с
точкой или запятой, проценты, даты, время, диапазоны, температуру и школьные
единицы измерения. Химические формулы, научная нотация, уравнения, URL, имена
файлов, лабораторные ID и буквенно-цифровые идентификаторы маскируются на время
преобразования и восстанавливаются без изменений. Общий `normalize_text` из
OmniVoice не используется, поэтому результат не зависит от `num2words` или
языка fallback-нормализатора библиотеки.

Примеры текста, который получает модель:

```text
Суды жиырма бес градус Цельсийге дейін қыздырыңыз.
Ерітіндіге үш бүтін жүзден он төрт грамм тұз қосыңыз.
Тәжірибе екі мың жиырма алтыншы жылғы он екінші наурыз күні сағат он төрт отызда басталады.
H2O молекуласында екі сутек атомы бар.

Нагрейте воду до двадцати пяти градусов Цельсия.
Добавьте три целых четырнадцать сотых грамма соли.
Опыт начнётся двенадцатого марта две тысячи двадцать шестого года в четырнадцать часов тридцать минут.
В молекуле H2O есть два атома водорода.
```

## Эндпоинты

Все требуют `Authorization: Bearer <INTERNAL_API_KEY>` (кроме `/health`, `/ready`).

| Метод | Путь | Назначение |
|------|------|-----------|
| POST | `/ask` | Вопрос с грунтингом: `{query, conversation_id?, scenario_id?, scenario_state?, lab?}` → ответ + `citations` + `conversation_id` |
| POST | `/v1/chat/completions` | OpenAI-совместимый чат (`stream` поддерживается), расширен полями `scenario_id`, `scenario_state`, `lab` |
| POST | `/hint` | Перефразирование подсказки: `{hint_text, hint_level, scenario_id?, scenario_state?}` (ТЗ §3 задача 2) |
| POST | `/stt` | multipart `file`, необязательный `language=ru/kk/auto` → `{text, language}`; без `language` включается автоопределение |
| POST | `/tts` | `{text, language?, backend?, voice?, format?, instructions?}` → WAV; русский `backend` = `supertonic` (default) / `qwen` / `mms`, казахский = `omnivoice` (default, фиксированный молодой мужской голос) / `mms` |
| POST | `/voice_ask` | multipart `file` + необязательный `conversation_id`, `language=ru/kk/auto`, `tts_backend` и поля сценария, состояния и лаборатории → `{question, language, answer, citations, conversation_id, audio_base64}` или SSE |
| DELETE | `/v1/conversations/{conversation_id}` | немедленно удалить эфемерную историю VR-диалога |
| GET | `/admin/corpus_status` | состояние коллекции Qdrant |
| POST | `/admin/documents` | загрузить общий или структурированный документ в KB: multipart `file`, метаданные и необязательный OCR |
| GET/DELETE | `/admin/documents[/{file_id}]` | список / удаление по `file_id` |
| GET | `/admin/scenarios` | список сценариев |
| GET | `/health`, `/ready` | проверки |

### Пример: `/ask`

```bash
curl -s localhost:8000/ask -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{
    "query": "Что делать дальше?",
    "scenario_id": "physics_lab_02_heating",
    "lab": {"subject": "physics", "grade": 10, "lang": "ru", "lab_number": 2}
  }'
```

```json
{
  "answer": "Следующий шаг: начните равномерно нагревать воду и следите за термометром...",
  "citations": [{
    "filename": "Лабораторная работа №2.docx",
    "file_id": "...",
    "source_type": "lab_instruction",
    "lab_id": "physics-10-ru-02",
    "lab_number": 2,
    "display_label": "Инструкция к лабораторной работе №2"
  }],
  "primary_source": {
    "filename": "Лабораторная работа №2.docx",
    "file_id": "...",
    "source_type": "lab_instruction",
    "lab_id": "physics-10-ru-02",
    "lab_number": 2,
    "display_label": "Инструкция к лабораторной работе №2"
  },
  "conversation_id": "0e8c0ea8-3ad6-49c4-8678-7f12c33024b2",
  "scenario_id": "physics_lab_02_heating",
  "observability": {"latency_ms": {"embed": 30.5, "retrieval": 18.1, "llm": 1840.2, "total": 1888.8}}
}
```

Клиент рисует под ответом блок «Источник: `primary_source.filename`». Для
«объёмных» ответов (ТЗ §4) клиент может показать источник по запросу. Данные
уже в `citations`. Поля страницы, главы и раздела добавляются только тогда,
когда ingestion смог надёжно определить их из исходного документа.
Для вопросов о текущем шаге и порядке действий инструкция к лабораторной работе
ставится первой. Для теоретических вопросов первым остаётся найденный учебник.

Для follow-up вопросов VR-клиент сохраняет `conversation_id` из первого ответа
и передаёт его в следующих `/ask` или `/voice_ask` запросах. Подробный контракт,
SSE-поля и lifecycle описаны в
[`docs/memory-backend-guide.md`](docs/memory-backend-guide.md).

## Сценарии (контекст ПО)

Каждая лабораторная использует один JSON-файл в `scenarios/` (имя файла = `scenario_id`).
Поля см. в `scenarios/physics_lab_02_heating.json`: `scenario_name`, `subject`,
`topic`, `environment_description`, `objects`, `action_sequence`, `risks`,
`common_mistakes`, `correct_behavior`, `regulations` и т.д. Заполняются только
нужные поля. Это статическое описание сцены, а не инструкция в базе знаний.
Тренажёр передаёт `scenario_id`, структурированный `lab` и актуальный
`scenario_state` в каждом запросе. Поля `scenario_state` включают текущий и
следующий шаги, завершённые шаги, предметы в руках и в поле зрения, разрешённые
действия, а также результат последнего действия.

## Тесты

```bash
pytest          # без сети и GPU (OpenAI, Qdrant и embedder замоканы)
```

Для ручной проверки API откройте `test_ui.html` в браузере, укажите Base URL и
введите `INTERNAL_API_KEY`. Ключ не хранится в файле или репозитории. Консоль
проверяет health/readiness, текстовые запросы, streaming, STT, TTS и полный
`voice_ask`-конвейер.

## Замечания по приёмке (ТЗ §7)

- **Только из загруженных файлов**: системный промт запрещает выдумывать факты;
  ответы грунтятся подставленными чанками из гибридного поиска по Qdrant,
  источники возвращаются в `citations`.
- **Текущий шаг**: актуальный `scenario_state` имеет приоритет над статическим
  сценарием по `scenario_id`.
- **≤5 c**: `/voice_ask` возвращает по-стадийные задержки (`stt`/`llm`/`tts`).
  STT/TTS обслуживает встроенный сайдкар `voice` (`VOICE_BASE_URL`, см. `./voice`);
  для минимума латентности держите сайдкар «прогретым» и используйте быстрый
  `OPENAI_MODEL`.
- **Источники на каждый ответ**: `primary_source` + `citations` в каждом ответе.
