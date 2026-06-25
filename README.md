# VR AI Assistant (локальный гибридный RAG + OpenAI)

ИИ-помощник для школьного VR-тренажёра (физика, химия, биология). Тонкий
**stateless FastAPI-прокси** между VR-клиентом, локальным ретривал-стеком и
OpenAI:

- **База знаний — локальная** (self-hosted **Qdrant**). Гибридный поиск: плотный
  вектор (`dense`, 1024-d, cosine) + разреженный (`sparse`), две ветки `Prefetch`
  в Qdrant Query API, слитые через **RRF** (Reciprocal Rank Fusion).
- **Эмбеддинги — локальный GPU-сайдкар** `embedder` на модели **BAAI/bge-m3**
  (мультиязычная, сильна на русском и казахском; FlagEmbedding `BGEM3FlagModel`,
  отдаёт dense + learned-sparse). Рассчитан на NVIDIA RTX 3060 12GB (Ampere sm_86).
- **Генерация** — OpenAI **Responses API**. RAG явный: *retrieve → inject →
  generate* — найденные чанки подставляются в системный промт (а не hosted
  `file_search`). Ссылки на источники собираются из метаданных чанков (ТЗ §4).
- **Контекст сценария** (текущая сцена, шаги, объекты) подставляется в системный
  промт по `scenario_id` (ТЗ §3.2).
- **Голос**: STT и TTS обслуживает встроенный GPU-сайдкар `voice` (Whisper
  ru/kk/auto + supertonic ru / MMS kaz, каталог `./voice`); полный конвейер
  `/voice_ask` под целевую задержку ≤5 c (ТЗ §5, §7).

Прокси держит секретный ключ OpenAI у себя — VR-клиент аутентифицируется только
коротким `INTERNAL_API_KEY`, а реальный ключ OpenAI в приложение не попадает.

## Архитектура

```
VR-клиент (Unity)
   │  Authorization: Bearer <INTERNAL_API_KEY>
   ▼
FastAPI-прокси (этот репозиторий)            ┌─ scenarios/*.json  (контекст сцен)
   │                                          └─ инъекция в system prompt
   ├─► embedder (bge-m3, GPU)  ── dense + sparse эмбеддинг запроса
   ├─► Qdrant  ── гибридный поиск (dense + sparse, RRF) → top-k чанки
   │                                          └─ инъекция чанков в system prompt
   ├─► STT/TTS-сайдкар (GPU)  ── Whisper (ru/kk/auto) + MMS/Silero (ru/kk)
   ▼
OpenAI: Responses API (генерация, ключ OpenAI)
```

Документы (PDF/DOCX/TXT/MD) парсятся локально (pypdf / python-docx), нарезаются на
чанки, эмбеддятся через `embedder` и кладутся в Qdrant — всё on-prem. Ретривал
изолирован за `app/services/embeddings.py` (HTTP-клиент к эмбеддеру) и
`app/services/vectorstore.py` (обёртка над Qdrant Query API + RRF); генерация
осталась в облаке OpenAI. Форма ответа для VR-клиента (`citations`,
`primary_source`) не изменилась — ссылки пересобираются из метаданных чанков.

## Быстрый старт

Проще всего поднять весь стек (api + qdrant + embedder) через Docker:

```bash
cp .env.example .env          # заполнить INTERNAL_API_KEY и OPENAI_API_KEY
docker compose up --build     # api (хост-порт 8001) + qdrant + embedder
```

`embedder` — GPU-сайдкар, требует **NVIDIA Container Toolkit**; первый запуск
скачивает модель bge-m3, поэтому стартует медленно.

Локальная разработка прокси (Qdrant и embedder при этом удобно держать в Docker):

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # INTERNAL_API_KEY, OPENAI_API_KEY,
                              # QDRANT_URL, EMBEDDING_BASE_URL и т.д.

# 1) создать коллекцию в Qdrant (dense + sparse именованные векторы)
python -m scripts.manage_corpus create-collection

# 2) загрузить учебные материалы (парсинг → чанки → эмбеддинг → upsert в Qdrant)
python -m scripts.manage_corpus upload materials/physics_8.pdf
python -m scripts.manage_corpus upload materials/chem_9.docx
python -m scripts.manage_corpus list
python -m scripts.manage_corpus status

# 3) запустить сервис
uvicorn app.main:app --reload --port 8000
# docs: http://localhost:8000/docs
```

Удалить документ из коллекции: `python -m scripts.manage_corpus delete <doc_id>`.

### Переменные окружения (ретривал)

| Переменная | Назначение |
|-----------|-----------|
| `QDRANT_URL` | адрес Qdrant (напр. `http://qdrant:6333`) |
| `QDRANT_COLLECTION` | имя коллекции |
| `EMBEDDING_BASE_URL` | адрес сайдкара-эмбеддера (`POST /embed`) |
| `EMBEDDING_DIM` | размерность плотного вектора (1024 для bge-m3) |
| `RETRIEVAL_TOP_K` | сколько чанков подставить в промт |
| `RETRIEVAL_CANDIDATES` | сколько кандидатов тянуть из каждой ветки до RRF |
| `RETRIEVAL_SCORE_THRESHOLD` | минимальный скор после слияния |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | параметры нарезки при загрузке |

## Эндпоинты

Все требуют `Authorization: Bearer <INTERNAL_API_KEY>` (кроме `/health`, `/ready`).

| Метод | Путь | Назначение |
|------|------|-----------|
| POST | `/ask` | Вопрос с грунтингом: `{query, scenario_id?}` → ответ + `citations` |
| POST | `/v1/chat/completions` | OpenAI-совместимый чат (`stream` поддерживается), расширен полем `scenario_id` |
| POST | `/hint` | Перефразирование подсказки: `{hint_text, hint_level, scenario_id?}` (ТЗ §3 задача 2) |
| POST | `/stt` | multipart `file` → `{text}` |
| POST | `/tts` | `{text, language?}` → аудио (WAV) |
| POST | `/voice_ask` | multipart `file` (+`scenario_id?`) → `{question, answer, citations, audio_base64}` |
| GET | `/admin/corpus_status` | состояние коллекции Qdrant |
| POST | `/admin/documents` | загрузить документ в KB (multipart): парсинг → чанки → Qdrant |
| GET/DELETE | `/admin/documents[/{doc_id}]` | список / удаление по `doc_id` |
| GET | `/admin/scenarios` | список сценариев |
| GET | `/health`, `/ready` | проверки |

### Пример: `/ask`

```bash
curl -s localhost:8000/ask -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"query":"Зачем мы нагреваем пробирку?","scenario_id":"physics_lab_02_heating"}'
```
```json
{
  "answer": "Мы нагреваем пробирку, чтобы довести воду до кипения и измерить...",
  "citations": [{"filename": "physics_8.pdf", "doc_id": "..."}],
  "primary_source": {"filename": "physics_8.pdf", "doc_id": "..."},
  "scenario_id": "physics_lab_02_heating",
  "observability": {"latency_ms": {"embed": 30.5, "retrieval": 18.1, "llm": 1840.2, "total": 1888.8}}
}
```

Клиент рисует под ответом блок «Источник: `primary_source.filename`». Для
«объёмных» ответов (ТЗ §4) клиент может показать источник по запросу — данные
уже в `citations`.

## Сценарии (контекст ПО)

Каждая лабораторная — один JSON-файл в `scenarios/` (имя файла = `scenario_id`).
Поля см. в `scenarios/physics_lab_02_heating.json`: `scenario_name`, `subject`,
`topic`, `environment_description`, `objects`, `action_sequence`, `risks`,
`common_mistakes`, `correct_behavior`, `regulations` и т.д. Заполняются только
нужные поля. Тренажёр передаёт `scenario_id` (и при необходимости — отдельный
шаг/предметы в руках) в каждом запросе.

## Тесты

```bash
pytest          # без сети и GPU (OpenAI, Qdrant и embedder замоканы)
```

## Замечания по приёмке (ТЗ §7)

- **Только из загруженных файлов**: системный промт запрещает выдумывать факты;
  ответы грунтятся подставленными чанками из гибридного поиска по Qdrant,
  источники возвращаются в `citations`.
- **Текущий шаг**: подставляется из сценария по `scenario_id`.
- **≤5 c**: `/voice_ask` возвращает по-стадийные задержки (`stt`/`llm`/`tts`).
  STT/TTS обслуживает встроенный сайдкар `voice` (`VOICE_BASE_URL`, см. `./voice`);
  для минимума латентности держите сайдкар «прогретым» и используйте быстрый
  `OPENAI_MODEL`.
- **Источники на каждый ответ**: `primary_source` + `citations` в каждом ответе.
