# Профили нескольких ассистентов

## Назначение

Функция профилей позволяет обслуживать несколько типов ассистентов в одном
экземпляре FastAPI. Все профили используют общую инфраструктуру приложения:

- OpenAI Responses API для генерации ответа;
- локальный BAAI/bge-m3 embedder;
- один сервер Qdrant;
- сценарии лабораторий;
- STT и TTS sidecar-сервисы;
- очередь фоновой индексации.

При этом каждый профиль получает собственные доверенные настройки:

- базовый системный prompt;
- отдельную коллекцию Qdrant;
- отдельный корень файлового корпуса.

Клиент передаёт только идентификатор `assistant_type`. Клиент не может передать
произвольный системный prompt, имя коллекции Qdrant или путь к корпусу.
Фактическая конфигурация всегда выбирается на сервере: сначала из неизменяемого
реестра встроенных профилей, затем из общей SQLite-базы динамических профилей.

Если `assistant_type` не указан, используется профиль `vr_lab_teacher`. Поэтому
старые клиенты продолжают работать без изменений.

## Встроенные профили

| `assistant_type` | Назначение | Системный prompt | Коллекция Qdrant | Корень корпуса |
| --- | --- | --- | --- | --- |
| `vr_lab_teacher` | Основной VR-ассистент лаборатории | `You are a VR laboratory teaching assistant.` | `settings.QDRANT_COLLECTION` | `settings.CORPUS_ROOT` |
| `other_assistant` | Пример второго доменного ассистента | `You are the configured domain assistant.` | `other_assistant_kb` | `/data/other-corpus` |

В текущем Docker Compose профиль `vr_lab_teacher` обычно разрешается в
коллекцию `school_kb` и путь `/corpus`.

`other_assistant` является зарегистрированным примером профиля. Перед рабочим
использованием необходимо:

1. Смонтировать `/data/other-corpus` в контейнеры `api` и
   `ingestion-worker`.
2. Создать коллекцию `other_assistant_kb`.
3. Загрузить в неё документы.

Без этих шагов выбор профиля проходит проверку, но retrieval или корпусная
индексация не смогут работать полноценно.

Администратор может создавать дополнительные динамические профили через
`POST /admin/assistant-profiles`. Они не добавляются в Python-код и не требуют
перезапуска `api` или `ingestion-worker`.

## Модель профиля

Профиль описан неизменяемым dataclass:

```python
@dataclass(frozen=True)
class AssistantProfile:
    assistant_type: str
    system_prompt: str
    qdrant_collection: str
    corpus_root: str
```

Встроенный реестр `ASSISTANT_PROFILES` обёрнут в `MappingProxyType`, поэтому его
нельзя изменить во время работы процесса. При импорте модуля выполняется
проверка:

- профиль по умолчанию существует;
- идентификаторы соответствуют `^[a-z][a-z0-9_]{0,63}$`;
- ключ реестра совпадает с `profile.assistant_type`;
- системный prompt не пустой, не имеет внешних пробелов и не превышает
  `MAX_INPUT_CHARS`;
- имя коллекции не пустое;
- путь корпуса не пустой;
- одна коллекция Qdrant не назначена двум профилям;
- один corpus root не назначен двум профилям.

Уникальность коллекции является обязательной частью изоляции. Два профиля не
могут намеренно или случайно читать и изменять одну и ту же базу знаний.

Динамические профили хранятся отдельно:

```text
Path(settings.INGESTION_DATA_DIR) / "assistant_profiles.sqlite3"
```

Таблица содержит только доверенные серверные значения:

```sql
CREATE TABLE assistant_profiles (
    assistant_type TEXT PRIMARY KEY,
    system_prompt TEXT NOT NULL,
    qdrant_collection TEXT NOT NULL UNIQUE,
    corpus_root TEXT NOT NULL UNIQUE
);
```

База работает в WAL-режиме с `busy_timeout=5000`. Схема инициализируется
идемпотентно при старте API и worker, а при вызове из CLI или unit test может
быть создана лениво. Строки профилей не кэшируются: каждый процесс читает общую
SQLite-базу при разрешении динамического `assistant_type`, поэтому новая запись
видна API, worker и CLI сразу.

Разрешение всегда выполняется в таком порядке:

1. встроенный `ASSISTANT_PROFILES`;
2. динамическая строка SQLite;
3. `ValueError("Unknown assistant_type: ...")`.

Встроенные профили в SQLite не записываются и не могут быть заменены
динамической записью.

## Общий поток запроса

```text
assistant_type от клиента
  -> проверка идентификатора по доверенному реестру
  -> получение AssistantProfile
  -> загрузка памяти в namespace профиля
  -> embedding запроса общим bge-m3
  -> hybrid retrieval из profile.qdrant_collection
  -> загрузка инструкции лаборатории из той же коллекции
  -> сборка prompt из profile.system_prompt и общих правил
  -> генерация через общий OpenAI Responses API
  -> цитаты только из выбранной коллекции
  -> запись ответа в cache и память с ключом профиля
  -> общий TTS, если запрос голосовой
```

Код не изменяет `settings.QDRANT_COLLECTION` во время запроса. Имя коллекции
передаётся явно по цепочке вызовов. Это позволяет одновременно выполнять
запросы разных профилей без гонки за глобальную настройку.

## Публичный API

### Общие правила `assistant_type`

- Тип: строка.
- Минимальная длина: 1 символ.
- Максимальная длина: 64 символа.
- Значение по умолчанию: `vr_lab_teacher`.
- Неизвестное значение: HTTP `422` до retrieval или генерации.

Пример ошибки:

```json
{
  "detail": "Unknown assistant_type: 'missing'. Available assistant types: vr_lab_teacher, other_assistant"
}
```

### `POST /ask`

`assistant_type` передаётся в JSON на верхнем уровне:

```bash
curl -sS "$API_BASE/ask" \
  -H "Authorization: Bearer $INTERNAL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "assistant_type": "other_assistant",
    "query": "Объясни основную идею документа",
    "conversation_id": "student-session-42",
    "language": "ru"
  }'
```

Поле работает одинаково для обычного и потокового режима `stream=true`.
Форма ответа, SSE-события, `citations`, `primary_source`, `conversation_id` и
`usage` не изменились. Поле `assistant_type` в ответ автоматически не
добавляется, поэтому клиент должен хранить выбранный тип вместе с состоянием
сессии.

### `POST /v1/chat/completions`

`assistant_type` является расширением совместимого chat-запроса и передаётся
рядом с `model`, `messages` и `stream`:

```bash
curl -sS "$API_BASE/v1/chat/completions" \
  -H "Authorization: Bearer $INTERNAL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "assistant_type": "other_assistant",
    "model": "gpt-4.1-mini",
    "messages": [
      {"role": "user", "content": "Кратко объясни этот термин"}
    ]
  }'
```

Этот endpoint остаётся client-managed: клиент передаёт полный массив
`messages` при каждом запросе. Серверная `conversation_id`-память здесь не
используется, но профиль выбирает системный prompt и коллекцию retrieval.

### `POST /voice_ask`

Для голосового запроса `assistant_type` передаётся как multipart form field:

```bash
curl -sS "$API_BASE/voice_ask" \
  -H "Authorization: Bearer $INTERNAL_API_KEY" \
  -F "file=@question.wav" \
  -F "assistant_type=other_assistant" \
  -F "conversation_id=student-session-42" \
  -F "language=auto" \
  -F "response_language=ru"
```

Профиль применяется только к генерации, retrieval, цитатам и памяти. STT и TTS
остаются общими и продолжают выбираться по языку, voice и backend. Потоковый
режим голосового endpoint использует тот же профиль для всех SSE-событий одного
запроса.

### Очистка разговора

Разговор удаляется только из namespace выбранного профиля:

```bash
curl -sS -X DELETE \
  "$API_BASE/v1/conversations/student-session-42?assistant_type=other_assistant" \
  -H "Authorization: Bearer $INTERNAL_API_KEY"
```

Если удалить тот же `conversation_id` без `assistant_type`, будет очищена
память только профиля `vr_lab_teacher`.

### Endpoints без профиля

Следующие endpoints не принимают `assistant_type`:

- `POST /hint`;
- `POST /stt`;
- `POST /tts`;
- `GET /health`;
- `GET /ready`.

`/stt` и `/tts` не обращаются к профильной базе знаний. `/hint` использует
отдельный фиксированный prompt для переформулирования уже подготовленной
подсказки и не выполняет профильный retrieval.

## Сборка системного prompt

Профиль не заменяет весь существующий prompt. Итоговая инструкция собирается
послойно:

1. `profile.system_prompt` задаёт роль выбранного ассистента.
2. Общие правила сервиса задают формат ответа, использование доказательств,
   язык, ограничения по области и поведение при недостатке данных.
3. Добавляется точный язык ответа.
4. При наличии лабораторного контекста добавляются ограничения текущего
   предмета и лаборатории.
5. Из выбранной коллекции добавляется инструкция текущей лаборатории.
6. Добавляются статический сценарий и актуальный `scenario_state`.
7. Добавляются найденные фрагменты базы знаний.

Таким образом, профиль меняет роль и источник знаний, но сохраняет общие
правила безопасности и grounding текущего VR-приложения.

Это важно для новых доменов: текущий общий prompt всё ещё описывает школьную
VR-лабораторию по физике, химии и биологии. Для полностью независимого
ассистента вне этой области потребуется отдельная доработка общей prompt-части,
а не только добавление записи в реестр.

## Изоляция Qdrant и retrieval

Все операции vector store принимают необязательный `collection_name`:

- создание коллекции;
- upsert чанков;
- dense+sparse hybrid search с RRF;
- получение инструкции по `lab_id`;
- список документов;
- удаление документа;
- статус коллекции.

Если внутренний старый вызов не передаёт `collection_name`, используется
`settings.QDRANT_COLLECTION`. Профильные request-path и ingestion-path всегда
передают `profile.qdrant_collection` явно.

Для каждого ответа выполняются следующие профильные операции:

1. Теоретические чанки ищутся только в коллекции профиля.
2. Инструкция лаборатории по `lab_id` читается из той же коллекции.
3. Фильтры `doc_type`, `subject`, `grade` и `lang` применяются внутри выбранной
   коллекции.
4. Цитаты строятся из payload найденных чанков этой коллекции.

Embedder остаётся общим. Разделение выполняется на уровне хранилища, поэтому
одинаковые embedding-векторы запроса могут использоваться для разных профилей,
но Qdrant никогда не смешивает результаты между их коллекциями.

Для динамического профиля коллекция создаётся или проверяется автоматически во
время `POST /admin/assistant-profiles`, до записи профиля в SQLite. Если Qdrant
недоступен, профиль не сохраняется. Если коллекцию позднее удалить вручную,
retrieval вернёт ошибку Qdrant и не переключится на коллекцию профиля по
умолчанию.

## Изоляция кэша ответов

Ключ in-process кэша ответа теперь начинается с `assistant_type`. Остальные
компоненты ключа сохранены:

- нормализованный текст вопроса;
- статический сценарий;
- live `scenario_state`;
- структурированный `lab`;
- лимит токенов;
- язык ответа.

Одинаковый вопрос двум профилям создаёт две независимые записи. Потоковая и
обычная генерация используют общий кэш, но попадание всегда ограничено типом
ассистента.

Кэш применяется только к одиночным запросам без содержательной истории
диалога. Ответы с несколькими turns не кэшируются.

`POST /admin/cache/answers/clear` очищает весь кэш сразу. После изменения
корпуса worker также очищает общий кэш. Это безопасно для изоляции, но грубо:
изменение одного профиля сбрасывает кэш всех профилей.

## Изоляция памяти разговора

Серверная память хранит разговор по составному ключу:

```text
(assistant_type, conversation_id)
```

Следствия:

- один `conversation_id` можно технически использовать в разных профилях;
- история профилей при этом не смешивается;
- смена `assistant_type` в середине клиентской сессии создаёт отдельную историю;
- очистка разговора требует передать тот же `assistant_type`, который
  использовался при общении.

Память по-прежнему является TTL/LRU состоянием процесса. Перезапуск или deploy
контейнера `api` очищает её. В multi-replica конфигурации профили не решают
проблему распределённой памяти: всё ещё нужны sticky sessions или общий backend.

## Административный API

Все административные endpoints сохраняют существующую авторизацию через
`ADMIN_API_KEY`. Профиль задаётся следующим образом:

| Endpoint | Где передаётся `assistant_type` | Что выбирает профиль |
| --- | --- | --- |
| `GET /admin/assistant-profiles` | не передаётся | Список встроенных и динамических профилей |
| `POST /admin/assistant-profiles` | JSON | Создание динамического профиля |
| `GET /admin/corpus_status` | query | Коллекцию для статистики |
| `POST /admin/documents` | multipart form | Коллекцию для прямой загрузки |
| `POST /admin/ingestion/preview` | JSON | Проверку допустимого профиля |
| `POST /admin/ingestion/jobs/upload` | multipart form | Профиль фоновой upload-задачи |
| `POST /admin/ingestion/corpus/preview` | JSON | Корень корпуса и коллекцию сравнения |
| `POST /admin/ingestion/jobs/corpus` | JSON | Корень корпуса и коллекцию фоновой задачи |
| `GET /admin/documents` | query | Коллекцию для списка документов |
| `DELETE /admin/documents/{file_id}` | query | Коллекцию для удаления документа |

В endpoints управления корпусом значение по умолчанию равно
`vr_lab_teacher`. При создании профиля `assistant_type` обязателен, а endpoint
списка не принимает его.

### Список профилей

```bash
curl -sS "$API_BASE/admin/assistant-profiles" \
  -H "Authorization: Bearer $ADMIN_API_KEY"
```

Встроенные профили возвращаются первыми в порядке `ASSISTANT_PROFILES`, затем
динамические профили по алфавиту `assistant_type`:

```json
{
  "profiles": [
    {
      "assistant_type": "vr_lab_teacher",
      "system_prompt": "You are a VR laboratory teaching assistant.",
      "qdrant_collection": "school_kb",
      "corpus_root": "/corpus",
      "built_in": true
    },
    {
      "assistant_type": "math_tutor",
      "system_prompt": "You are a mathematics tutor.",
      "qdrant_collection": "math_tutor_kb",
      "corpus_root": "/data/ingestion/assistant_corpora/math_tutor",
      "built_in": false
    }
  ]
}
```

### Создание профиля

Запрос принимает ровно два поля:

```bash
curl -sS -X POST "$API_BASE/admin/assistant-profiles" \
  -H "Authorization: Bearer $ADMIN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "assistant_type": "math_tutor",
    "system_prompt": "You are a mathematics tutor."
  }'
```

Успешный ответ имеет HTTP `201`:

```json
{
  "assistant_type": "math_tutor",
  "system_prompt": "You are a mathematics tutor.",
  "qdrant_collection": "math_tutor_kb",
  "corpus_root": "/data/ingestion/assistant_corpora/math_tutor",
  "built_in": false
}
```

Backend сам выводит инфраструктурные значения:

```text
qdrant_collection = "<assistant_type>_kb"
corpus_root = Path(INGESTION_DATA_DIR) / "assistant_corpora" / "<assistant_type>"
```

Corpus-каталог создаётся автоматически на общем volume `ingestion_data`.
Поля `qdrant_collection`, `corpus_root`, `model`, `temperature` и любые другие
неизвестные поля запрещены и возвращают HTTP `422`.

Правила валидации:

- `assistant_type` должен точно соответствовать
  `^[a-z][a-z0-9_]{0,63}$`;
- идентификатор не приводится к нижнему регистру и не обрезается;
- внешние пробелы `system_prompt` удаляются перед сохранением;
- пустой prompt и prompt длиннее `MAX_INPUT_CHARS` возвращают HTTP `422`;
- внутренние пробелы и Unicode сохраняются.

HTTP `409` возвращается при повторе встроенного или динамического
`assistant_type`, конфликте производной коллекции или corpus root, а также если
конкурирующий запрос первым записал профиль. Ошибка создания коллекции Qdrant
проходит через общую обработку upstream-ошибок (`502` или `504`) и не создаёт
строку SQLite.

### Проверка статуса коллекции

```bash
curl -sS \
  "$API_BASE/admin/corpus_status?assistant_type=other_assistant" \
  -H "Authorization: Bearer $ADMIN_API_KEY"
```

### Прямая загрузка документа

```bash
curl -sS -X POST "$API_BASE/admin/documents" \
  -H "Authorization: Bearer $ADMIN_API_KEY" \
  -F "assistant_type=other_assistant" \
  -F "file=@manual.pdf"
```

Структурированные поля `doc_type`, `subject`, `grade`, `lang`, `lab_number` по
прежнему поддерживаются и записываются в payload выбранной коллекции.

### Фоновая индексация корпуса

```bash
curl -sS -X POST "$API_BASE/admin/ingestion/jobs/corpus" \
  -H "Authorization: Bearer $ADMIN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "assistant_type": "other_assistant",
    "subtree": "",
    "ocr": false,
    "prune": false
  }'
```

`subtree` вычисляется внутри `profile.corpus_root`. При `prune=true` разрешён
только полный профильный corpus root, без `subtree`.

### Глобальные endpoints очереди

Endpoints списка, просмотра, отмены, retry и удаления jobs не фильтруются по
профилю:

- `GET /admin/ingestion/jobs`;
- `GET /admin/ingestion/jobs/{job_id}`;
- `POST /admin/ingestion/jobs/{job_id}/cancel`;
- `POST /admin/ingestion/jobs/{job_id}/retry`;
- `DELETE /admin/ingestion/jobs/{job_id}`.

Выбранный профиль виден в `job.options.assistant_type`. Retry сохраняет
оригинальный `assistant_type`.

Текущий web admin UI не создаёт и не изменяет профили и продолжает работать с
`vr_lab_teacher` по умолчанию. Профиль создаётся через backend admin API, а его
корпус можно обслуживать через admin API или CLI, пока в UI не добавлено
управление профилями.

## Безопасность фоновых jobs

Очередь хранится в существующей SQLite базе. Изменение схемы базы не
потребовалось, поскольку профиль записывается в `options_json`.

При постановке задачи в очередь выполняются следующие действия:

1. `assistant_type` проверяется по встроенному реестру и динамической SQLite-базе.
2. Значение нормализуется до канонического `profile.assistant_type`.
3. Поля `system_prompt`, `qdrant_collection` и `corpus_root`, если они появились
   во входных options, удаляются.
4. Для upload-job сохраняется только доверенный `assistant_type`.
5. Для corpus-job сохраняются рабочие параметры `subtree`, `ocr`, `prune` и
   доверенный `assistant_type`.

Worker не доверяет сохранённым путям и именам коллекций. Перед выполнением он
снова разрешает `assistant_type` через встроенный реестр или общую SQLite-базу и
использует:

- `profile.corpus_root` для чтения corpus-файлов;
- `profile.qdrant_collection` для записи, prune и проверки документов.

Файлы corpus-job дополнительно проверяются на нахождение внутри профильного
corpus root. Файлы ручного upload-job проверяются на нахождение внутри
`INGESTION_DATA_DIR`.

Старые jobs без `assistant_type` выполняются как `vr_lab_teacher`. Новая
динамическая запись доступна уже работающему worker при разрешении следующей
задачи. Если строку профиля удалить из SQLite вручную до выполнения поставленной
задачи, worker завершит её с `Unknown assistant_type`, а не направит данные в
другую коллекцию.

Конфигурация профиля разрешается во время выполнения. Поэтому не следует менять
коллекцию или corpus root существующего `assistant_type`, пока в очереди есть
ожидающие jobs этого профиля.

## CLI управления корпусом

Каждая команда `scripts.manage_corpus` поддерживает
`--assistant-type`. Значение по умолчанию равно `vr_lab_teacher`.

CLI использует тот же `get_assistant_profile()`. При одинаковом
`INGESTION_DATA_DIR` он видит созданные через admin API динамические профили без
изменения кода.

### Создать коллекцию

```bash
python -m scripts.manage_corpus create-collection \
  --assistant-type other_assistant
```

Команда создаёт коллекцию с теми же named vectors `dense` и `sparse`, размером
embedding и distance, которые использует основной профиль.

### Загрузить отдельные файлы

```bash
python -m scripts.manage_corpus upload manual.pdf glossary.md \
  --assistant-type other_assistant
```

### Загрузить профильный корпус

```bash
python -m scripts.manage_corpus bulk-ingest \
  --assistant-type other_assistant
```

Если позиционный `root` не указан, CLI использует `profile.corpus_root`.
Оператор может явно передать другой root, но запись всё равно выполняется в
коллекцию выбранного профиля:

```bash
python -m scripts.manage_corpus bulk-ingest /srv/import/other-corpus \
  --assistant-type other_assistant \
  --ocr
```

`--only` и `--prune` сохраняют прежнюю семантику. `--prune` удаляет документы
только из коллекции выбранного профиля и не может использоваться вместе с
`--only`.

### Проверить и обслуживать коллекцию

```bash
python -m scripts.manage_corpus status \
  --assistant-type other_assistant

python -m scripts.manage_corpus list \
  --assistant-type other_assistant

python -m scripts.manage_corpus delete <doc_id> \
  --assistant-type other_assistant
```

### Сформировать manifest

```bash
python -m scripts.manage_corpus gen-manifest \
  --assistant-type other_assistant \
  --out other-assistant-labs.json
```

Для `gen-manifest` профиль выбирает default corpus root. Сама генерация
manifest остаётся offline-операцией и не обращается к Qdrant. Для нескольких
профилей рекомендуется указывать отдельный `--out`, чтобы не перезаписывать
общий `labs.json`.

Неизвестный `assistant_type` завершается ошибкой argparse до сетевых или
файловых операций.

## Добавление нового динамического профиля

### 1. Создать профиль через backend

Вызовите `POST /admin/assistant-profiles`, как показано выше. Backend проверит
идентификатор и конфликты, создаст коллекцию Qdrant, создаст corpus-каталог и
сохранит профиль в SQLite.

Изменение кода, новый Docker volume, rebuild и перезапуск сервисов не нужны.
Встроенные `vr_lab_teacher` и `other_assistant` остаются только в
`ASSISTANT_PROFILES`, неизменяемы и не могут быть повторно созданы через API.

### 2. Заполнить базу знаний

Для отдельных документов используйте `POST /admin/documents` или upload-job с
тем же `assistant_type`. Для corpus-job положите дерево файлов в автоматически
созданный каталог
`/data/ingestion/assistant_corpora/<assistant_type>` на общем volume и запустите:

```bash
curl -sS -X POST "$API_BASE/admin/ingestion/jobs/corpus" \
  -H "Authorization: Bearer $ADMIN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "assistant_type": "physics_exam_coach",
    "subtree": "",
    "ocr": false,
    "prune": false
  }'
```

Коллекция уже существует после создания профиля. Команда
`scripts.manage_corpus create-collection` остаётся идемпотентной и полезна для
диагностики или восстановления коллекции, удалённой вручную.

### 3. Подключить клиента

Клиент должен передавать один и тот же `assistant_type`:

- во всех turns разговора;
- в текстовом и голосовом продолжении одной сессии;
- при очистке `conversation_id`;
- при административных операциях с корпусом.

Не следует позволять пользователю вводить произвольную строку как имя
коллекции. UI должен выбирать только известные продуктовые типы и отправлять их
как `assistant_type`.

## Проверка после настройки

### Список профилей backend

```bash
curl -sS "$API_BASE/admin/assistant-profiles" \
  -H "Authorization: Bearer $ADMIN_API_KEY"
```

### Статус профильной коллекции

```bash
curl -sS \
  "$API_BASE/admin/corpus_status?assistant_type=physics_exam_coach" \
  -H "Authorization: Bearer $ADMIN_API_KEY"
```

Ожидаемый результат после загрузки:

- `status` равен `ready`;
- `collection` содержит имя коллекции профиля;
- `points` больше нуля;
- `documents` больше нуля.

### Проверка изоляции

1. Задайте одинаковый вопрос двум профилям.
2. Убедитесь, что цитаты относятся к соответствующим коллекциям.
3. Используйте одинаковый `conversation_id` для двух профилей.
4. Убедитесь, что второй профиль не получил историю первого.
5. Удалите документ из одного профиля и проверьте, что документ второго
   профиля сохранился.

### Состояние сервисов

```bash
docker compose ps api ingestion-worker
curl -sS "$API_BASE/health"
curl -sS "$API_BASE/ready"
docker compose logs --since 10m --tail 200 api ingestion-worker
```

## Обратная совместимость

Функция добавлена без обязательных изменений для существующих клиентов:

- отсутствие `assistant_type` выбирает `vr_lab_teacher`;
- существующие коллекция `school_kb` и corpus root продолжают использоваться;
- формы JSON, SSE и OpenAI-compatible ответов не изменены;
- старые внутренние вызовы vectorstore и ingestion без `collection_name`
  продолжают использовать `settings.QDRANT_COLLECTION`;
- старые ingestion jobs без `assistant_type` выполняются в default-профиле;
- существующая SQLite-база ingestion jobs не мигрируется, профили используют
  отдельный файл `assistant_profiles.sqlite3`;
- STT, TTS и их кэши не изменены.

## Что разделено и что остаётся общим

| Ресурс | Изоляция |
| --- | --- |
| Системный prompt профиля | Отдельный |
| Коллекция Qdrant | Отдельная |
| Корень корпуса | Отдельный |
| Retrieval и lab instruction | Отдельные через коллекцию |
| Цитаты | Отдельные через результаты коллекции |
| Кэш ответов | Разделён ключом `assistant_type` |
| Память разговора | Разделена ключом `(assistant_type, conversation_id)` |
| Ingestion upload, bulk ingest, prune, list, delete | Выполняются в профиле |
| OpenAI model и generation settings | Общие |
| BAAI/bge-m3 embedder | Общий |
| Qdrant server | Общий, коллекции разные |
| STT и TTS | Общие |
| Сценарии лабораторий | Общие |
| Rate limiting и API keys | Общие |
| Очередь ingestion jobs | Общая, профиль хранится в options |
| Очистка answer cache | Общая для всех профилей |

## Текущие ограничения

1. Обновление и удаление динамических профилей не реализованы.
2. Встроенные профили по-прежнему меняются только через код и deploy.
3. Профиль не выбирает отдельную OpenAI model, temperature, token limit,
   rate limit или voice-конфигурацию.
4. Общая часть prompt ориентирована на школьную VR-лабораторию.
5. Web admin UI не управляет профилями и использует default-профиль, если не
   расширен отдельно.
6. Список ingestion jobs общий и не имеет server-side фильтра по профилю.
7. Очистка answer cache после изменения корпуса затрагивает все профили.
8. Память и answer cache находятся в процессе API и не разделяются между
   несколькими replicas.
9. `other_assistant` требует отдельного volume mount и наполнения коллекции.
10. Ответ API не повторяет `assistant_type`, поэтому корректное состояние
    профиля является ответственностью клиента.

## Диагностика

### HTTP `422 Unknown assistant_type`

Причина: идентификатор отсутствует и во встроенном `ASSISTANT_PROFILES`, и в
`assistant_profiles.sqlite3`, либо клиент отправил опечатку.

Проверка:

```bash
docker compose exec -T api python -c \
  "from app.services.assistant_profiles import available_assistant_types; print(available_assistant_types())"
```

### Ошибка Qdrant только у одного профиля

Проверьте существование и статус его коллекции:

```bash
python -m scripts.manage_corpus status --assistant-type <type>
python -m scripts.manage_corpus create-collection --assistant-type <type>
```

Сервис намеренно не переключается на `school_kb`, если профильная коллекция
отсутствует.

### Corpus root не найден

Для динамического профиля проверьте автоматически созданный каталог на общем
volume `ingestion_data` в обоих процессах:

```bash
docker compose exec -T api \
  ls -la /data/ingestion/assistant_corpora/<assistant_type>
docker compose exec -T ingestion-worker \
  ls -la /data/ingestion/assistant_corpora/<assistant_type>
```

Для встроенного `other_assistant` по-прежнему требуется его отдельный mount.

### Follow-up не видит предыдущий вопрос

Проверьте, что клиент повторно передал одновременно:

- тот же `conversation_id`;
- тот же `assistant_type`;
- актуальные `scenario_state` и `lab`, если они нужны запросу.

### Job завершился с `Unknown assistant_type`

Строка динамического профиля отсутствует или повреждена в общей SQLite-базе.
Проверьте `GET /admin/assistant-profiles` и целостность общего volume. Backend не
имеет endpoint удаления, поэтому штатное создание профиля не удаляет уже
поставленные jobs.

## Тестовое покрытие

Автоматические тесты проверяют:

- default и явное разрешение профиля;
- неизменяемость и валидацию реестра;
- идемпотентную SQLite-схему и чтение через новые соединения;
- создание и список динамических профилей через admin API;
- строгую валидацию, запрет лишних полей и конфликты `409`;
- отсутствие SQLite-записи при ошибке Qdrant;
- HTTP `422` для неизвестного типа до генерации и файловых операций;
- передачу профиля через `/ask`, `/v1/chat/completions` и `/voice_ask`;
- изоляцию conversation memory;
- изоляцию answer cache;
- явную передачу коллекции в retrieval и vectorstore;
- профильные admin upload, preview, corpus job, list, status и delete;
- очистку недоверенных полей в queued jobs;
- сохранение профиля при retry;
- немедленное разрешение динамического профиля worker во время выполнения;
- безопасность corpus paths;
- профильные команды CLI.

Полная проверка проекта:

```bash
pytest
```

## Основные файлы реализации

| Файл | Ответственность |
| --- | --- |
| `app/services/assistant_profiles.py` | Встроенный реестр, SQLite-хранилище, создание и разрешение профилей |
| `app/api/routes.py` | Выбор профиля в публичных endpoints и namespace памяти |
| `app/services/llm.py` | Профильный prompt, retrieval, lab lookup и cache key |
| `app/services/vectorstore.py` | Явный `collection_name` для всех операций Qdrant |
| `app/services/memory.py` | Ключ `(assistant_type, conversation_id)` |
| `app/services/ingestion.py` | Профильная запись, list, delete, status и prune |
| `app/api/admin_routes.py` | Создание, список и выбор профиля в административных операциях |
| `app/services/ingestion_jobs.py` | Валидация и сохранение безопасных job options |
| `app/services/ingestion_worker.py` | Разрешение доверенного профиля при выполнении job |
| `scripts/manage_corpus.py` | CLI-флаг `--assistant-type` |
