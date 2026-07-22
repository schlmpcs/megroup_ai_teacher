import os
import socket
import subprocess
import sys
import time

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_ADMIN_UI_SMOKE") != "1",
    reason="set RUN_ADMIN_UI_SMOKE=1 to run browser smoke tests",
)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_ready(url: str) -> None:
    import urllib.request

    for _ in range(100):
        try:
            with urllib.request.urlopen(url, timeout=0.2):
                return
        except Exception:
            time.sleep(0.05)
    raise RuntimeError(f"Server did not become ready: {url}")


def test_admin_ui_end_to_end(tmp_path):
    from playwright.sync_api import expect, sync_playwright

    from admin_ui.auth import hash_password

    backend_port = free_port()
    ui_port = free_port()
    env = {
        **os.environ,
        "ADMIN_UI_USERNAME": "admin",
        "ADMIN_UI_PASSWORD_HASH": hash_password("secret"),
        "ADMIN_UI_SESSION_SECRET": "session-secret-1234567890-abcdef",
        "ADMIN_UI_COOKIE_SECURE": "false",
        "BACKEND_BASE_URL": f"http://127.0.0.1:{backend_port}",
        "BACKEND_ADMIN_API_KEY": "smoke-backend-key",
        "FAKE_CORPUS_STATUS_FAIL": "false",
        "FAKE_INGESTION_STATUS_DELAY_S": "0",
        "FAKE_OCR_DEFAULT": "true",
    }
    backend = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "admin_ui.tests.fake_backend:app", "--port", str(backend_port)],
        env=env,
    )
    ui = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "admin_ui.main:app", "--port", str(ui_port)],
        env=env,
    )
    try:
        wait_ready(f"http://127.0.0.1:{backend_port}/docs")
        wait_ready(f"http://127.0.0.1:{ui_port}/health")
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.goto(f"http://127.0.0.1:{ui_port}")
            page.fill("#username", "admin")
            page.fill("#password", "secret")
            page.click("#loginForm button[type=submit]")
            expect(page.locator("#appView")).to_be_visible()
            expect(page.locator("#loginView")).to_be_hidden()
            expect(page.locator("#corpusOcr")).to_be_checked()

            page.set_input_files(
                "#fileInput",
                {"name": "notes.md", "mimeType": "text/markdown", "buffer": b"notes"},
            )
            expect(page.locator("#stagingTable tbody tr")).to_have_count(1)
            upload_ocr = page.locator('#stagingTable input[type="checkbox"]')
            expect(upload_ocr).to_be_checked()
            upload_ocr.uncheck()
            expect(upload_ocr).not_to_be_checked()
            page.select_option("#stagingTable select[data-field=kind]", "textbook")
            page.select_option("#stagingTable select[data-field=subject]", "physics")
            page.select_option("#stagingTable select[data-field=grade]", "8")
            page.select_option("#stagingTable select[data-field=lang]", "ru")

            with page.expect_response(
                lambda response: response.url.endswith("/api/admin/ingestion/jobs/upload")
                and response.request.method == "POST"
            ) as upload_response:
                page.click("#queueUpload")
            upload_job = upload_response.value.json()
            expect(page.locator("#jobDetails")).to_contain_text(f"Задание {upload_job['id']}")
            expect(page.locator("#jobDetails")).to_contain_text("notes.md")

            page.click('[data-view="ingest"]')
            page.click('[data-source="corpus"]')
            page.locator("#corpusOcr").uncheck()
            page.click("#previewCorpus")
            expect(page.locator("#corpusPreview")).to_contain_text("Распознано 1 из 2")
            expect(page.locator("#corpusPreview")).to_contain_text("misc.md")

            with page.expect_response(
                lambda response: response.url.endswith("/api/admin/ingestion/jobs/corpus")
                and response.request.method == "POST"
            ) as corpus_response:
                page.click("#queueCorpus")
            failed_job = corpus_response.value.json()
            expect(page.locator("#jobDetails")).to_contain_text(f"Задание {failed_job['id']}")
            expect(page.locator("#jobDetails")).to_contain_text("Synthetic smoke-test failure")

            failed_row = page.locator("#jobsTable tbody tr").filter(has_text="Ошибка").first
            expect(failed_row.get_by_role("button", name="Повторить")).to_be_visible()
            with page.expect_response(
                lambda response: response.url.endswith(
                    f"/api/admin/ingestion/jobs/{failed_job['id']}/retry"
                )
                and response.request.method == "POST"
            ) as retry_response:
                failed_row.get_by_role("button", name="Повторить").click()
            retried_job = retry_response.value.json()
            assert retried_job["retry_of"] == failed_job["id"]
            expect(page.locator("#jobDetails")).to_contain_text(f"Задание {retried_job['id']}")
            expect(page.locator("#jobDetails")).to_contain_text("Завершено; выполнено 1/1")

            page.click('[data-view="documents"]')
            page.fill("#documentSearch", "Physics")
            expect(page.locator("#documentsTable tbody tr")).to_have_count(1)

            desktop_screenshot = tmp_path / "admin-ui-desktop.png"
            page.screenshot(path=str(desktop_screenshot), full_page=True)
            page.locator("#documentsTable tbody tr").get_by_role("button", name="Удалить").click()
            expect(page.locator("#confirmDialog")).to_be_visible()
            expect(page.locator("#confirmMessage")).to_have_text(
                "Удалить Physics 8.md из базы знаний?"
            )
            with page.expect_response(
                lambda response: response.url.endswith("/api/admin/documents/doc-1")
                and response.request.method == "DELETE"
            ):
                page.click("#confirmAction")
            expect(page.locator("#documentsTable tbody tr")).to_have_count(0)

            mobile = browser.new_page(viewport={"width": 390, "height": 844})
            mobile.goto(f"http://127.0.0.1:{ui_port}")
            mobile.fill("#username", "admin")
            mobile.fill("#password", "secret")
            mobile.click("#loginForm button[type=submit]")
            expect(mobile.locator("#appView")).to_be_visible()
            expect(mobile.locator("#loginView")).to_be_hidden()
            mobile.set_input_files(
                "#fileInput",
                {"name": "mobile.md", "mimeType": "text/markdown", "buffer": b"mobile"},
            )
            expect(mobile.locator("#stagingTable tbody tr")).to_have_count(1)
            table_wrap = mobile.locator("#stagingTable").locator("xpath=..")
            assert table_wrap.evaluate("node => node.scrollWidth > node.clientWidth")
            topbar_box = mobile.locator(".topbar").bounding_box()
            tabs_box = mobile.locator(".tabs").bounding_box()
            table_box = table_wrap.bounding_box()
            assert topbar_box and tabs_box and table_box
            assert topbar_box["y"] + topbar_box["height"] <= tabs_box["y"] + 1
            assert tabs_box["y"] + tabs_box["height"] <= table_box["y"] + 1
            mobile_screenshot = tmp_path / "admin-ui-mobile.png"
            mobile.screenshot(path=str(mobile_screenshot), full_page=True)

            browser.close()
            assert desktop_screenshot.stat().st_size > 1000
            assert mobile_screenshot.stat().st_size > 1000
    finally:
        ui.terminate()
        backend.terminate()
        ui.wait(timeout=10)
        backend.wait(timeout=10)


def test_ocr_default_survives_corpus_failure_and_early_operator_override():
    from playwright.sync_api import expect, sync_playwright

    from admin_ui.auth import hash_password

    backend_port = free_port()
    ui_port = free_port()
    env = {
        **os.environ,
        "ADMIN_UI_USERNAME": "admin",
        "ADMIN_UI_PASSWORD_HASH": hash_password("secret"),
        "ADMIN_UI_SESSION_SECRET": "session-secret-1234567890-abcdef",
        "ADMIN_UI_COOKIE_SECURE": "false",
        "BACKEND_BASE_URL": f"http://127.0.0.1:{backend_port}",
        "BACKEND_ADMIN_API_KEY": "smoke-backend-key",
        "FAKE_CORPUS_STATUS_FAIL": "true",
        "FAKE_INGESTION_STATUS_DELAY_S": "1",
        "FAKE_OCR_DEFAULT": "true",
    }
    backend = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "admin_ui.tests.fake_backend:app", "--port", str(backend_port)],
        env=env,
    )
    ui = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "admin_ui.main:app", "--port", str(ui_port)],
        env=env,
    )
    try:
        wait_ready(f"http://127.0.0.1:{backend_port}/docs")
        wait_ready(f"http://127.0.0.1:{ui_port}/health")
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page()
            page.goto(f"http://127.0.0.1:{ui_port}")
            page.fill("#username", "admin")
            page.fill("#password", "secret")
            with page.expect_response(
                lambda response: response.url.endswith("/api/admin/ingestion/status")
            ) as status_response, page.expect_response(
                lambda response: response.url.endswith("/api/admin/corpus_status")
            ) as corpus_response:
                page.click("#loginForm button[type=submit]")
                expect(page.locator("#appView")).to_be_visible()
                page.click('[data-source="corpus"]')
                corpus_ocr = page.locator("#corpusOcr")
                corpus_ocr.check()
                corpus_ocr.uncheck()
                page.click('[data-source="upload"]')
                page.set_input_files(
                    "#fileInput",
                    {"name": "early.md", "mimeType": "text/markdown", "buffer": b"early"},
                )
                upload_ocr = page.locator('#stagingTable input[type="checkbox"]')
                expect(upload_ocr).not_to_be_checked()

            assert status_response.value.status == 200
            assert corpus_response.value.status == 503
            expect(upload_ocr).to_be_checked()
            expect(corpus_ocr).not_to_be_checked()
            page.evaluate("refreshAll().catch(() => {})")
            expect(corpus_ocr).not_to_be_checked()
            browser.close()
    finally:
        ui.terminate()
        backend.terminate()
        ui.wait(timeout=10)
        backend.wait(timeout=10)
