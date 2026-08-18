import os
import re
import sys
import time
from pathlib import Path

from playwright.sync_api import Browser, Page, sync_playwright


DEFAULT_APP_URLS = (
    "https://re4mo-price-prediction.streamlit.app",
)
APP_URLS = tuple(
    url.strip()
    for url in os.environ.get("STREAMLIT_APP_URLS", ",".join(DEFAULT_APP_URLS)).split(",")
    if url.strip()
)
STREAMLIT_WEBSOCKET_PATH = "/_stcore/stream"
ACCESS_DENIED_PATTERN = re.compile(
    r"you do not have access to this app or it does not exist",
    re.IGNORECASE,
)
WAKE_BUTTON_PATTERN = re.compile(
    r"yes,?\s*get this app back up!?|get this app back up!?|wake (?:this )?app",
    re.IGNORECASE,
)
APP_TIMEOUT_SECONDS = 300
SESSION_HOLD_SECONDS = 15
ARTIFACT_DIRECTORY = Path("playwright-artifacts")


def click_wake_button_if_visible(page: Page) -> bool:
    wake_buttons = page.get_by_role("button", name=WAKE_BUTTON_PATTERN)
    if wake_buttons.count() == 0:
        return False

    wake_button = wake_buttons.first
    if not wake_button.is_visible():
        return False

    wake_button.click(timeout=10_000)
    print("  Sleeping app detected; clicked the wake-up button.", flush=True)
    return True


def keep_app_awake(browser: Browser, url: str) -> None:
    context = browser.new_context()
    page = context.new_page()
    streamlit_socket_opened = False

    def record_websocket(websocket) -> None:
        nonlocal streamlit_socket_opened
        print(f"  WebSocket opened: {websocket.url}", flush=True)
        if STREAMLIT_WEBSOCKET_PATH in websocket.url:
            streamlit_socket_opened = True

    page.on("websocket", record_websocket)
    deadline = time.monotonic() + APP_TIMEOUT_SECONDS
    wake_button_clicked = False
    next_status_log = time.monotonic()

    try:
        print(f"Opening {url}", flush=True)
        page.goto(url, wait_until="domcontentloaded", timeout=APP_TIMEOUT_SECONDS * 1_000)

        while time.monotonic() < deadline:
            access_denied_message = page.get_by_text(ACCESS_DENIED_PATTERN)
            if (
                access_denied_message.count() > 0
                and access_denied_message.first.is_visible()
            ):
                raise PermissionError(
                    "The app is private, unavailable, or does not exist for an "
                    "unauthenticated browser."
                )

            if not wake_button_clicked:
                wake_button_clicked = click_wake_button_if_visible(page)

            if time.monotonic() >= next_status_log:
                print(
                    "  Waiting for readiness: "
                    f"url={page.url}, streamlit_websocket={streamlit_socket_opened}",
                    flush=True,
                )
                next_status_log = time.monotonic() + 30

            if streamlit_socket_opened:
                page.wait_for_timeout(SESSION_HOLD_SECONDS * 1_000)
                print(f"Successfully opened a live Streamlit session: {url}", flush=True)
                return

            page.wait_for_timeout(1_000)

        raise TimeoutError(
            "Timed out before a live Streamlit WebSocket session became ready."
        )
    except Exception:
        ARTIFACT_DIRECTORY.mkdir(exist_ok=True)
        screenshot_name = url.removeprefix("https://").replace("/", "_") + ".png"
        page.screenshot(path=str(ARTIFACT_DIRECTORY / screenshot_name), full_page=True)
        raise
    finally:
        context.close()


def main() -> int:
    failures = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            args=["--disable-dev-shm-usage"],
        )
        try:
            for url in APP_URLS:
                try:
                    keep_app_awake(browser, url)
                except Exception as error:
                    failures.append((url, error))
                    print(f"Failed to open {url}: {error}", file=sys.stderr, flush=True)
        finally:
            browser.close()

    if failures:
        print("\nKeep-alive failures:", file=sys.stderr)
        for url, error in failures:
            print(f"- {url}: {error}", file=sys.stderr)
        return 1

    print("\nAll Streamlit apps have active browser sessions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
