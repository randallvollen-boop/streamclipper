"""The review page on a phone-sized screen.

One check with a real browser at 390x844 (an iPhone-sized viewport): the page
loads, the cards render, nothing scrolls sideways, and the console is clean.

The kit is a synthetic test pattern. The video requests are aborted on purpose (the route abort below), so no frame
of any recording is ever rendered. The console messages that the abort itself
causes are counted separately and are not page errors.
"""
from __future__ import annotations

import re
import threading

import pytest

from clipper.review.server import make_server
from test_review import world  # noqa: F401  (the fixture: a data folder with one kit)

playwright_api = pytest.importorskip("playwright.sync_api", reason="playwright is not installed")


@pytest.fixture
def server(world):  # noqa: F811
    """The review server on an unused port, over the synthetic kit."""
    instance = make_server(world.config, data_dir=world.data_dir, port=0)
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    yield instance
    instance.shutdown()
    instance.server_close()


def test_the_page_fits_a_phone(server, tmp_path):
    shot = tmp_path / "review-390.png"

    console_errors: list[str] = []
    page_errors: list[str] = []

    with playwright_api.sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(
            viewport={"width": 390, "height": 844},
            device_scale_factor=2,
            is_mobile=True,
            has_touch=True,
        )
        page.on(
            "console",
            lambda message: console_errors.append(message.text)
            if message.type == "error"
            else None,
        )
        page.on("pageerror", lambda error: page_errors.append(str(error)))
        # No recording is ever fetched, so no frame of any video is rendered.
        # The regex matches the cache-busting query string too.
        page.route(re.compile(r"\.mp4(\?.*)?$"), lambda route: route.abort())

        page.goto(f"http://127.0.0.1:{server.server_address[1]}/")
        page.wait_for_selector(".card", timeout=15000)

        cards = page.locator(".card").count()
        assert cards >= 1

        # Nothing scrolls sideways at 390 px.
        scroll_width = page.evaluate("document.documentElement.scrollWidth")
        client_width = page.evaluate("document.documentElement.clientWidth")
        assert scroll_width <= client_width + 1, (
            f"the page is {scroll_width} px wide, the screen is {client_width} px"
        )

        # The card carries what a person needs to decide.
        first = page.locator(".card").first
        assert page.locator(".card video").count() >= 1
        assert page.locator(".card .chip").count() >= 1
        assert first.locator("input[type=radio]").count() == 3
        assert page.get_by_role("button", name="Approve").count() >= 1
        assert page.get_by_role("button", name="Reject").count() >= 1
        assert first.locator("input[type=number]").count() == 2
        assert first.locator("input[type=range]").count() == 1

        page.screenshot(path=str(shot))
        browser.close()

    # The aborted video requests are the browser's own doing, not the page's.
    media_errors = [text for text in console_errors if ".mp4" in text or "ERR_FAILED" in text]
    other_errors = [text for text in console_errors if text not in media_errors]
    assert other_errors == [], f"console errors: {other_errors}"
    assert page_errors == [], f"page errors: {page_errors}"
    assert shot.exists()
