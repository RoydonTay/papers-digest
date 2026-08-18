"""Notifier strategy: email today, other channels (e.g. Telegram) can implement the same protocol later."""

from __future__ import annotations

import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Protocol

from src.retry import with_retry

logger = logging.getLogger(__name__)


class Notifier(Protocol):
    def send(self, subject: str, html: str, text: str) -> None: ...


class EmailNotifier:
    """Gmail SMTP over implicit SSL (port 465), authenticated with an App Password."""

    def __init__(self, host: str, port: int, user: str, password: str, to: str) -> None:
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.to = to

    def send(self, subject: str, html: str, text: str) -> None:
        message = MIMEMultipart("alternative")
        message["Subject"] = subject
        message["From"] = self.user
        message["To"] = self.to
        message.attach(MIMEText(text, "plain"))
        message.attach(MIMEText(html, "html"))

        def _do_send() -> None:
            with smtplib.SMTP_SSL(self.host, self.port, timeout=30) as server:
                server.login(self.user, self.password)
                server.sendmail(self.user, [self.to], message.as_string())

        with_retry(_do_send, attempts=3, backoff_seconds=2.0)
        logger.info("email sent to %s", self.to)


class ConsoleNotifier:
    """--dry-run target: writes the rendered HTML to out/digest.html and logs a preview."""

    def __init__(self, out_dir: Path = Path("out")) -> None:
        self.out_dir = out_dir

    def send(self, subject: str, html: str, text: str) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        html_path = self.out_dir / "digest.html"
        text_path = self.out_dir / "digest.txt"
        html_path.write_text(html, encoding="utf-8")
        text_path.write_text(text, encoding="utf-8")
        logger.info("dry-run: subject=%r written to %s", subject, html_path)
