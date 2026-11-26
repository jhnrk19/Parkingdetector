"""
Távoli AI parkolóhely-foglaltság detektáló szerver HTTP kliens modulja.

Tisztán vékony kliens hálózati réteg: kizárólag a Python standard könyvtárat
és a ``requests`` csomagot használja, semmilyen helyi gépi tanulási vagy
képfeldolgozó modult nem importál és nem futtat.
"""

from __future__ import annotations

import math
import mimetypes
import os
import re
import tempfile
import time
from typing import Any, Callable, Dict, Optional
from urllib.parse import urljoin, urlparse

import requests

DEFAULT_SERVER_HOST = "172.30.73.194"
DEFAULT_SERVER_PORT = 8000

ProgressCallback = Callable[[int, int, int, int], None]

_STATS_KEYS = ("total_slots", "final_occupied", "final_free", "frames_processed")
_JOB_ID_RE = re.compile(r"[A-Za-z0-9._-]+")


class RemoteClientError(Exception):
    """A távoli AI szerverrel való kommunikáció során fellépő általános hiba."""


class ServerUnreachableError(RemoteClientError):
    """A távoli AI szerver nem érhető el (kapcsolódási hiba, DNS-hiba vagy időtúllépés)."""


class ServerResponseError(RemoteClientError):
    """A távoli AI szerver hibás HTTP-státusszal, hibás JSON-nal vagy sikertelen feladattal válaszolt."""


def _is_non_bool_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _to_nonneg_int(value: Any, default: int = 0) -> int:
    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return default
        return max(0, int(value))
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return default
        try:
            parsed = float(stripped)
            if not math.isfinite(parsed):
                return default
            return max(0, int(parsed))
        except ValueError:
            return default
    return default


class ParkingRemoteClient:
    """HTTP kliens a távoli parkolóhely-elemző AI szerver eléréséhez."""

    def __init__(
        self,
        host: str = DEFAULT_SERVER_HOST,
        port: int | str = DEFAULT_SERVER_PORT,
        timeout: float = 30.0,
        poll_interval: float = 0.5,
        max_wait: float = 1800.0,
    ) -> None:
        self.host = self._normalize_host(host)
        self.port = self._normalize_port(port)
        self.timeout = self._normalize_positive_float(timeout, "timeout")
        self.poll_interval = self._normalize_positive_float(poll_interval, "poll_interval")
        self.max_wait = self._normalize_positive_float(max_wait, "max_wait")

    @staticmethod
    def _normalize_host(host: Any) -> str:
        if not isinstance(host, str):
            raise ValueError("Érvénytelen szerver cím (host): szöveges érték szükséges.")
        cleaned = host.strip()
        lower = cleaned.lower()
        if lower.startswith("https://"):
            raise ValueError(
                "Érvénytelen szerver cím (host): csak HTTP protokoll támogatott ('https://' nem engedélyezett)."
            )
        if lower.startswith("http://"):
            cleaned = cleaned[7:]
        cleaned = cleaned.rstrip("/")
        if (
            not cleaned
            or any(ch.isspace() for ch in cleaned)
            or "/" in cleaned
            or any(ch in cleaned for ch in (":", "?", "#", "@", "\\"))
        ):
            raise ValueError(f"Érvénytelen szerver cím (host): {host!r}")
        return cleaned

    @staticmethod
    def _normalize_port(port: Any) -> int:
        if isinstance(port, bool):
            raise ValueError("Érvénytelen port szám (1–65535 közötti egész szám szükséges).")
        if isinstance(port, str):
            stripped = port.strip()
            if not stripped or not stripped.isascii() or not stripped.isdigit():
                raise ValueError("Érvénytelen port szám (1–65535 közötti egész szám szükséges).")
            parsed = int(stripped)
        elif isinstance(port, int):
            parsed = port
        else:
            raise ValueError("Érvénytelen port szám (1–65535 közötti egész szám szükséges).")

        if not (1 <= parsed <= 65535):
            raise ValueError("Érvénytelen port szám (1–65535 közötti egész szám szükséges).")
        return parsed

    @staticmethod
    def _normalize_positive_float(value: Any, name: str) -> float:
        if isinstance(value, bool):
            raise ValueError(f"Érvénytelen {name} érték: pozitív szám szükséges.")
        try:
            num = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Érvénytelen {name} érték: pozitív szám szükséges.") from exc
        if not math.isfinite(num) or num <= 0:
            raise ValueError(f"Érvénytelen {name} érték: pozitív véges szám szükséges.")
        return num

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _map_request_exception(self, exc: requests.RequestException, action: str) -> RemoteClientError:
        if isinstance(exc, requests.exceptions.Timeout):
            return ServerUnreachableError(
                f"Időtúllépés a távoli AI szerverrel való kommunikáció során "
                f"({action}, {self.base_url}, időkorlát: {self.timeout:.1f}s). "
                f"Helyi feldolgozás nem futtatható."
            )
        if isinstance(exc, requests.exceptions.ConnectionError):
            msg_lower = str(exc).lower()
            dns_markers = (
                "name or service not known",
                "nodename",
                "getaddrinfo",
                "nameresolutionerror",
                "gaierror",
                "failed to resolve",
            )
            if any(marker in msg_lower for marker in dns_markers):
                return ServerUnreachableError(
                    f"A távoli AI szerver címe nem oldható fel (DNS-hiba): {self.host}. "
                    f"Ellenőrizd az IP-címet vagy hostnevet!"
                )
            return ServerUnreachableError(
                f"A távoli AI szerver nem érhető el ({self.base_url}). "
                f"Ellenőrizd, hogy a szerver fut-e és elérhető-e a hálózaton!"
            )
        return ServerResponseError(f"Hálózati adatátviteli hiba ({action}): {exc}")

    @staticmethod
    def _extract_http_error_detail(response: requests.Response) -> str:
        try:
            payload = response.json()
            if isinstance(payload, dict):
                for key in ("error", "message", "detail"):
                    val = payload.get(key)
                    if val is not None and str(val).strip():
                        return str(val).strip()
        except ValueError:
            pass
        text = (response.text or "").strip()
        if text:
            return text[:300]
        if response.reason:
            return str(response.reason)
        return "Ismeretlen szerverhiba"

    @staticmethod
    def _parse_json_dict(response: requests.Response, endpoint: str) -> Dict[str, Any]:
        try:
            data = response.json()
        except (ValueError, requests.exceptions.JSONDecodeError) as exc:
            raise ServerResponseError(
                f"Érvénytelen JSON válasz érkezett a szervertől ({endpoint})."
            ) from exc
        if not isinstance(data, dict):
            raise ServerResponseError(
                f"Váratlan válaszformátum ({endpoint}): JSON objektum szükséges."
            )
        return data

    @staticmethod
    def _validate_io_paths(
        video_path: Any,
        output_path: Any,
        require_output: bool = True,
    ) -> tuple[str, Optional[str]]:
        if not isinstance(video_path, str) or not video_path.strip():
            raise RemoteClientError("A bemeneti videófájl elérési útja üres vagy érvénytelen.")
        cleaned_video = video_path.strip()
        if not os.path.isfile(cleaned_video):
            raise RemoteClientError(f"A kiválasztott videófájl nem található: {cleaned_video}")
        try:
            if os.path.getsize(cleaned_video) <= 0:
                raise RemoteClientError(f"A kiválasztott videófájl üres (0 bájt): {cleaned_video}")
        except OSError as exc:
            raise RemoteClientError(
                f"Nem sikerült ellenőrizni a videófájl méretét ({cleaned_video}): {exc}"
            ) from exc

        if output_path is None and not require_output:
            return cleaned_video, None
        if not isinstance(output_path, str) or not output_path.strip():
            raise RemoteClientError("A kimeneti videófájl elérési útja üres vagy érvénytelen.")
        cleaned_output = output_path.strip()
        if os.path.isdir(cleaned_output):
            raise RemoteClientError(
                f"A kimeneti elérési út egy könyvtár, érvényes fájlnév szükséges: {cleaned_output}"
            )
        return cleaned_video, cleaned_output

    @staticmethod
    def _validate_job_id(job_id: Any) -> str:
        if not isinstance(job_id, str):
            raise ServerResponseError("Hiányzó vagy érvénytelen 'job_id' mező a szerver válaszában.")
        if (
            not job_id
            or job_id in (".", "..")
            or ".." in job_id
            or not _JOB_ID_RE.fullmatch(job_id)
        ):
            raise ServerResponseError(f"Érvénytelen feladat-azonosító (job_id) a szerver válaszában: {job_id!r}")
        return job_id

    @staticmethod
    def _extract_payload_error(data: Dict[str, Any], default: str) -> str:
        for key in ("error", "message"):
            val = data.get(key)
            if val is not None:
                cleaned = str(val).strip()
                if cleaned:
                    return cleaned
        return default

    @staticmethod
    def _format_duration(seconds: float) -> str:
        if seconds < 1.0:
            return f"{seconds:g}s"
        if seconds < 10.0 and not seconds.is_integer():
            return f"{seconds:.1f}s"
        return f"{seconds:.0f}s"

    @staticmethod
    def _validate_stats(stats: Any) -> Dict[str, int]:
        if not isinstance(stats, dict):
            raise ServerResponseError(
                "Hiányzó vagy érvénytelen 'stats' mező a befejezett szerveroldali feladat válaszában."
            )
        validated: Dict[str, int] = {}
        for key in _STATS_KEYS:
            val = stats.get(key)
            if not _is_non_bool_int(val) or val < 0:
                raise ServerResponseError(
                    f"Hiányzó vagy érvénytelen statisztikai mező a szerver válaszában: {key!r}"
                )
            validated[key] = val
        return validated

    def check_health(self) -> Dict[str, Any]:
        """Ellenőrzi a távoli AI szerver elérhetőségét és állapotát (`GET /api/health`)."""
        url = f"{self.base_url}/api/health"
        try:
            response = requests.get(url, timeout=self.timeout, allow_redirects=False)
        except requests.RequestException as exc:
            raise self._map_request_exception(exc, "GET /api/health") from exc

        if 300 <= response.status_code < 400:
            raise ServerResponseError(
                f"A szerver nem engedélyezett átirányítással (HTTP {response.status_code}) válaszolt (/api/health)."
            )
        if response.status_code != 200:
            detail = self._extract_http_error_detail(response)
            raise ServerResponseError(
                f"A szerver állapotellenőrzése sikertelen (HTTP {response.status_code}): {detail}"
            )

        data = self._parse_json_dict(response, "/api/health")
        if "status" not in data or data.get("status") is None or not str(data.get("status")).strip():
            raise ServerResponseError("Hiányzó 'status' mező a /api/health válaszból.")
        return data

    def run_training(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Dict[str, Any]:
        """Feltölti a tanító videót (`POST /api/train`), követi a feldolgozást és opcionálisan letölti a videót."""
        clean_video, clean_output = self._validate_io_paths(
            video_path, output_path, require_output=False
        )
        mime_type = mimetypes.guess_type(clean_video)[0] or "video/mp4"
        url = f"{self.base_url}/api/train"

        try:
            with open(clean_video, "rb") as video_file:
                response = requests.post(
                    url,
                    files={"video": (os.path.basename(clean_video), video_file, mime_type)},
                    timeout=self.timeout,
                    allow_redirects=False,
                )
        except requests.RequestException as exc:
            raise self._map_request_exception(exc, "POST /api/train") from exc
        except OSError as exc:
            raise RemoteClientError(
                f"Nem sikerült megnyitni a tanító videófájlt ({clean_video}): {exc}"
            ) from exc

        if 300 <= response.status_code < 400:
            raise ServerResponseError(
                f"A szerver nem engedélyezett átirányítással (HTTP {response.status_code}) válaszolt (/api/train)."
            )
        if response.status_code not in (200, 202):
            detail = self._extract_http_error_detail(response)
            raise ServerResponseError(
                f"A tanító videó feltöltése sikertelen (HTTP {response.status_code}): {detail}"
            )

        submit_data = self._parse_json_dict(response, "/api/train")
        if submit_data.get("status") == "failed":
            err = self._extract_payload_error(submit_data, "Ismeretlen szerverhiba")
            raise ServerResponseError(f"A tanítási feladat indítása sikertelen: {err}")

        clean_job_id = self._validate_job_id(submit_data.get("job_id"))
        session_id = submit_data.get("session_id")
        if not isinstance(session_id, str) or not session_id.strip():
            raise ServerResponseError("Hiányzó 'session_id' mező a /api/train válaszából.")

        job_data = self._poll_job(clean_job_id, progress_callback)
        saved_path: Optional[str] = None
        if clean_output is not None:
            video_url = self._resolve_video_url(clean_job_id, job_data.get("result_video_url"))
            saved_path = self._download_video(video_url, clean_output)
        return self._build_result_dict(job_data, fallback_session_id=session_id.strip(), saved_path=saved_path)

    def run_analysis(
        self,
        video_path: str,
        output_path: str,
        session_id: str,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Dict[str, Any]:
        """Feltölti a tesztelő videót (`POST /api/analyze`) a munkamenet-azonosítóval, és letölti az eredményt."""
        if not isinstance(session_id, str) or not session_id.strip():
            raise RemoteClientError(
                "Hiányzó vagy üres munkamenet-azonosító (session_id). Először futtasd a tanítást (1. lépés)!"
            )
        clean_sid = session_id.strip()
        clean_video, clean_output = self._validate_io_paths(video_path, output_path)
        mime_type = mimetypes.guess_type(clean_video)[0] or "video/mp4"
        url = f"{self.base_url}/api/analyze"

        try:
            with open(clean_video, "rb") as video_file:
                response = requests.post(
                    url,
                    data={"session_id": clean_sid},
                    files={"video": (os.path.basename(clean_video), video_file, mime_type)},
                    timeout=self.timeout,
                    allow_redirects=False,
                )
        except requests.RequestException as exc:
            raise self._map_request_exception(exc, "POST /api/analyze") from exc
        except OSError as exc:
            raise RemoteClientError(
                f"Nem sikerült megnyitni a teszt videófájlt ({clean_video}): {exc}"
            ) from exc

        if 300 <= response.status_code < 400:
            raise ServerResponseError(
                f"A szerver nem engedélyezett átirányítással (HTTP {response.status_code}) válaszolt (/api/analyze)."
            )
        if response.status_code not in (200, 202):
            detail = self._extract_http_error_detail(response)
            raise ServerResponseError(
                f"A teszt videó beküldése sikertelen (HTTP {response.status_code}): {detail}"
            )

        submit_data = self._parse_json_dict(response, "/api/analyze")
        if submit_data.get("status") == "failed":
            err = self._extract_payload_error(submit_data, "Ismeretlen szerverhiba")
            raise ServerResponseError(f"Az elemzési feladat indítása sikertelen: {err}")

        clean_job_id = self._validate_job_id(submit_data.get("job_id"))
        resp_session_id = submit_data.get("session_id")
        if not isinstance(resp_session_id, str) or not resp_session_id.strip():
            resp_session_id = clean_sid

        job_data = self._poll_job(clean_job_id, progress_callback)
        video_url = self._resolve_video_url(clean_job_id, job_data.get("result_video_url"))
        saved_path = self._download_video(video_url, clean_output)
        return self._build_result_dict(
            job_data, fallback_session_id=resp_session_id.strip(), saved_path=saved_path
        )

    def _poll_job(
        self,
        job_id: str,
        progress_callback: Optional[ProgressCallback],
    ) -> Dict[str, Any]:
        clean_job_id = self._validate_job_id(job_id)
        url = f"{self.base_url}/api/jobs/{clean_job_id}"
        start_time = time.monotonic()
        last_emitted: Optional[tuple[str, int, int, int, int]] = None

        while True:
            remaining = self.max_wait - (time.monotonic() - start_time)
            if remaining <= 0:
                raise ServerUnreachableError(
                    f"A szerveroldali feldolgozás túllépte a maximális várakozási időt "
                    f"({self._format_duration(self.max_wait)})."
                )
            try:
                response = requests.get(
                    url,
                    timeout=min(self.timeout, remaining),
                    allow_redirects=False,
                )
            except requests.RequestException as exc:
                if (
                    isinstance(exc, requests.exceptions.Timeout)
                    and (time.monotonic() - start_time) >= self.max_wait
                ):
                    raise ServerUnreachableError(
                        f"A szerveroldali feldolgozás túllépte a maximális várakozási időt "
                        f"({self._format_duration(self.max_wait)})."
                    ) from exc
                raise self._map_request_exception(exc, f"GET /api/jobs/{clean_job_id}") from exc

            if 300 <= response.status_code < 400:
                raise ServerResponseError(
                    f"A szerver nem engedélyezett átirányítással (HTTP {response.status_code}) "
                    f"válaszolt (/api/jobs/{clean_job_id})."
                )
            if response.status_code != 200:
                detail = self._extract_http_error_detail(response)
                raise ServerResponseError(
                    f"A feldolgozási állapot lekérdezése sikertelen (HTTP {response.status_code}): {detail}"
                )

            data = self._parse_json_dict(response, f"/api/jobs/{clean_job_id}")
            status = data.get("status")

            if status == "failed":
                err_detail = self._extract_payload_error(data, "Ismeretlen feldolgozási hiba")
                raise ServerResponseError(f"A szerveroldali feldolgozás sikertelen volt: {err_detail}")

            if status == "completed":
                stats = self._validate_stats(data.get("stats"))
                total = _to_nonneg_int(data.get("total_frames"), default=0)
                cur = _to_nonneg_int(data.get("current_frame"), default=0)
                if total == 0 and stats["frames_processed"] > 0:
                    total = stats["frames_processed"]
                    cur = stats["frames_processed"]
                elif total > 0 and cur == 0:
                    cur = total
                occ = (
                    _to_nonneg_int(data.get("occupied"), default=stats["final_occupied"])
                    if data.get("occupied") is not None
                    else stats["final_occupied"]
                )
                free = (
                    _to_nonneg_int(data.get("free"), default=stats["final_free"])
                    if data.get("free") is not None
                    else stats["final_free"]
                )
                snapshot = ("completed", cur, total, occ, free)
                if progress_callback is not None and snapshot != last_emitted:
                    progress_callback(cur, total, occ, free)
                return data

            if status in ("queued", "processing"):
                cur = _to_nonneg_int(data.get("current_frame"), default=0)
                total = _to_nonneg_int(data.get("total_frames"), default=0)
                occ = _to_nonneg_int(data.get("occupied"), default=0)
                free = _to_nonneg_int(data.get("free"), default=0)
                if cur == 0 and total == 0:
                    raw_progress = data.get("progress")
                    if (
                        raw_progress is not None
                        and not isinstance(raw_progress, bool)
                        and isinstance(raw_progress, (int, float))
                        and math.isfinite(raw_progress)
                        and raw_progress > 0
                    ):
                        cur = min(100, int(raw_progress))
                        total = 100

                snapshot = (str(status), cur, total, occ, free)
                if progress_callback is not None and snapshot != last_emitted:
                    progress_callback(cur, total, occ, free)
                    last_emitted = snapshot

                elapsed = time.monotonic() - start_time
                remaining = self.max_wait - elapsed
                if remaining <= 0:
                    raise ServerUnreachableError(
                        f"A szerveroldali feldolgozás túllépte a maximális várakozási időt "
                        f"({self._format_duration(self.max_wait)})."
                    )
                time.sleep(min(self.poll_interval, remaining))
                if (time.monotonic() - start_time) >= self.max_wait or remaining <= self.poll_interval:
                    raise ServerUnreachableError(
                        f"A szerveroldali feldolgozás túllépte a maximális várakozási időt "
                        f"({self._format_duration(self.max_wait)})."
                    )
                continue

            raise ServerResponseError(f"Ismeretlen feladatállapot a szervertől: {status!r}")

    def _resolve_video_url(self, job_id: str, raw_url: Any) -> str:
        clean_job_id = self._validate_job_id(job_id)
        fallback_path = f"/api/jobs/{clean_job_id}/video"
        candidate = raw_url.strip() if isinstance(raw_url, str) and raw_url.strip() else fallback_path
        resolved = urljoin(f"{self.base_url}/", candidate)
        parsed = urlparse(resolved)
        expected_netloc = urlparse(self.base_url).netloc
        if (
            parsed.scheme != "http"
            or parsed.netloc != expected_netloc
            or not parsed.path.startswith("/")
        ):
            raise ServerResponseError(
                f"Érvénytelen vagy idegen tartományú videó letöltési URL a szerver válaszában: {resolved!r}"
            )
        return resolved

    def _download_video(self, video_url: str, output_path: str) -> str:
        abs_out = os.path.abspath(output_path)
        parent_dir = os.path.dirname(abs_out) or os.getcwd()
        try:
            os.makedirs(parent_dir, exist_ok=True)
        except OSError as exc:
            raise RemoteClientError(
                f"Nem sikerült létrehozni a kimeneti könyvtárat ({parent_dir}): {exc}"
            ) from exc

        tmp_file = None
        tmp_path: Optional[str] = None
        bytes_written = 0

        try:
            with requests.get(
                video_url, stream=True, timeout=self.timeout, allow_redirects=False
            ) as response:
                if 300 <= response.status_code < 400:
                    raise ServerResponseError(
                        f"A feldolgozott videó letöltése nem engedélyezett átirányítást kapott "
                        f"(HTTP {response.status_code})."
                    )
                if response.status_code != 200:
                    detail = self._extract_http_error_detail(response)
                    raise ServerResponseError(
                        f"A feldolgozott videó letöltése sikertelen (HTTP {response.status_code}): {detail}"
                    )

                raw_len = (response.headers.get("Content-Length") or "").strip()
                expected_len: Optional[int] = int(raw_len) if raw_len.isdigit() else None

                tmp_file = tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=parent_dir,
                    prefix=".tmp_dl_",
                    suffix=".mp4",
                    delete=False,
                )
                tmp_path = tmp_file.name

                for chunk in response.iter_content(chunk_size=65536):
                    if chunk:
                        tmp_file.write(chunk)
                        bytes_written += len(chunk)

                tmp_file.close()
                tmp_file = None

                if bytes_written <= 0:
                    raise ServerResponseError("A letöltött eredményvideó üres (0 bájt).")
                if expected_len is not None and bytes_written != expected_len:
                    raise ServerResponseError(
                        f"Hiányos videóletöltés: {bytes_written}/{expected_len} bájt érkezett."
                    )

                os.replace(tmp_path, abs_out)
                tmp_path = None
                return abs_out

        except RemoteClientError:
            raise
        except requests.RequestException as exc:
            raise self._map_request_exception(exc, "eredményvideó letöltése") from exc
        except OSError as exc:
            raise RemoteClientError(
                f"Nem sikerült elmenteni a letöltött videót ({abs_out}): {exc}"
            ) from exc
        finally:
            if tmp_file is not None:
                try:
                    tmp_file.close()
                except OSError:
                    pass
            if tmp_path is not None and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    @staticmethod
    def _build_result_dict(
        job_data: Dict[str, Any],
        fallback_session_id: str,
        saved_path: Optional[str],
    ) -> Dict[str, Any]:
        stats = ParkingRemoteClient._validate_stats(job_data.get("stats"))
        raw_sid = job_data.get("session_id")
        session_id = raw_sid.strip() if isinstance(raw_sid, str) and raw_sid.strip() else fallback_session_id
        return {
            "session_id": session_id,
            "total_slots": stats["total_slots"],
            "final_occupied": stats["final_occupied"],
            "final_free": stats["final_free"],
            "frames_processed": stats["frames_processed"],
            "output_path": saved_path,
        }
