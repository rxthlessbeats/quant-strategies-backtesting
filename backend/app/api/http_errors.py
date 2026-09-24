from fastapi import HTTPException


def public_http_detail(exc: BaseException, *, fallback: str) -> str:
    text = str(exc).strip()
    lowered = text.lower()
    if any(
        token in lowered
        for token in (
            "[sql:",
            "sqlite",
            "sqlalchemy",
            "operationalerror",
            "pendingrollback",
        )
    ):
        if "locked" in lowered or "busy" in lowered:
            return (
                "This ticker is being fetched for the first time. "
                "Please wait a moment."
            )
        return fallback
    if not text or "traceback" in lowered or len(text) > 180:
        return fallback
    return text


def http_502(exc: BaseException, *, fallback: str) -> HTTPException:
    return HTTPException(
        status_code=502, detail=public_http_detail(exc, fallback=fallback)
    )
