import hmac

from fastapi import Request
from fastapi.responses import JSONResponse

from app.schemas.settings import settings

_PUBLIC_PATHS = frozenset({"/health"})


async def require_api_key(request: Request, call_next):
    if request.method == "OPTIONS" or request.url.path in _PUBLIC_PATHS:
        return await call_next(request)

    expected = settings.api_secret
    if not expected:
        return JSONResponse(
            {"detail": "API_SECRET is not configured"},
            status_code=503,
        )

    provided = request.headers.get("x-api-key") or ""
    try:
        authorized = hmac.compare_digest(provided, expected)
    except (TypeError, ValueError):
        authorized = False
    if not authorized:
        return JSONResponse({"detail": "Unauthorized"}, status_code=401)
    return await call_next(request)
