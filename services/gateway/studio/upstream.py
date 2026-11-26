"""The services the Studio talks to over HTTP: the renderer's admin API, the job runner and the LLM server."""
import httpx
from fastapi import HTTPException
from loguru import logger


def detail_of(response):
    """The message of an error response from one of those services (they answer {"detail": "..."})."""
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        detail = None
    return detail[:300] if isinstance(detail, str) and detail else "The request was not accepted."


class Upstream:
    def __init__(self, name, base_url, timeout=8.0, transport=None):
        self.name = name
        self.client = httpx.AsyncClient(base_url=base_url, timeout=httpx.Timeout(timeout, connect=3.0), transport=transport)

    async def call(self, method, path, **kwargs):
        """The response of a request that succeeded; anything else becomes an HTTPException the browser can show."""
        try:
            response = await self.client.request(method, path, **kwargs)
        except httpx.TimeoutException:
            raise HTTPException(504, f"The {self.name} did not answer in time.") from None
        except httpx.HTTPError:
            raise HTTPException(503, f"The {self.name} is not reachable.") from None
        if response.status_code >= 500:  # the service's own error text is for the log, not for the browser
            logger.warning(f"Studio: the {self.name} answered {response.status_code} to {method} {path}: {detail_of(response)}")
            raise HTTPException(502, f"The {self.name} reported an error.")
        if response.status_code >= 400:
            raise HTTPException(response.status_code, detail_of(response))
        return response

    async def json(self, method, path, **kwargs):
        return (await self.call(method, path, **kwargs)).json()

    async def probe(self, path, timeout=2.0):
        """(True, the parsed body or None) when the service answers with a success status in time, else (False, None)."""
        try:
            response = await self.client.get(path, timeout=timeout)
        except httpx.HTTPError:
            return False, None
        if response.status_code >= 400:
            return False, None
        try:
            return True, response.json()
        except ValueError:
            return True, None
