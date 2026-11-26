"""Avatar Studio: the front end of this gateway.

It replaces the stock OpenAvatarChat page with an avatar library (create an avatar from a photo, manage it) and a conversation page per
avatar (/a/<id>). The conversation itself is still OpenAvatarChat's WebRTC endpoint; nothing of the engine is changed. launch.py calls
`attach()` once, before the engine starts.
"""
import os

from fastapi.staticfiles import StaticFiles
from loguru import logger

from .routes import Studio
from .settings import Settings

__all__ = ["attach", "mount"]


class PageFiles(StaticFiles):
    """The page's own scripts, styles and images. They are checked again on every load, so a new version is used at once."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response


def mount(app, settings=None, transports=None):
    """Add the Studio's pages, API and files to a FastAPI app. `transports` lets a test replace the HTTP connections to the other services."""
    settings = settings or Settings.from_env()
    studio = Studio(settings, transports)
    app.include_router(studio.router())
    app.mount("/ui", PageFiles(directory=settings.static_dir), name="studio-ui")
    logger.info(f"Studio: serving {settings.static_dir}; renderer {settings.renderer_url}, job runner {settings.worker_url}, "
                f"key {'required for changes' if settings.key else 'not set'}")
    return studio


def attach(client_module):
    """Make the engine's RTC client module register the Studio instead of the stock page.

    The client calls `register_frontend` while it sets up its routes. The Studio's routes are added first (the router answers in the
    order routes were added, so `/` and `/ui` are the Studio's), and the engine's own registration then only adds what the page needs from
    it: the `/openavatarchat/initconfig` endpoint with the WebRTC settings. It is pointed at the Studio's files so that it does not load
    or report the stock page.
    """
    from service.frontend_service import FrontendRegistrationOptions

    original = client_module.register_frontend
    studio_files = os.path.abspath(Settings.from_env().static_dir)

    def register_frontend(app, ui, parent_block, init_config, options=None):
        mount(app)
        original(app=app, ui=ui, parent_block=parent_block, init_config=init_config,
                 options=FrontendRegistrationOptions(frontend_dist_relative_path=studio_files))

    client_module.register_frontend = register_frontend
