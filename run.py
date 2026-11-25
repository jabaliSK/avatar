import logging
import os

import uvicorn

from avatar import config

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        "avatar.app:create_app", factory=True,
        host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")),
        workers=1,
    )
