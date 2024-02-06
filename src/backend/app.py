from typing import Union

from fastapi import FastAPI

from src.models.predict import predict

app = FastAPI()


@app.get("/highlights/{video_id}")
async def get_highlights(video_id: Union[int, str]):
    if isinstance(video_id, int):
        video_id = str(video_id)
    highlight_intervals = await predict(video_id)

    return highlight_intervals
