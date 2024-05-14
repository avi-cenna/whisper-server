import asyncio
import time

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def read_root():
    return {"Hello": "World"}


# @app.get("/items/{item_id}")
# async def read_item(item_id: int, q: str = None):
#     # async sleep here
#     # TODO: why isn't this working concurrently?
#     await asyncio.sleep(1)
#     print('after async sleep')
#     return {"item_id": item_id, "q": q}


@app.get("/items/{item_id}")
def read_item3(item_id: int, q: str = None):
    time.sleep(1)
    return {"item_id": item_id, "q": q}


@app.get("/error")
def error():
    time.sleep(.1)
    raise Exception("This is an exception")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
