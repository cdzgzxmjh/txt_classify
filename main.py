import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from abstract import get_abstract
from predict_single import get_news_type

app = FastAPI() # 创建 api 对象


class Request(BaseModel):
    title: str = ''
    article: str = ''


@app.get("/") # 根路由
def root():
    return {"Hello！！！"}


@app.post("/news")
def news(news_type_req: Request):
    print(news_type_req.title)
    return get_news_type(news_type_req.title)


@app.post("/abstract")
def news(abstract_req: Request):
    print(abstract_req.article)
    return get_abstract(abstract_req.article)


if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)
