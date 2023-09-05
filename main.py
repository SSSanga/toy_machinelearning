# conda install -c conda-forge fastapi uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

#  script 오류남
# No 'Access-Control-Allow-Origin'_해결을 위해 추가 import가 필요함
# CORS 설정
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"],  # 실제 운영 환경에서는 접근 가능한 도메인만 허용하는 것이 좋습니다.
    allow_methods=["*"],
    allow_headers=["*"],
)



## preprocess에서 했던 minmaxscaler pkl 불러오기
@app.post()


## bestmodel pkl 불러오기

@app.post()