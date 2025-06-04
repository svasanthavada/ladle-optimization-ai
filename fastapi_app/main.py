
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi_app.api_optimize import optimize_alloys

app = FastAPI()

@app.post("/optimize")
def optimize(file: UploadFile = File(...)):
    result = optimize_alloys(file)
    return JSONResponse(content=result)
