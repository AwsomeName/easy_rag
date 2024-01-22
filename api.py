from fastapi import FastAPI, Request
import uvicorn, json
from RAG import KnowLedge

app = FastAPI()

@app.post("/api/v1/test")
async def create_item(request: Request):
    global kl
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    input_str = json_post_list.get('input_str')
    output_str, output_df = kl.search_result(input_str)
    return output_str, output_df

    
if __name__ == '__main__':
    kl = KnowLedge(
        global_dir="data_pdf/data1",
        gen_model_name_or_path="models/chatglm3-6b-32k",
        sen_embedding_model_name_or_path="models/chinese-roberta-wwm-ext")
    uvicorn.run(app, host='0.0.0.0', port=11073, workers=1)