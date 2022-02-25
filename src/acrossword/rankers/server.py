from ..rankers.rank import Ranker
ranker = Ranker(is_server=True)
from quart import Quart, render_template, websocket, request, send_from_directory, make_response, Response
import pickle
# We want to avoid annoying serialisation and deserialisation of numpy ndarrays, so we'll use pickle
app = Quart(__name__)

# Since this is internal, and we don't have to worry about security, we'll accept a Dict[Any, Any] as a POST request, where one of the fields must be called "method"

@app.post("/")
async def handle_post():
    data = await request.get_json()
    method = data["method"]
    # We then call eval using the method, and expand the remainder of the data into the function's arguments
    del data["method"]
    result = await eval(f"ranker.{method}")(**data)
    pickled = pickle.dumps(result)
    return Response(pickled)

@app.post("/shutdown")
async def shutdown():
    await app.shutdown()

def run():
    '''This will create a background server. It's useful if you frequently 
    need to restart whatever project you're working on and you're finding the 
    time spent reloading SentenceTransformers is too much'''
    app.run(host="localhost", port=22647, use_reloader=False)