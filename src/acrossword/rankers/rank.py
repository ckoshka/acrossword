
from typing import List, Dict, Union, Optional, Tuple, Any, Callable, ClassVar
import numpy
import threading
from loguru import logger
import asyncio
import functools
import aiohttp
import asyncio
import pickle
import requests

def server_wrapper(func):
    '''Checks for a server at http://localhost:22647/. If there isn't one, it sets server_is_running to False, and it returns the function as per normal. Otherwise, it returns a function that takes the **kwargs and makes a post request to the server using the kwargs, and a field called 'method' with the name of the function. If that fails, it calls the function as per normal.'''
    async def interceptor_inner_wrapper(self, **kwargs):
        if self.is_server:
            self.server_is_running = False
            return await func(self, **kwargs)
        if self.server_is_running == False:
            return await func(self, **kwargs)
        if self.server_is_running == None:
            try:
                requests.post("http://localhost:22647/")
                self.server_is_running = True
                pass
            except requests.exceptions.ConnectionError:
                self.server_is_running = False
                return await func(self, **kwargs)
        async with aiohttp.ClientSession() as session:
            body = {"method": func.__name__, **kwargs}
            async with session.post("http://localhost:22647/", json=body) as resp:
                r = await resp.read()
        try:
            obj = pickle.loads(r)
        except pickle.UnpicklingError:
            self.server_is_running == False
            return await func(self, **kwargs)
        return obj
    interceptor_inner_wrapper.__doc__ = func.__doc__
    interceptor_inner_wrapper.__annotations__ = func.__annotations__
    interceptor_inner_wrapper.__name__ = func.__name__
    return interceptor_inner_wrapper


class Ranker:
    '''
    This class is responsible for providing a high-level interface for ranking sentences by their semantic similarity. It underpins a lot of this library's functionality. The problem is, when you load it for the first time, it will download a 450M language model to your drive. Every time you want to reload your chatbot, it will take around 11-12 seconds just to load mpnet-large. So you may want to start a server in the background via from acrossword import run; run().'''
    from sentence_transformers import SentenceTransformer, models
    from torch import Tensor

    # Imports occur inside the class because loading these libraries is extremely slow and this prevents quick iterative testing. Is this a good rationale? I don't know, I just find it annoying so you'll have to cope.
    __text_embeddings_cache__: Dict[str, Dict[str, Tensor]] = dict()
    __models_cache__: Dict[str, SentenceTransformer] = dict()

    server_is_running = None
    is_loading_model = False

    def __init__(
        self, #model_locations: List[str] = list(),
        default_model: str = "all-mpnet-base-v2",
        is_server: bool = False,
    ) -> None:
        self.default_model = default_model
        self.is_server = is_server
        #for model_name in model_locations:
            #threading.Thread(
                #target=self._load_model, args=[model_name]
            #).start()
            #self._load_model(model_name)

    @server_wrapper
    async def is_empty(self) -> bool:
        return len(self.__models_cache__) == 0

    def _download_model(self, model_name: str) -> None:
        logger.debug(f"Downloading model {model_name}")
        model_name = self._split_at_last_slash(model_name)
        if model_name in self.__models_cache__:
            return
        model = self.SentenceTransformer(model_name)
        logger.debug(f"Downloaded model {model_name}")
        model.max_seq_length = 512
        self.__models_cache__[model_name] = model
        self.__text_embeddings_cache__[model_name] = dict()
        self.is_loading_model = False

    def _load_from_file(self, model_name: str) -> None:
        word_embedding_model = self.models.Transformer(model_name)
        pooling_model = self.models.Pooling(
            word_embedding_model.get_word_embedding_dimension(), pooling_mode="mean"
        )
        model = self.SentenceTransformer(modules=[word_embedding_model, pooling_model])
        model.max_seq_length = 512
        self.__models_cache__[model_name] = model
        self.__text_embeddings_cache__[model_name] = dict()
        self.is_loading_model = False

    @server_wrapper
    async def add_model(self, model_name: str, from_file: bool = False) -> None:
        #if from_file:
            #self._load_from_file(model_name)
        #else:
            #self._download_model(model_name)
        self.is_loading_model = True
        if from_file:
            threading.Thread(target=self._load_from_file, args=[model_name]).start()
        else:
            threading.Thread(target=self._download_model, args=[model_name]).start()

    def _split_at_last_slash(self, model_name: str) -> str:
        if "/" in model_name:
            model_name = model_name.split("/")[-1]
        return model_name

    @server_wrapper
    async def convert(
        self, model_name: str, sentences: Union[List[str], Tuple[str, ...]]
    ) -> List[Tensor]:
        not_in_cache = dict()
        try:
            for i, sentence in enumerate(sentences):
                if sentence not in self.__text_embeddings_cache__[model_name]:
                    not_in_cache[i] = sentence
        except KeyError:
            if not self.is_loading_model:
                await self.add_model(model_name=model_name)
            while model_name not in self.__models_cache__ and self.is_loading_model:
                await asyncio.sleep(0.1)
            return await self.convert(model_name=model_name, sentences=sentences)
        if not_in_cache:
            embeddings = await self._convert(model_name=model_name, sentences=tuple(not_in_cache.values()))
            for i, embedding in enumerate(embeddings):
                position = list(not_in_cache.keys())[i]
                self.__text_embeddings_cache__[model_name][
                    not_in_cache[position]
                ] = embedding
        return [
            self.__text_embeddings_cache__[model_name][sentence]
            for sentence in sentences
        ]

    async def _convert(
        self, model_name: str, sentences: Tuple[str, ...]
    ) -> Union[List[Tensor], numpy.ndarray, Tensor]:
        model = self.__models_cache__[model_name]
        current_loop = asyncio.get_running_loop()
        embeddings = await current_loop.run_in_executor(None, functools.partial(model.encode, 
            list(sentences),
            convert_to_tensor=True,
            batch_size=5,
            show_progress_bar=True,
            normalize_embeddings=True,
        ))
        for i, embedding in enumerate(embeddings):
            self.__text_embeddings_cache__[model_name][sentences[i]] = embedding
        return embeddings

    @server_wrapper
    async def rank(
        self,
        texts: Tuple[str, ...],
        query: str,
        top_k: int,
        model: str,
        threshold: float = 0.1,
        return_none_if_below_threshold: bool = False,
    ) -> List[str]:
        import numpy

        if isinstance(texts, list):
            texts = tuple(texts)

        text_embeddings, query_embedding = await self.convert(model_name=model, sentences=texts), await self.convert(model_name=model, sentences=tuple([query]))

        return await self._rank(
            texts,
            text_embeddings,
            query_embedding[0],
            top_k,
            threshold,
            return_none_if_below_threshold,
        )

    @server_wrapper
    async def weighted_rank(
        self,
        texts: Tuple[str, ...],
        queries: Tuple[str],
        weights: Tuple[float],
        top_k: int,
        model: str,
        threshold: float = 0.1,
        return_none_if_below_threshold: bool = False,
    ) -> List[str]:
        import numpy

        if isinstance(texts, list):
            texts = tuple(texts)

        text_embeddings, query_embeddings = await self.convert(model_name=model, sentences=texts), await self.convert(model_name=model, sentences=queries)
        weighted_average_query = numpy.average(query_embeddings, axis=0, weights=weights)
        return await self._rank(
            texts,
            text_embeddings,
            weighted_average_query,
            top_k,
            threshold,
            return_none_if_below_threshold,
        )

    async def _rank(
        self,
        texts: Tuple[str, ...],
        text_embeddings: List[Tensor],
        query_embedding: Tensor,
        top_k: int,
        threshold: float,
        return_none_if_below_threshold: bool,
    ) -> List[str]:
    
        results = list()
        #logger.debug(f"Query embedding shape: {query_embedding.shape}")
        #logger.debug(f"Query embeddings: {query_embedding}")
        for i, tensor_embedding in enumerate(text_embeddings):
            similarity_tuple = (
                texts[i],
                numpy.dot(query_embedding, tensor_embedding),
            )
            if (similarity_tuple[1] >= threshold and return_none_if_below_threshold) or not return_none_if_below_threshold:
                results.append(similarity_tuple)

        results.sort(key=lambda similarity_tuple: similarity_tuple[1], reverse=True)

        return [result[0] for result in results][:top_k]

# Note that the above takes a very long time to load, so often users will have a localhost:5000 server with an instance of Ranker already running, accessible via json requests in the form:
#async with aiohttp.ClientSession() as session:
    #async with session.post("http://localhost:5000/", json={"method": "convert", "sentences": ["cats", "stones", "Jupiter"],  "model_name": "all-mpnet-base-v2"}) as resp:
        #r = await resp.read()
#obj = pickle.loads(r)

# So this module also creates a class decorator that sneakily intercepts all of its methods, checking for the existence of a localhost:5000 server, and if it exists, it uses it to do the work.