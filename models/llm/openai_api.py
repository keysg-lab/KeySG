"""OpenAI GPT API interface for text, vision, and structured outputs."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Iterator, List, Optional, Type, Union

import numpy as np
import openai
from PIL import Image
from pydantic import BaseModel
from dotenv import load_dotenv

from models.llm._common import encode_image_data_url

load_dotenv()


class GPTInterface:
    """Interface for OpenAI GPT API supporting text, vision, structured outputs, and embeddings."""

    def __init__(self, client: Optional[openai.OpenAI] = None):
        self.client = client or openai.OpenAI()

    def _encode_image(self, image: Union[np.ndarray, Image.Image], format: str = "jpeg") -> str:
        """Encode image to data URL."""
        return encode_image_data_url(image, format=format)

    def _prepare_common_kwargs(
        self,
        instructions: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        verbosity: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Prepare common API arguments."""
        api_kwargs = {}
        if instructions:
            api_kwargs["instructions"] = instructions
        if reasoning_effort:
            api_kwargs["reasoning"] = {"effort": reasoning_effort}
        if verbosity:
            api_kwargs["text"] = {"verbosity": verbosity}
        api_kwargs.update(kwargs)
        return api_kwargs

    def text_prompt(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        model: str = "gpt-5-mini",
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[str, Iterator[str]]:
        """Generate text response from a prompt."""
        api_kwargs = self._prepare_common_kwargs(**kwargs)
        api_kwargs.update({"model": model, "input": prompt})

        if stream:
            response_stream = self.client.responses.create(stream=True, **api_kwargs)
            return (
                chunk.delta
                for chunk in response_stream
                if chunk.type == "response.output_text.delta"
            )

        response = self.client.responses.create(**api_kwargs)
        return response.output_text

    def vision_prompt(
        self,
        prompt: str,
        image: Union[str, np.ndarray, Image.Image, List[Union[str, np.ndarray, Image.Image]]],
        model: str = "gpt-5-mini",
        detail: str = "auto",
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[str, Iterator[str]]:
        """Analyze image(s) and generate text response."""
        api_kwargs = self._prepare_common_kwargs(**kwargs)

        images = image if isinstance(image, list) else [image]
        image_contents = [
            {
                "type": "input_image",
                "image_url": img if isinstance(img, str) else self._encode_image(img),
                "detail": detail,
            }
            for img in images
        ]

        content = [{"type": "input_text", "text": prompt}, *image_contents]
        api_kwargs.update({"model": model, "input": [{"role": "user", "content": content}]})

        if stream:
            response_stream = self.client.responses.create(stream=True, **api_kwargs)
            return (
                chunk.delta
                for chunk in response_stream
                if chunk.type == "response.output_text.delta"
            )

        response = self.client.responses.create(**api_kwargs)
        return response.output_text

    def structured_prompt(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        response_model: Type[BaseModel],
        model: str = "gpt-5-mini",
        stream: bool = False,
        image: Optional[Union[str, np.ndarray, Image.Image, List]] = None,
        detail: str = "auto",
        **kwargs: Any,
    ) -> Union[BaseModel, Iterator[Any]]:
        """Generate structured JSON response conforming to a Pydantic model."""
        api_kwargs = self._prepare_common_kwargs(**kwargs)

        final_input: Union[str, List[Dict[str, Any]]]
        if image:
            images = image if isinstance(image, list) else [image]
            image_contents = [
                {
                    "type": "input_image",
                    "image_url": img if isinstance(img, str) else self._encode_image(img),
                    "detail": detail,
                }
                for img in images
            ]

            if isinstance(prompt, str):
                content = [{"type": "input_text", "text": prompt}, *image_contents]
                final_input = [{"role": "user", "content": content}]
            else:
                final_input = [*prompt, {"role": "user", "content": image_contents}]
        else:
            final_input = prompt

        api_kwargs.update({
            "model": model,
            "input": final_input,
            "text_format": response_model,
        })

        if stream:
            return self.client.responses.stream(**api_kwargs)

        response = self.client.responses.parse(**api_kwargs)
        return response.output_parsed

    async def _process_one_structured_prompt(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        response_model: Type[BaseModel],
        model: str,
        image: Optional[Union[str, np.ndarray, Image.Image, List]],
        detail: str,
        **kwargs: Any,
    ) -> BaseModel:
        """Helper for async structured prompt processing."""
        api_kwargs = self._prepare_common_kwargs(**kwargs)

        final_input: Union[str, List[Dict[str, Any]]]
        if image:
            images = image if isinstance(image, list) else [image]
            image_contents = [
                {
                    "type": "input_image",
                    "image_url": img if isinstance(img, str) else self._encode_image(img),
                    "detail": detail,
                }
                for img in images
            ]

            if isinstance(prompt, str):
                final_input = [{
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}, *image_contents],
                }]
            else:
                final_input = [*prompt, {"role": "user", "content": image_contents}]
        else:
            final_input = prompt

        api_kwargs.update({
            "model": model,
            "input": final_input,
            "text_format": response_model,
        })

        response = await asyncio.to_thread(self.client.responses.parse, **api_kwargs)
        return response.output_parsed

    async def structured_prompt_batch(
        self,
        prompts: List[Union[str, List[Dict[str, Any]]]],
        response_model: Type[BaseModel],
        model: str = "gpt-5-mini",
        images: Optional[List[Optional[Union[str, np.ndarray, Image.Image, list]]]] = None,
        detail: str = "auto",
        **kwargs: Any,
    ) -> List[Union[BaseModel, Exception]]:
        """Process multiple structured prompts concurrently."""
        if images and len(prompts) != len(images):
            raise ValueError("Number of prompts must match number of image entries.")

        tasks = [
            self._process_one_structured_prompt(
                prompt=prompt,
                response_model=response_model,
                model=model,
                image=images[i] if images else None,
                detail=detail,
                **kwargs,
            )
            for i, prompt in enumerate(prompts)
        ]

        if not tasks:
            return []
        return await asyncio.gather(*tasks, return_exceptions=True)

    def embed_text(
        self,
        input_text: Union[str, List[str]],
        model: str = "text-embedding-3-small",
        *,
        normalize: bool = False,
        dimensions: Optional[int] = None,
        encoding_format: str = "float",
        return_usage: bool = False,
        max_total_tokens: int = 300_000,
        split_aggregate: str = "mean",
    ) -> Union[List[float], List[List[float]], Dict[str, Any]]:
        """Create vector embeddings for text strings."""
        try:
            import tiktoken
            _enc_cache: Dict[str, Any] = {}

            def count_tokens(text: str) -> int:
                enc = _enc_cache.get(model)
                if enc is None:
                    try:
                        enc = tiktoken.encoding_for_model(model)
                    except Exception:
                        enc = tiktoken.get_encoding("cl100k_base")
                    _enc_cache[model] = enc
                return len(enc.encode(text))
        except Exception:
            def count_tokens(text: str) -> int:
                return max(1, len(text) // 4)

        single_input = isinstance(input_text, str)
        original_inputs = [input_text] if single_input else list(input_text)
        cleaned_inputs = [s.replace("\n", " ") for s in original_inputs]

        def split_large_text(text: str) -> List[str]:
            tokens = count_tokens(text)
            if tokens <= max_total_tokens:
                return [text]
            parts = [p.strip() for p in text.replace("?", ".").replace("!", ".").split(".") if p.strip()]
            chunks: List[str] = []
            cur: List[str] = []
            cur_tokens = 0
            for sent in parts:
                sent_tokens = count_tokens(sent)
                if sent_tokens > max_total_tokens:
                    approx_chars = max_total_tokens * 4
                    for i in range(0, len(sent), approx_chars):
                        chunks.append(sent[i : i + approx_chars])
                    continue
                if cur_tokens + sent_tokens <= max_total_tokens:
                    cur.append(sent)
                    cur_tokens += sent_tokens
                else:
                    chunks.append(". ".join(cur))
                    cur = [sent]
                    cur_tokens = sent_tokens
            if cur:
                chunks.append(". ".join(cur))
            return chunks

        expanded_inputs: List[str] = []
        original_to_chunk_indices: List[List[int]] = []
        for text in cleaned_inputs:
            chunks = split_large_text(text)
            idx_list = [len(expanded_inputs) + i for i in range(len(chunks))]
            expanded_inputs.extend(chunks)
            original_to_chunk_indices.append(idx_list)

        # Batch by token count
        batches: List[List[int]] = []
        current_batch: List[int] = []
        current_tokens = 0
        for idx, txt in enumerate(expanded_inputs):
            tok = count_tokens(txt)
            if tok > max_total_tokens:
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_tokens = 0
                batches.append([idx])
                continue
            if current_tokens + tok > max_total_tokens and current_batch:
                batches.append(current_batch)
                current_batch = [idx]
                current_tokens = tok
            else:
                current_batch.append(idx)
                current_tokens += tok
        if current_batch:
            batches.append(current_batch)

        all_chunk_embeddings: List[Optional[List[float]]] = [None] * len(expanded_inputs)
        usage_records: List[Any] = []
        model_name: Optional[str] = None

        for batch_indices in batches:
            batch_inputs = [expanded_inputs[i] for i in batch_indices]
            create_kwargs: Dict[str, Any] = {
                "model": model,
                "input": batch_inputs,
                "encoding_format": encoding_format,
            }
            if dimensions is not None:
                create_kwargs["dimensions"] = dimensions
            response = self.client.embeddings.create(**create_kwargs)
            model_name = response.model
            for local_i, emb in zip(batch_indices, [d.embedding for d in response.data]):
                all_chunk_embeddings[local_i] = emb
            usage_records.append(getattr(response, "usage", None))

        # Aggregate chunks back to original inputs
        import numpy as _np

        def aggregate(emb_list: List[List[float]]) -> List[float]:
            if len(emb_list) == 1:
                return emb_list[0]
            arr = _np.asarray(emb_list, dtype=_np.float32)
            vec = arr.sum(axis=0) if split_aggregate == "sum" else arr.mean(axis=0)
            return vec.tolist()

        aggregated = [aggregate([all_chunk_embeddings[i] for i in idxs]) for idxs in original_to_chunk_indices]

        if normalize:
            def _l2_norm(vec: List[float]) -> List[float]:
                arr = _np.asarray(vec, dtype=_np.float32)
                norm = _np.linalg.norm(arr)
                return vec if norm == 0 else (arr / norm).tolist()
            aggregated = [_l2_norm(e) for e in aggregated]

        base_result = aggregated[0] if single_input else aggregated

        if return_usage:
            return {
                "embeddings": base_result,
                "usage": usage_records[0] if len(usage_records) == 1 else usage_records,
                "model": model_name,
            }
        return base_result
