"""
Defines a `ParserRegistry` and a `FunctionRegistry` to make it convenient
"""

import logging
import os
from argparse import ArgumentError
from dataclasses import dataclass
from typing import Any, TypeVar

# import google.generativeai as genai
from google import genai
from google.genai.types import GenerateContentResponse
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)


Model = TypeVar("Model", bound=BaseModel)


@dataclass
class FunctionCall:
    """Function call arguments and the result of the function call."""

    arguments: Model
    result: Any


class LLMError(Exception):
    """Base exception for LLM-related errors"""

    pass


class ModelFailedError(LLMError):
    """Raised when both mini and regular models fail"""

    pass


class MultipleToolCallsError(LLMError):
    """Raised when multiple tool calls are received but only one was expected"""

    pass


class NoToolCallsError(LLMError):
    """Raised when no tool calls are received"""

    pass


class BaseRegistry:
    """Base registry for LLM function calls and parsing"""

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: str | None = None,
        api_key_env: str = "GEMINI_API_KEY",
    ):
        if not api_key:
            try:
                from dotenv import load_dotenv

                load_dotenv()
            except ImportError:
                pass
            try:
                api_key = os.environ[api_key_env]
            except KeyError:
                raise ValueError("api_key_env not found in environment")
            raise ValueError("api_key not specified")

        # genai.configure(api_key=api_key)
        # self.client = genai.GenerativeModel(model)
        self.client = genai.Client(api_key=api_key)
        self.model = model


class ParserRegistry(BaseRegistry):
    """Registry for parsing unstructured responses into structured data"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def parse_response(
        self,
        messages: list[dict[str, str]] | str,
        model: type[Model],
    ) -> tuple[GenerateContentResponse, Model]:
        """Parse multiple unstructured responses into structured data"""
        response = self.client.models.generate_content(
            model=self.model,
            contents=messages,
            config={
                "response_mime_type": "application/json",
                "response_schema": model,
            },
        )

        return response, response.parsed


def test():
    from pydantic import BaseModel

    class Recipe(BaseModel):
        recipe_name: str
        ingredients: list[str]

    messages = "List a few popular cookie recipes. Be sure to include the amounts of ingredients."
    model = list[Recipe]
    r = ParserRegistry()
    response, models = r.parse_response(messages, model)
