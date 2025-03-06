"""
Shows example usage of the `FunctionRegistry` and `ParserRegistry`.
"""

import random
from enum import StrEnum

from pydantic import BaseModel, Field

from openai_functionregistry.client import Client
from openai_functionregistry.tool_registry import (FunctionRegistry,
                                                   ParserRegistry,
                                                   get_tool_call_id,
                                                   get_tool_call_ids)


class TemperatureUnit(StrEnum):
    CELSIUS = "celsius"
    FAHRENHEIT = "fahrenheit"


class GetWeather(BaseModel):
    """Get the weather at the given location in the given temperature unit of measurement (doc1)"""

    location: str = Field(description="The location to get the weather for")
    unit: TemperatureUnit = Field(description="Temperature Unit")


def get_weather(location: str, unit: TemperatureUnit = TemperatureUnit.CELSIUS) -> str:
    """Get the weather at the given location in the given temperature unit of measurement (doc2)"""
    return f"The weather in {location} is {random.randint(20, 40)} degrees {unit} with a windy breeze."


def get_weather_alt(params: GetWeather) -> str:
    """Get the weather at the given location in the given temperature unit of measurement (doc3)"""
    location = params.location
    unit = params.unit
    return f"The weather in {location} is {random.randint(20, 40)} degrees {unit} with a windy breeze."


class WeatherResponse(BaseModel):
    """The temperature in a given unit at a specified location"""

    location: str
    temperature: float
    unit: str


def main():
    # Create shared clients
    mini_client = Client(is_mini=True)
    regular_client = Client(is_mini=False)

    # Register function calls
    func_registry = FunctionRegistry(mini_client, regular_client)
    # Register with explicit param model
    func_registry.register(get_weather, GetWeather)
    # Or register a function that takes a parameter spec as the first argument and autodetect parameter model
    # func_registry.register(get_weather_alt)

    # Make function call
    messages = [
        {"role": "system", "content": "Extract the weather information."},
        {"role": "user", "content": "What's the weather like in London?"},
    ]
    response1, funcion_call1 = func_registry.call_function(
        messages, target_function="GetWeather"
    )

    # Parse the result
    parser_registry = ParserRegistry(mini_client, regular_client)
    parser_registry.register(WeatherResponse)

    # The previous messages are not required in order to parse the response.
    parse_messages = [
        response1.choices[0].message.model_dump(),
        {
            "role": "tool",
            "content": function_call1.result,
            "tool_call_id": get_tool_call_id(response1),
        },
    ]

    response2, weather_response = parser_registry.parse_response(
        messages=parse_messages,
        target_model="WeatherResponse",
    )

    print(f"Parsed response: {weather_response}")

    ### TRY MULTIPLE TOOL CALLS. Get the weather for London and New York at the same time.

    # Make function call
    messages = [
        {"role": "system", "content": "Extract the weather information."},
        {"role": "user", "content": "What's the weather like in London and New York??"},
    ]

    # NOTE: When `func_registry.call_functions` is given a `target_function`
    # (i.e. `chat_completion` is given a `tool_choice`), then it only returns one input
    # even though it might be proper to return many.
    # Use `function_subset` to attempt restrict the model to a specific function instead

    response3, function_calls3 = func_registry.call_functions(messages)

    # The previous messages are not required in order to parse the response.
    parse_messages = [
        {
            "role": "system",
            "content": "Call the supplied functions in order to structure the given information:",
        },
        response1.choices[0].message.model_dump(),
    ]
    for function_call, tool_call_id in zip(
        function_calls3, get_tool_call_ids(response1)
    ):
        parse_messages.append(
            {
                "role": "tool",
                "content": function_call.result,
                "tool_call_id": tool_call_id,
            },
        )

    response4, weather_responses = parser_registry.parse_responses(
        messages=parse_messages,
        model_subset=["WeatherResponse"],
    )

    print(f"Parsed response: {weather_responses}")


if __name__ == "__main__":
    main()
