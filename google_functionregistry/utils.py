import inspect
import re
from pprint import pprint
from typing import get_type_hints

import openai
from langchain_core.utils.function_calling import convert_to_openai_function
from pydantic import (BaseModel, Field,  # do not remove unused imports
                      ValidationError)


def generate_pydantic_model(func):
    func_name = func.__name__
    camel_case_name = "".join(word.capitalize() for word in func_name.split("_"))
    docstring = inspect.getdoc(func)
    signature = inspect.signature(func)
    type_hints = get_type_hints(func)

    fields = []
    for param_name, param in signature.parameters.items():
        annotation = type_hints.get(param_name, param.annotation)
        default = param.default if param.default is not inspect.Parameter.empty else ...

        # Extracting the parameter description from the function's source code
        source_lines = inspect.getsourcelines(func)[0]
        param_line = next((line for line in source_lines if param_name in line), None)
        description_match = re.search(r"#\s*(.*)", param_line) if param_line else None
        description = description_match.group(1) if description_match else None

        # Handle the annotation correctly for both simple and complex types
        annotation_str = (
            annotation.__name__ if "class" in repr(annotation) else repr(annotation)
        )

        fields.append(
            f"{param_name}: {annotation_str} = Field({default}, description='{description}')"
        )

    fields_str = "\n    ".join(fields)

    model_code = f"""
class {camel_case_name}(BaseModel):
    \"\"\"{docstring}\"\"\"
    {fields_str}
    """

    namespace = {}
    exec(model_code, globals(), namespace)
    return namespace[camel_case_name]


# Example usage
def some_function(
    parameter1: int,  # Some description
    parameter2: tuple[int, int] = (1, 2),  # p2 description
):
    """
    some_function docstring
    """
    pass


SomeFunctionModel = generate_pydantic_model(some_function)
SomeFunctionModel(parameter1=1, parameter2=(1, 2))
try:
    SomeFunctionModel(parameter1="1", parameter2=(1, 2))
except ValidationError:
    # Expected
    ...
try:
    SomeFunctionModel(parameter1=1, parameter2=(1, "2"))
except ValidationError:
    # Expected
    ...

pprint(openai.pydantic_function_tool(SomeFunctionModel)["function"])
print("\n")
pprint(convert_to_openai_function(some_function))
