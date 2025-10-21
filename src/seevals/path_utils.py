from typing import Type, Dict
import pydantic
from jsonpath_ng import parse


def path_exists_in_model(model_class: Type[pydantic.BaseModel], query_path: str) -> bool:
    """
    Validate that a JSONPath query is compatible with a Pydantic model.

    Args:
        model_class: The Pydantic model class to validate against
        query_path: JSONPath query string (e.g., "$.foobar.baz.[*].x")

    Returns:
        True if the path exists in the model structure, False otherwise
    """
    # Step 1: Generate dummy instance from schema
    dummy_instance = _generate_dummy_instance(model_class)

    # Step 2: Parse JSONPath and search in dummy instance
    try:
        jsonpath_expr = parse(query_path)
        matches = jsonpath_expr.find(dummy_instance.model_dump())

        # Step 3: Return validation result
        return len(matches) > 0
    except Exception:
        return False


def _generate_dummy_instance(model_class: Type[pydantic.BaseModel]) -> pydantic.BaseModel:
    """Generate a minimal valid instance from a Pydantic model's schema."""
    schema = model_class.model_json_schema()
    dummy_data = _generate_from_schema(schema, schema.get('$defs', {}))
    return model_class.model_validate(dummy_data)


def _generate_from_schema(schema: Dict, defs: Dict) -> any:
    """Recursively generate dummy data from JSON schema."""
    # Handle $ref
    if '$ref' in schema:
        ref_path = schema['$ref'].split('/')[-1]
        return _generate_from_schema(defs.get(ref_path, {}), defs)

    # Handle anyOf/oneOf - pick first option
    if 'anyOf' in schema:
        return _generate_from_schema(schema['anyOf'][0], defs)
    if 'oneOf' in schema:
        return _generate_from_schema(schema['oneOf'][0], defs)

    schema_type = schema.get('type')

    if schema_type == 'object':
        properties = schema.get('properties', {})
        return {key: _generate_from_schema(prop, defs) for key, prop in properties.items()}

    elif schema_type == 'array':
        # Handle tuples (prefixItems)
        if 'prefixItems' in schema:
            return [_generate_from_schema(item, defs) for item in schema['prefixItems']]
        # Handle regular arrays
        items_schema = schema.get('items', {})
        # Generate at least one item for wildcard paths to work
        return [_generate_from_schema(items_schema, defs)]

    elif schema_type == 'string':
        return ""

    elif schema_type == 'integer':
        return 0

    elif schema_type == 'number':
        return 0.0

    elif schema_type == 'boolean':
        return False

    elif schema_type == 'null':
        return None

    # Default fallback
    return None
