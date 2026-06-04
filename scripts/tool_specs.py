#!/usr/bin/env python3
"""Tool schema pools + grounded-call generator for the interface robustness gate.

Shared by build_interface_sft.py (training tools) and build_product_eval.py (held-out
eval tools). Training and eval tool NAMES are disjoint, so the eval measures
generalization to UNSEEN tool names. Each generated record embeds its argument values
verbatim in the task text, so argument-value correctness is exactly scorable.

A schema covers one of four arg structures (subtype):
  single_arg   one required arg
  multi_arg    several required args
  optional_arg one required + one optional (included ~half the time)
  nested_arg   one required object arg with sub-fields

make_tool_record(schema, rng) -> {
  'task', 'schema' (OpenAI-ish dict), 'call' ({name, arguments}),
  'required' [keys], 'optional' [keys], 'arg_values' {flat path -> value}, 'subtype'
}
"""
from __future__ import annotations
import json
import random
from typing import Any


def tool_prompt(task: str, schema: dict[str, Any]) -> str:
    """Schema-in-prompt tool format (shared by training + eval). The function schema
    is provided in-context so the model copies the name + required arg keys from it
    and extracts values from the task — the realistic function-calling setup, and the
    only way argument correctness on UNSEEN tool names is well-defined."""
    return (f"Task: {task}\n"
            f"Function: {json.dumps(schema, ensure_ascii=False)}\n"
            f"Response:")


def tool_response(call: dict[str, Any], rng: random.Random) -> str:
    """Gold Hermes response with a short NL preamble (leading with the <tool_call>
    special token straight after 'Response:' is unreachable with a frozen tied LM head)."""
    payload = json.dumps({"name": call["name"], "arguments": call["arguments"]}, ensure_ascii=False)
    pre = rng.choice([f"I will call the {call['name']} function.",
                      f"Calling {call['name']}:",
                      f"Using {call['name']}:"])
    return f"{pre}\n<tool_call>\n{payload}\n</tool_call>"

VALUE_POOLS: dict[str, list[str]] = {
    'city': ['Paris', 'Tokyo', 'Boston', 'Denver', 'Cairo', 'Oslo', 'Lima', 'Seoul', 'Madrid', 'Austin'],
    'date': ['2026-07-01', '2026-08-15', '2026-09-30', '2026-12-25', '2026-03-10', '2026-05-22'],
    'ticker': ['AAPL', 'MSFT', 'GOOG', 'TSLA', 'AMZN', 'NVDA'],
    'email': ['bob@example.com', 'alice@corp.io', 'carol@mail.net', 'dan@team.org'],
    'number': ['5', '10', '25', '100', '3', '50', '7'],
    'word': ['report', 'invoice', 'summary', 'backup', 'contract', 'dataset', 'roadmap'],
    'name': ['Alice', 'Bob', 'Carol', 'Dave', 'Erin'],
    'currency': ['USD', 'EUR', 'JPY', 'GBP'],
    'topic': ['budget', 'roadmap', 'hiring', 'launch', 'review'],
    'timezone': ['UTC', 'PST', 'EST', 'CET'],
    'language': ['French', 'Spanish', 'German', 'Japanese'],
    'text': ['hello', 'good morning', 'thank you', 'welcome'],
    'city2': ['London', 'Berlin', 'Rome', 'Cairo', 'Dubai', 'Quito'],
}

# ── schema pools (train and eval NAMES disjoint) ───────────────────────────────

TRAIN_TOOLS: list[dict[str, Any]] = [
    {'name': 'get_weather', 'subtype': 'single_arg', 'description': 'Get the current weather for a location.',
     'params': [{'key': 'location', 'kind': 'city', 'required': True}],
     'templates': ['What is the weather in {location}?', 'Get the current weather in {location}.']},
    {'name': 'define_word', 'subtype': 'single_arg', 'description': 'Look up the definition of a word.',
     'params': [{'key': 'word', 'kind': 'word', 'required': True}],
     'templates': ['Define the word {word}.', 'What does {word} mean?']},
    {'name': 'book_flight', 'subtype': 'multi_arg', 'description': 'Book a flight between two cities.',
     'params': [{'key': 'origin', 'kind': 'city', 'required': True},
                {'key': 'destination', 'kind': 'city2', 'required': True},
                {'key': 'date', 'kind': 'date', 'required': True}],
     'templates': ['Book a flight from {origin} to {destination} on {date}.']},
    {'name': 'convert_currency', 'subtype': 'multi_arg', 'description': 'Convert an amount between currencies.',
     'params': [{'key': 'amount', 'kind': 'number', 'required': True},
                {'key': 'from_currency', 'kind': 'currency', 'required': True},
                {'key': 'to_currency', 'kind': 'currency', 'required': True}],
     'templates': ['Convert {amount} {from_currency} to {to_currency}.']},
    {'name': 'search_products', 'subtype': 'optional_arg', 'description': 'Search products, optionally limiting results.',
     'params': [{'key': 'query', 'kind': 'word', 'required': True},
                {'key': 'max_results', 'kind': 'number', 'required': False}],
     'templates': ['Search for {query}.'],
     'optional_templates': ['Search for {query} and return {max_results} results.']},
    {'name': 'list_files', 'subtype': 'optional_arg', 'description': 'List files in a directory, optional pattern.',
     'params': [{'key': 'directory', 'kind': 'word', 'required': True},
                {'key': 'pattern', 'kind': 'word', 'required': False}],
     'templates': ['List the files in the {directory} directory.'],
     'optional_templates': ['List files in the {directory} directory matching {pattern}.']},
    {'name': 'create_event', 'subtype': 'nested_arg', 'description': 'Create a calendar event.',
     'params': [{'key': 'event', 'kind': 'object', 'required': True,
                 'fields': [{'key': 'title', 'kind': 'topic'}, {'key': 'date', 'kind': 'date'}]}],
     'templates': ['Create an event titled {title} on {date}.']},
    {'name': 'send_message', 'subtype': 'nested_arg', 'description': 'Send a message to a recipient.',
     'params': [{'key': 'message', 'kind': 'object', 'required': True,
                 'fields': [{'key': 'to', 'kind': 'email'}, {'key': 'body', 'kind': 'text'}]}],
     'templates': ['Send a message to {to} saying {body}.']},
    {'name': 'set_timer', 'subtype': 'single_arg', 'description': 'Set a timer for N minutes.',
     'params': [{'key': 'minutes', 'kind': 'number', 'required': True}],
     'templates': ['Set a timer for {minutes} minutes.']},
    {'name': 'translate', 'subtype': 'multi_arg', 'description': 'Translate text to a language.',
     'params': [{'key': 'text', 'kind': 'text', 'required': True},
                {'key': 'target_language', 'kind': 'language', 'required': True}],
     'templates': ['Translate {text} into {target_language}.']},
    # extra single-arg tools — name diversity so single-arg name-copying is robust
    {'name': 'get_time', 'subtype': 'single_arg', 'description': 'Get the current time in a timezone.',
     'params': [{'key': 'timezone', 'kind': 'timezone', 'required': True}],
     'templates': ['What time is it in {timezone}?']},
    {'name': 'find_synonym', 'subtype': 'single_arg', 'description': 'Find a synonym for a word.',
     'params': [{'key': 'word', 'kind': 'word', 'required': True}],
     'templates': ['Find a synonym for {word}.']},
    {'name': 'get_exchange_rate', 'subtype': 'single_arg', 'description': 'Get the exchange rate for a currency.',
     'params': [{'key': 'currency', 'kind': 'currency', 'required': True}],
     'templates': ['What is the exchange rate for {currency}?']},
    {'name': 'check_inventory', 'subtype': 'single_arg', 'description': 'Check inventory for a product.',
     'params': [{'key': 'product', 'kind': 'word', 'required': True}],
     'templates': ['Check inventory for {product}.']},
    {'name': 'geocode_city', 'subtype': 'single_arg', 'description': 'Get coordinates for a city.',
     'params': [{'key': 'city', 'kind': 'city', 'required': True}],
     'templates': ['Get the coordinates of {city}.']},
    {'name': 'get_altitude', 'subtype': 'single_arg', 'description': 'Get the altitude of a city.',
     'params': [{'key': 'city', 'kind': 'city', 'required': True}],
     'templates': ['What is the altitude of {city}?']},
    {'name': 'lookup_phone', 'subtype': 'single_arg', 'description': 'Look up a phone number by name.',
     'params': [{'key': 'name', 'kind': 'name', 'required': True}],
     'templates': ['Look up the phone number for {name}.']},
    {'name': 'get_rating', 'subtype': 'single_arg', 'description': 'Get the rating of an item.',
     'params': [{'key': 'item', 'kind': 'word', 'required': True}],
     'templates': ['What is the rating of {item}?']},
    {'name': 'fetch_balance', 'subtype': 'single_arg', 'description': 'Fetch the balance of an account.',
     'params': [{'key': 'account', 'kind': 'word', 'required': True}],
     'templates': ['Fetch the balance of the {account} account.']},
    {'name': 'describe_image', 'subtype': 'single_arg', 'description': 'Describe an image file.',
     'params': [{'key': 'filename', 'kind': 'word', 'required': True}],
     'templates': ['Describe the {filename} image.']},
]

EVAL_TOOLS: list[dict[str, Any]] = [
    {'name': 'lookup_stock', 'subtype': 'single_arg', 'description': 'Look up a stock price by ticker.',
     'params': [{'key': 'ticker', 'kind': 'ticker', 'required': True}],
     'templates': ['Look up the stock price for {ticker}.', 'What is {ticker} trading at?']},
    {'name': 'get_population', 'subtype': 'single_arg', 'description': 'Get the population of a city.',
     'params': [{'key': 'city', 'kind': 'city', 'required': True}],
     'templates': ['What is the population of {city}?']},
    {'name': 'reserve_table', 'subtype': 'multi_arg', 'description': 'Reserve a restaurant table.',
     'params': [{'key': 'restaurant', 'kind': 'word', 'required': True},
                {'key': 'party_size', 'kind': 'number', 'required': True},
                {'key': 'time', 'kind': 'date', 'required': True}],
     'templates': ['Reserve a table at {restaurant} for {party_size} people on {time}.']},
    {'name': 'transfer_funds', 'subtype': 'multi_arg', 'description': 'Transfer funds between accounts.',
     'params': [{'key': 'amount', 'kind': 'number', 'required': True},
                {'key': 'from_account', 'kind': 'word', 'required': True},
                {'key': 'to_account', 'kind': 'word', 'required': True}],
     'templates': ['Transfer {amount} from {from_account} to {to_account}.']},
    {'name': 'query_database', 'subtype': 'optional_arg', 'description': 'Query a table, optional filter.',
     'params': [{'key': 'table', 'kind': 'word', 'required': True},
                {'key': 'filter', 'kind': 'word', 'required': False}],
     'templates': ['Query the {table} table.'],
     'optional_templates': ['Query the {table} table where status is {filter}.']},
    {'name': 'fetch_logs', 'subtype': 'optional_arg', 'description': 'Fetch service logs, optional limit.',
     'params': [{'key': 'service', 'kind': 'word', 'required': True},
                {'key': 'limit', 'kind': 'number', 'required': False}],
     'templates': ['Fetch the logs for the {service} service.'],
     'optional_templates': ['Fetch the last {limit} logs for the {service} service.']},
    {'name': 'schedule_meeting', 'subtype': 'nested_arg', 'description': 'Schedule a meeting.',
     'params': [{'key': 'meeting', 'kind': 'object', 'required': True,
                 'fields': [{'key': 'attendee', 'kind': 'email'}, {'key': 'topic', 'kind': 'topic'}]}],
     'templates': ['Schedule a meeting with {attendee} about {topic}.']},
    {'name': 'register_user', 'subtype': 'nested_arg', 'description': 'Register a new user.',
     'params': [{'key': 'user', 'kind': 'object', 'required': True,
                 'fields': [{'key': 'name', 'kind': 'name'}, {'key': 'email', 'kind': 'email'}]}],
     'templates': ['Register a user named {name} with email {email}.']},
    {'name': 'get_distance', 'subtype': 'multi_arg', 'description': 'Get distance between two cities.',
     'params': [{'key': 'from_city', 'kind': 'city', 'required': True},
                {'key': 'to_city', 'kind': 'city2', 'required': True}],
     'templates': ['How far is it from {from_city} to {to_city}?']},
    {'name': 'summarize_doc', 'subtype': 'single_arg', 'description': 'Summarize a document by name.',
     'params': [{'key': 'document', 'kind': 'word', 'required': True}],
     'templates': ['Summarize the {document} document.']},
    # neutral single-arg eval tools (broaden the per-subtype name sample so the metric
    # measures the copying skill, not luck on 2 semantically-colliding names)
    {'name': 'get_elevation', 'subtype': 'single_arg', 'description': 'Get the elevation of a city.',
     'params': [{'key': 'city', 'kind': 'city', 'required': True}],
     'templates': ['What is the elevation of {city}?']},
    {'name': 'get_zipcode', 'subtype': 'single_arg', 'description': 'Get the zipcode of a city.',
     'params': [{'key': 'city', 'kind': 'city', 'required': True}],
     'templates': ['What is the zipcode of {city}?']},
    {'name': 'count_pages', 'subtype': 'single_arg', 'description': 'Count the pages of a document.',
     'params': [{'key': 'document', 'kind': 'word', 'required': True}],
     'templates': ['How many pages are in the {document} document?']},
    {'name': 'check_warranty', 'subtype': 'single_arg', 'description': 'Check the warranty of a product.',
     'params': [{'key': 'product', 'kind': 'word', 'required': True}],
     'templates': ['Check the warranty for the {product}.']},
]


def _pick(rng: random.Random, kind: str, used: set[str]) -> str:
    pool = VALUE_POOLS[kind]
    for _ in range(10):
        v = rng.choice(pool)
        if v not in used:
            used.add(v)
            return v
    return rng.choice(pool)


def make_tool_record(schema: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Generate one grounded tool record from a schema."""
    used: set[str] = set()
    args: dict[str, Any] = {}
    arg_values: dict[str, str] = {}     # flat path -> value, for exact scoring
    required: list[str] = []
    optional: list[str] = []
    fill: dict[str, str] = {}

    for p in schema['params']:
        if p['kind'] == 'object':
            obj = {}
            for f in p['fields']:
                v = _pick(rng, f['kind'], used)
                obj[f['key']] = v
                arg_values[f"{p['key']}.{f['key']}"] = v
                fill[f['key']] = v
            args[p['key']] = obj
            required.append(p['key'])
        elif p['required']:
            v = _pick(rng, p['kind'], used)
            args[p['key']] = v
            arg_values[p['key']] = v
            fill[p['key']] = v
            required.append(p['key'])
        else:
            optional.append(p['key'])

    # optional arg: include ~half the time, choosing the matching template
    include_opt = bool(optional) and rng.random() < 0.5
    if include_opt:
        op = next(p for p in schema['params'] if p['key'] == optional[0])
        v = _pick(rng, op['kind'], used)
        args[op['key']] = v
        arg_values[op['key']] = v
        fill[op['key']] = v
        template = rng.choice(schema['optional_templates'])
    else:
        template = rng.choice(schema['templates'])

    task = template.format(**fill)
    json_schema = _to_json_schema(schema)
    return {
        'task': task,
        'schema': json_schema,
        'call': {'name': schema['name'], 'arguments': args},
        'required': required,
        'optional': optional,
        'arg_values': arg_values,
        'subtype': schema['subtype'],
        'name': schema['name'],
    }


def _to_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    props: dict[str, Any] = {}
    req: list[str] = []
    for p in schema['params']:
        if p['kind'] == 'object':
            sub = {f['key']: {'type': 'string'} for f in p['fields']}
            props[p['key']] = {'type': 'object', 'properties': sub,
                               'required': [f['key'] for f in p['fields']]}
        else:
            props[p['key']] = {'type': 'string'}
        if p.get('required'):
            req.append(p['key'])
    return {'name': schema['name'], 'description': schema['description'],
            'parameters': {'type': 'object', 'properties': props, 'required': req}}
