"""Helper functions for Extended OpenAI Conversation component."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from functools import partial
import logging
import re
from typing import Any

from openai import AsyncAzureOpenAI, AsyncClient, AsyncOpenAI, OpenAIError
from openai._exceptions import APIConnectionError

from homeassistant.components import conversation
from homeassistant.components.homeassistant.exposed_entities import async_should_expose
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.template import Template

from .const import (
    CONF_API_PROVIDER,
    CONF_API_VERSION,
    CONF_BASE_URL,
    CONF_ORGANIZATION,
    DEFAULT_API_PROVIDER,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_RETRY_BACKOFF_FACTOR,
    DEFAULT_RETRY_INITIAL_DELAY,
    DEFAULT_RETRY_MAX_ATTEMPTS,
    DEFAULT_RETRY_MAX_DELAY,
    DEFAULT_TOKEN_PARAM,
    MODEL_CONFIG_PATTERNS,
    MODEL_TOKEN_PARAMETER_SUPPORT,
)

_LOGGER = logging.getLogger(__name__)


AZURE_DOMAIN_PATTERN = r"\.(openai\.azure\.com|azure-api\.net|services\.ai\.azure\.com)"


def get_model_config(model: str) -> dict[str, bool]:
    """Get model-specific parameter configuration."""
    # Check patterns in order; first match wins
    for entry in MODEL_CONFIG_PATTERNS:
        pattern = str(entry["pattern"])
        entry_config = entry["config"]
        if re.match(pattern, model, re.IGNORECASE):
            # Type assertion since we know the structure from MODEL_CONFIG_PATTERNS
            return (
                dict(entry_config)
                if isinstance(entry_config, dict)
                else DEFAULT_MODEL_CONFIG
            )

    # Default configuration for standard models (gpt-4, gpt-4o, etc.)
    return DEFAULT_MODEL_CONFIG


def get_exposed_entities(hass: HomeAssistant) -> list[dict[str, Any]]:
    """Get exposed entities."""
    states = [
        state
        for state in hass.states.async_all()
        if async_should_expose(hass, conversation.DOMAIN, state.entity_id)
    ]
    entity_registry = er.async_get(hass)
    exposed_entities = []
    for state in states:
        entity_id = state.entity_id
        entity = entity_registry.async_get(entity_id)

        aliases: list[str] = []
        if entity and entity.aliases:
            aliases = list(entity.aliases)

        exposed_entities.append(
            {
                "entity_id": entity_id,
                "name": state.name,
                "state": state.state,
                "aliases": aliases,
            }
        )
    return exposed_entities


def is_azure_url(base_url: str | None) -> bool:
    """Check if the base URL is an Azure OpenAI URL."""
    return bool(base_url and re.search(AZURE_DOMAIN_PATTERN, base_url))


def get_token_param_for_model(model: str) -> str:
    """Return the token parameter name for a model."""
    model_lower = model.lower()
    for entry in MODEL_TOKEN_PARAMETER_SUPPORT:
        if re.search(entry["pattern"], model_lower):
            return entry["token_param"]
    return DEFAULT_TOKEN_PARAM


def convert_to_template(
    settings: Any,
    template_keys: list[str] | None = None,
    hass: HomeAssistant | None = None,
) -> None:
    if template_keys is None:
        template_keys = ["data", "event_data", "target", "service"]
    _convert_to_template(settings, template_keys, hass, [])


def _convert_to_template(
    settings: Any,
    template_keys: list[str],
    hass: HomeAssistant | None,
    parents: list[str],
) -> None:
    if isinstance(settings, dict):
        for key, value in settings.items():
            if isinstance(value, str) and (
                key in template_keys or set(parents).intersection(template_keys)
            ):
                settings[key] = Template(value, hass)
            if isinstance(value, dict):
                parents.append(key)
                _convert_to_template(value, template_keys, hass, parents)
                parents.pop()
            if isinstance(value, list):
                parents.append(key)
                for item in value:
                    _convert_to_template(item, template_keys, hass, parents)
                parents.pop()
    if isinstance(settings, list):
        for setting in settings:
            _convert_to_template(setting, template_keys, hass, parents)


async def get_authenticated_client(
    hass: HomeAssistant,
    api_key: str,
    base_url: str | None,
    api_version: str | None,
    organization: str | None,
    api_provider: str | None,
    skip_authentication: bool = False,
) -> AsyncClient:
    """Validate OpenAI authentication."""

    client: AsyncClient
    if base_url and (is_azure_url(base_url) or api_provider == "azure"):
        client = AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=base_url,
            api_version=api_version,
            organization=organization,
            http_client=get_async_client(hass),
        )
    else:
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            http_client=get_async_client(hass),
        )

    if skip_authentication:
        return client

    response = await hass.async_add_executor_job(
        partial(client.models.list, timeout=10)
    )

    async for _ in response:
        break
    return client


async def ensure_client_healthy(
    hass: HomeAssistant,
    entry: ConfigEntry,
) -> None:
    """Ensure the OpenAI client is healthy, recreating if necessary.

    Checks if the underlying httpx client is closed and recreates
    the OpenAI client if needed, updating entry.runtime_data.
    """
    client: AsyncClient = entry.runtime_data
    if hasattr(client, "is_closed") and callable(client.is_closed):
        closed = client.is_closed()
    elif hasattr(client, "_client") and hasattr(client._client, "is_closed"):
        closed = client._client.is_closed
    else:
        closed = False

    if closed:
        _LOGGER.warning("OpenAI client connection is closed, recreating client")
        new_client = await get_authenticated_client(
            hass=hass,
            api_key=entry.data[CONF_API_KEY],
            base_url=entry.data.get(CONF_BASE_URL),
            api_version=entry.data.get(CONF_API_VERSION),
            organization=entry.data.get(CONF_ORGANIZATION),
            skip_authentication=True,
            api_provider=entry.data.get(CONF_API_PROVIDER, DEFAULT_API_PROVIDER),
        )
        entry.runtime_data = new_client


async def retry_with_backoff(
    coro_factory: Callable[[], Coroutine],
    hass: HomeAssistant,
    entry: ConfigEntry,
    max_attempts: int = DEFAULT_RETRY_MAX_ATTEMPTS,
    initial_delay: float = DEFAULT_RETRY_INITIAL_DELAY,
    max_delay: float = DEFAULT_RETRY_MAX_DELAY,
    backoff_factor: float = DEFAULT_RETRY_BACKOFF_FACTOR,
) -> None:
    """Execute an async callable with exponential backoff on OpenAI errors.

    Before each retry, ensures the OpenAI client is healthy.
    Raises the last OpenAIError if all attempts fail.
    """
    last_error: OpenAIError | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            await ensure_client_healthy(hass, entry)
            await coro_factory()
            return
        except OpenAIError as err:
            last_error = err
            if attempt < max_attempts:
                delay = min(
                    initial_delay * (backoff_factor ** (attempt - 1)),
                    max_delay,
                )
                _LOGGER.warning(
                    "OpenAI request attempt %d/%d failed (%s: %s), "
                    "retrying in %.1fs",
                    attempt,
                    max_attempts,
                    type(err).__name__,
                    err,
                    delay,
                )
                await asyncio.sleep(delay)
            else:
                _LOGGER.error(
                    "All %d OpenAI request attempts failed. Last error: %s",
                    max_attempts,
                    err,
                )

    if last_error is not None:
        raise last_error
