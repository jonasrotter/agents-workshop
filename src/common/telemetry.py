"""OpenTelemetry setup for the Agentic AI Patterns Workshop.

This module configures distributed tracing and metrics collection
using OpenTelemetry with Azure Monitor integration.

Example:
    from src.common.telemetry import setup_telemetry, get_tracer

    # Initialize telemetry (call once at startup)
    setup_telemetry()

    # Get a tracer for your module
    tracer = get_tracer(__name__)

    # Create spans
    with tracer.start_as_current_span("my_operation") as span:
        span.set_attribute("custom.attribute", "value")
        result = do_work()
"""

import logging
from functools import lru_cache
from typing import Optional

from opentelemetry import trace
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import Tracer

from src.common.config import get_settings

logger = logging.getLogger(__name__)

# Service identification for telemetry
SERVICE_NAME = "agents-workshop"
SERVICE_VERSION = "0.1.0"

_telemetry_initialized = False


def setup_telemetry(
    *,
    service_name: str = SERVICE_NAME,
    service_version: str = SERVICE_VERSION,
    force_console_export: bool = False,
) -> None:
    """Initialize OpenTelemetry with optional Azure Monitor integration.

    This function sets up distributed tracing. Call it once at application
    startup, typically in your main entry point.

    Args:
        service_name: Name to identify this service in traces.
        service_version: Version string for the service.
        force_console_export: If True, always use console export (for testing).

    Note:
        - If APPLICATIONINSIGHTS_CONNECTION_STRING is set and enable_otel is True,
          traces will be sent to Azure Monitor.
        - Otherwise, traces are exported to console.
        - Calling this function multiple times is safe; it only initializes once.
    """
    global _telemetry_initialized

    if _telemetry_initialized:
        logger.debug("Telemetry already initialized, skipping")
        return

    settings = get_settings()

    if not settings.enable_otel:
        logger.info("OpenTelemetry disabled by configuration")
        _telemetry_initialized = True
        return

    # Create resource with service information
    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": service_version,
        }
    )

    # Create tracer provider
    provider = TracerProvider(resource=resource)

    # Configure exporter based on environment
    if (
        settings.is_observability_configured
        and not force_console_export
    ):
        try:
            # Import Azure Monitor exporter only when needed
            from azure.monitor.opentelemetry.exporter import (
                AzureMonitorTraceExporter,
            )

            azure_exporter = AzureMonitorTraceExporter(
                connection_string=settings.applicationinsights_connection_string
            )
            provider.add_span_processor(BatchSpanProcessor(azure_exporter))
            logger.info("OpenTelemetry configured with Azure Monitor exporter")
        except ImportError:
            logger.warning(
                "azure-monitor-opentelemetry-exporter not installed, "
                "falling back to console export"
            )
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        except Exception as e:
            logger.warning(
                f"Failed to configure Azure Monitor exporter: {e}, "
                "falling back to console export"
            )
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    else:
        # Use console exporter for development/testing
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        logger.info("OpenTelemetry configured with console exporter")

    # Set the global tracer provider
    trace.set_tracer_provider(provider)

    # Instrument logging to correlate logs with traces
    LoggingInstrumentor().instrument()

    _telemetry_initialized = True
    logger.info(f"Telemetry initialized for {service_name} v{service_version}")


@lru_cache(maxsize=32)
def get_tracer(name: str) -> Tracer:
    """Get a tracer instance for the specified module.

    Tracers are cached, so multiple calls with the same name
    return the same tracer instance.

    Args:
        name: The name of the module or component requesting the tracer.
              Typically pass __name__.

    Returns:
        A Tracer instance for creating spans.

    Example:
        tracer = get_tracer(__name__)
        with tracer.start_as_current_span("process_request") as span:
            span.set_attribute("request.id", request_id)
            result = handle_request(request)
    """
    return trace.get_tracer(name, SERVICE_VERSION)


def create_span_attributes(
    *,
    agent_name: Optional[str] = None,
    tool_name: Optional[str] = None,
    model: Optional[str] = None,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    **kwargs: str,
) -> dict[str, str | int]:
    """Create a dictionary of span attributes following semantic conventions.

    This helper creates consistently named attributes for common operations.

    Args:
        agent_name: Name of the agent being traced.
        tool_name: Name of the tool being invoked.
        model: Model name/identifier being used.
        prompt_tokens: Number of tokens in the prompt.
        completion_tokens: Number of tokens in the completion.
        **kwargs: Additional custom attributes.

    Returns:
        Dictionary of attribute name-value pairs.

    Example:
        attrs = create_span_attributes(
            agent_name="researcher",
            model="gpt-4o",
            prompt_tokens=150
        )
        span.set_attributes(attrs)
    """
    attributes: dict[str, str | int] = {}

    if agent_name:
        attributes["agent.name"] = agent_name
    if tool_name:
        attributes["tool.name"] = tool_name
    if model:
        attributes["llm.model"] = model
    if prompt_tokens is not None:
        attributes["llm.prompt_tokens"] = prompt_tokens
    if completion_tokens is not None:
        attributes["llm.completion_tokens"] = completion_tokens

    # Add any additional custom attributes
    for key, value in kwargs.items():
        # Normalize key to use dots as separators
        normalized_key = key.replace("_", ".")
        if isinstance(value, (str, int)):
            attributes[normalized_key] = value

    return attributes


def record_exception(span: trace.Span, exception: Exception) -> None:
    """Record an exception on a span with proper formatting.

    This helper ensures exceptions are recorded consistently across
    the codebase.

    Args:
        span: The span to record the exception on.
        exception: The exception to record.

    Example:
        with tracer.start_as_current_span("risky_operation") as span:
            try:
                do_risky_thing()
            except Exception as e:
                record_exception(span, e)
                raise
    """
    span.record_exception(exception)
    span.set_status(trace.Status(trace.StatusCode.ERROR, str(exception)))


def get_current_span() -> trace.Span:
    """Get the current active span.

    Returns:
        The current span from the context.

    Example:
        span = get_current_span()
        span.add_event("checkpoint_reached", {"checkpoint": "validation"})
    """
    return trace.get_current_span()


def is_telemetry_enabled() -> bool:
    """Check if telemetry is initialized and enabled.

    Returns:
        True if telemetry has been initialized.
    """
    return _telemetry_initialized


def reset_telemetry() -> None:
    """Reset telemetry state (for testing only).

    Warning:
        This is intended for testing only. Do not use in production code.
    """
    global _telemetry_initialized
    _telemetry_initialized = False
