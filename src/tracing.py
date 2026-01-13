"""Phoenix tracing setup for observability."""
import phoenix as px
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from .config import get_settings


_tracer_initialized = False


def init_tracing() -> None:
    """Initialize Phoenix tracing with OpenTelemetry instrumentation.
    
    This sets up:
    - Connection to Phoenix collector
    - LangChain/LangGraph instrumentation
    - Automatic tracing of all LLM calls
    """
    global _tracer_initialized
    
    if _tracer_initialized:
        return
    
    settings = get_settings()
    
    # Register tracer provider with Phoenix
    tracer_provider = register(
        project_name=settings.phoenix_project_name,
        endpoint=settings.phoenix_collector_endpoint
    )
    
    # Instrument LangChain (covers LangGraph too)
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    
    _tracer_initialized = True
    print(f"✓ Phoenix tracing initialized - project: {settings.phoenix_project_name}")


def get_tracer(name: str = "mriia-tutor"):
    """Get a tracer instance for manual spans."""
    return trace.get_tracer(name)
