"""Phoenix tracing setup for observability."""
import phoenix as px
from phoenix.otel import register
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor

try:
    from .config import get_settings
except ImportError:
    from config import get_settings


_tracer_initialized = False
_openai_instrumented = False


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
    
    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor
    except ImportError:
        print("Warning: openinference.instrumentation.langchain not available. LangChain instrumentation skipped.")
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
    print(f"Phoenix tracing initialized - project: {settings.phoenix_project_name}")


def init_openai_tracing() -> None:
    """Initialize Phoenix tracing specifically for OpenAI client calls.
    
    This instruments direct OpenAI client calls (not LangChain).
    """
    global _openai_instrumented
    
    if _openai_instrumented:
        return
    
    try:
        from opentelemetry.instrumentation.openai import OpenAIInstrumentor
        
        settings = get_settings()
        
        # Register tracer provider with Phoenix
        tracer_provider = register(
            project_name=settings.phoenix_project_name,
            endpoint=settings.phoenix_collector_endpoint
        )
        
        # Instrument OpenAI client
        OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
        
        _openai_instrumented = True
        print(f"OpenAI tracing initialized - project: {settings.phoenix_project_name}")
    except ImportError:
        print("Warning: opentelemetry-instrumentation-openai not installed. Install it with: pip install opentelemetry-instrumentation-openai")


def get_tracer(name: str = "mriia-tutor"):
    """Get a tracer instance for manual spans."""
    return trace.get_tracer(name)
