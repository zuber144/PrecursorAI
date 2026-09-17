# Import all models here so SQLAlchemy registers them before create_all()
from app.models.report import User, Report  # noqa: F401
from app.models.analysis import ReportAnalysis  # noqa: F401
from app.models.embedding import ReportEmbedding  # noqa: F401
from app.models.knowledge import KnowledgeChunk, KnowledgeEmbedding  # noqa: F401
from app.models.pattern import Pattern, PatternReport  # noqa: F401
from app.models.alert import Alert  # noqa: F401
from app.models.clarification import ReportClarification  # noqa: F401
