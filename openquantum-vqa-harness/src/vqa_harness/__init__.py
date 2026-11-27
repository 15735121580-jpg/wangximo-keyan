"""Public client API for complete host programs and typed diagram search."""
from .deployment import DeploymentContract, PhysicalEvidence, admit_deployment, audit_physical_evidence, dispatch_qualified, freeze_deployment, qualify_saved
from .diagram_context import OpenDiagram, Port, Stage, replace_region
from .diagram_search import Match, direct_diagram, search_diagram
from .implementation_relation import implementation_relation
from .optimality import INFEASIBLE, compile_joint, optimize_physical_domain
from .guarded_release import Assessment, ReleaseContract
from .host_compilation import Branch, HostJob, compile_bounded, compile_direct

__version__ = "0.3.1"
__all__ = [
    "DeploymentContract", "PhysicalEvidence", "admit_deployment", "audit_physical_evidence", "dispatch_qualified", "freeze_deployment", "qualify_saved",
    "Assessment", "Branch", "HostJob", "Match", "OpenDiagram", "Port", "ReleaseContract",
    "Stage", "compile_bounded", "compile_direct", "direct_diagram", "implementation_relation", "replace_region", "search_diagram",
    "INFEASIBLE", "compile_joint", "optimize_physical_domain",
]
