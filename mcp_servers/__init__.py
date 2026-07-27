# MCP servers — deliberately OUTSIDE app/: each subpackage is meant to look
# like an independently deployable service (own Dockerfile, own uvicorn
# process, own port). The main app talks to them only over MCP's streamable
# HTTP transport; nothing in here may import from app/ EXCEPT runbook_server,
# which intentionally demonstrates exposing an internal capability
# (SemanticMemoryStore) over MCP.
