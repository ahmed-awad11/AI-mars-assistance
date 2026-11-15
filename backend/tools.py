"""
Marswalk AI Assistant - Tool Calling System
Registry and execution of tools for the AI assistant.
"""
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from functools import wraps
import inspect

logger = logging.getLogger(__name__)


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""
    name: str
    type: str  # string, number, boolean, array, object
    description: str
    required: bool = False
    default: Any = None
    enum: List[Any] = field(default_factory=list)


@dataclass
class ToolDefinition:
    """Definition of a callable tool."""
    name: str
    description: str
    parameters: List[ToolParameter] = field(default_factory=list)
    returns: str = "object"  # Description of return type
    requires_permission: bool = False
    category: str = "general"

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema for LLM consumption."""
        properties = {}
        required = []

        for param in self.parameters:
            prop = {
                "type": param.type,
                "description": param.description
            }
            if param.enum:
                prop["enum"] = param.enum
            if param.default is not None:
                prop["default"] = param.default

            properties[param.name] = prop
            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }


class ToolResult:
    """Result of a tool execution."""
    def __init__(
        self,
        success: bool,
        data: Any = None,
        error: str = None,
        metadata: Dict = None
    ):
        self.success = success
        self.data = data
        self.error = error
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata
        }

    @classmethod
    def success_result(cls, data: Any, metadata: Dict = None) -> "ToolResult":
        return cls(success=True, data=data, metadata=metadata)

    @classmethod
    def error_result(cls, error: str, metadata: Dict = None) -> "ToolResult":
        return cls(success=False, error=error, metadata=metadata)


class ToolRegistry:
    """Registry for managing and executing tools."""

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        self._handlers: Dict[str, Callable[..., Awaitable[ToolResult]]] = {}

    def register(
        self,
        definition: ToolDefinition,
        handler: Callable[..., Awaitable[ToolResult]]
    ):
        """Register a tool with its handler."""
        self._tools[definition.name] = definition
        self._handlers[definition.name] = handler
        logger.info(f"Registered tool: {definition.name}")

    def unregister(self, name: str):
        """Unregister a tool."""
        self._tools.pop(name, None)
        self._handlers.pop(name, None)

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Get tool definition by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[ToolDefinition]:
        """List all registered tools."""
        return list(self._tools.values())

    def get_tools_for_llm(self) -> List[Dict[str, Any]]:
        """Get tool definitions in JSON schema format for LLM."""
        return [tool.to_json_schema() for tool in self._tools.values()]

    async def execute(self, name: str, arguments: Dict[str, Any]) -> ToolResult:
        """Execute a tool by name with arguments."""
        handler = self._handlers.get(name)
        if not handler:
            return ToolResult.error_result(f"Tool not found: {name}")

        try:
            # Validate arguments against tool definition
            tool_def = self._tools.get(name)
            if tool_def:
                validation_error = self._validate_arguments(tool_def, arguments)
                if validation_error:
                    return ToolResult.error_result(validation_error)

            # Execute the handler
            result = await handler(**arguments)
            return result

        except Exception as e:
            logger.error(f"Tool execution error for {name}: {e}")
            return ToolResult.error_result(f"Execution failed: {str(e)}")

    def _validate_arguments(self, tool_def: ToolDefinition, arguments: Dict[str, Any]) -> Optional[str]:
        """Validate arguments against tool definition."""
        # Check required parameters
        for param in tool_def.parameters:
            if param.required and param.name not in arguments:
                return f"Missing required parameter: {param.name}"

            if param.name in arguments:
                value = arguments[param.name]
                # Basic type checking
                if param.type == "number" and not isinstance(value, (int, float)):
                    return f"Parameter {param.name} must be a number"
                elif param.type == "string" and not isinstance(value, str):
                    return f"Parameter {param.name} must be a string"
                elif param.type == "boolean" and not isinstance(value, bool):
                    return f"Parameter {param.name} must be a boolean"
                elif param.type == "array" and not isinstance(value, list):
                    return f"Parameter {param.name} must be an array"
                elif param.type == "object" and not isinstance(value, dict):
                    return f"Parameter {param.name} must be an object"

                # Check enum
                if param.enum and value not in param.enum:
                    return f"Parameter {param.name} must be one of: {param.enum}"

        return None


# Global tool registry
tool_registry = ToolRegistry()


def tool(
    name: str,
    description: str,
    parameters: List[ToolParameter] = None,
    returns: str = "object",
    requires_permission: bool = False,
    category: str = "general"
):
    """Decorator to register a tool."""
    def decorator(func: Callable[..., Awaitable[ToolResult]]):
        definition = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters or [],
            returns=returns,
            requires_permission=requires_permission,
            category=category
        )
        tool_registry.register(definition, func)
        return func
    return decorator


# ============================================================================
# Marswalk-Specific Tools
# ============================================================================

@tool(
    name="get_current_mission_state",
    description="Get the current mission state including active route, selected location, map view, and mission objectives.",
    parameters=[],
    returns="object with mission state",
    category="mission"
)
async def get_current_mission_state() -> ToolResult:
    """Get current mission state from the application."""
    # This will be connected to the real Marswalk app later
    # For now, return mock data structure
    return ToolResult.success_result({
        "mission": "EVA-01 · Jezero delta",
        "current_route": None,
        "selected_location": None,
        "map_view": "2d",
        "map_center": {"lat": 18.4545, "lon": 77.4040, "zoom": 15},
        "active_layers": ["viking", "ctx_regional"],
        "mission_objectives": [
            "Explore delta front",
            "Analyze sedimentary layers",
            "Collect rock samples"
        ],
        "eva_status": "planning"
    })


@tool(
    name="generate_route",
    description="Generate an exploration route between waypoints with hazard avoidance and slope constraints.",
    parameters=[
        ToolParameter("waypoints", "array", "List of waypoints as [{lat, lon}]", required=True),
        ToolParameter("max_slope", "number", "Maximum walkable slope in degrees (5-25)", required=False, default=15),
        ToolParameter("avoid_hazards", "boolean", "Whether to avoid known hazards", required=False, default=True),
        ToolParameter("route_type", "string", "Type of route: 'outbound', 'return', or 'loop'", required=False, default="outbound"),
    ],
    returns="RouteAnalysis object with profile, distance, elevation gain/loss, slopes",
    category="routing"
)
async def generate_route(
    waypoints: List[Dict[str, float]],
    max_slope: float = 15,
    avoid_hazards: bool = True,
    route_type: str = "outbound"
) -> ToolResult:
    """Generate a route using the Marswalk backend (mock for now)."""
    # Mock implementation - will connect to real /api/route/auto later
    if len(waypoints) < 2:
        return ToolResult.error_result("At least 2 waypoints required")

    # Calculate mock route analysis
    import math
    total_distance = 0
    elevation_gain = 0
    elevation_loss = 0
    max_slope_encountered = 0
    profile = []

    for i in range(len(waypoints) - 1):
        wp1 = waypoints[i]
        wp2 = waypoints[i + 1]

        # Haversine distance on Mars (radius = 3396190m)
        R = 3396190
        lat1, lon1 = math.radians(wp1["lat"]), math.radians(wp1["lon"])
        lat2, lon2 = math.radians(wp2["lat"]), math.radians(wp2["lon"])

        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        dist = 2 * R * math.asin(math.sqrt(a))
        total_distance += dist

        # Mock elevation and slope
        elev1 = -2500 + (i * 10)  # Mock elevation
        elev2 = -2500 + ((i + 1) * 10)
        slope = ((elev2 - elev1) / dist) * 100 if dist > 0 else 0
        slope_deg = math.degrees(math.atan(slope / 100))

        max_slope_encountered = max(max_slope_encountered, abs(slope_deg))
        if slope_deg > 0:
            elevation_gain += elev2 - elev1
        else:
            elevation_loss += elev1 - elev2

        profile.append({
            "lat": wp2["lat"],
            "lon": wp2["lon"],
            "dist_m": round(total_distance, 1),
            "elevation_m": round(elev2, 1),
            "slope_deg": round(slope_deg, 1)
        })

    return ToolResult.success_result({
        "profile": profile,
        "distance_m": round(total_distance, 1),
        "elevation_gain_m": round(elevation_gain, 1),
        "elevation_loss_m": round(elevation_loss, 1),
        "max_slope_deg": round(max_slope_encountered, 1),
        "avg_slope_deg": round(max_slope_encountered * 0.6, 1),
        "over_limit": max_slope_encountered > max_slope,
        "data_coverage_pct": 85.0,
        "method": "mock_hazard_avoidance",
        "source_dataset": "HiRISE DTM (mock)",
        "waypoints": waypoints
    }, metadata={"note": "Mock data - connect to real backend for actual terrain analysis"})


@tool(
    name="compare_routes",
    description="Compare two or more routes and provide analysis of differences.",
    parameters=[
        ToolParameter("routes", "array", "List of route objects to compare", required=True),
        ToolParameter("criteria", "array", "Criteria to compare: distance, max_slope, elevation_gain, data_coverage", required=False, default=["distance", "max_slope", "elevation_gain", "data_coverage"]),
    ],
    returns="Comparison analysis with recommendations",
    category="routing"
)
async def compare_routes(
    routes: List[Dict[str, Any]],
    criteria: List[str] = None
) -> ToolResult:
    """Compare multiple routes."""
    if len(routes) < 2:
        return ToolResult.error_result("At least 2 routes required for comparison")

    criteria = criteria or ["distance", "max_slope", "elevation_gain", "data_coverage"]

    comparison = {
        "routes": [],
        "best_by_criterion": {},
        "summary": ""
    }

    for i, route in enumerate(routes):
        route_info = {
            "index": i,
            "label": f"Route {chr(65 + i)}",
            "distance_m": route.get("distance_m", 0),
            "max_slope_deg": route.get("max_slope_deg", 0),
            "elevation_gain_m": route.get("elevation_gain_m", 0),
            "elevation_loss_m": route.get("elevation_loss_m", 0),
            "data_coverage_pct": route.get("data_coverage_pct", 0),
            "over_limit": route.get("over_limit", False)
        }
        comparison["routes"].append(route_info)

    # Find best for each criterion
    for criterion in criteria:
        if criterion == "distance":
            best = min(comparison["routes"], key=lambda r: r["distance_m"])
        elif criterion == "max_slope":
            best = min(comparison["routes"], key=lambda r: r["max_slope_deg"])
        elif criterion == "elevation_gain":
            best = min(comparison["routes"], key=lambda r: r["elevation_gain_m"])
        elif criterion == "data_coverage":
            best = max(comparison["routes"], key=lambda r: r["data_coverage_pct"])
        else:
            continue

        comparison["best_by_criterion"][criterion] = {
            "route": best["label"],
            "value": best.get(criterion.replace("_m", "").replace("_deg", ""))
        }

    # Generate summary
    route_a = comparison["routes"][0]
    route_b = comparison["routes"][1] if len(comparison["routes"]) > 1 else None

    if route_b:
        dist_diff_pct = abs(route_a["distance_m"] - route_b["distance_m"]) / max(route_a["distance_m"], 1) * 100
        slope_diff = abs(route_a["max_slope_deg"] - route_b["max_slope_deg"])

        summary_parts = []
        if route_a["max_slope_deg"] < route_b["max_slope_deg"]:
            summary_parts.append(f"Route A has lower max slope ({route_a['max_slope_deg']:.1f}° vs {route_b['max_slope_deg']:.1f}°)")
        else:
            summary_parts.append(f"Route B has lower max slope ({route_b['max_slope_deg']:.1f}° vs {route_a['max_slope_deg']:.1f}°)")

        if route_a["distance_m"] < route_b["distance_m"]:
            summary_parts.append(f"Route A is shorter ({route_a['distance_m']:.0f}m vs {route_b['distance_m']:.0f}m)")
        else:
            summary_parts.append(f"Route B is shorter ({route_b['distance_m']:.0f}m vs {route_a['distance_m']:.0f}m)")

        if dist_diff_pct > 25 or slope_diff > 5:
            summary_parts.append("Significant difference detected (>25% distance or >5° slope)")

        summary_parts.append("Preliminary terrain assessment from elevation data only")
        comparison["summary"] = ". ".join(summary_parts)

    return ToolResult.success_result(comparison)


@tool(
    name="analyse_terrain",
    description="Analyze terrain at a specific location or along a path. Returns elevation, slope, and data source information.",
    parameters=[
        ToolParameter("lat", "number", "Latitude in degrees", required=True),
        ToolParameter("lon", "number", "Longitude in degrees", required=True),
        ToolParameter("radius", "number", "Analysis radius in meters (for area analysis)", required=False, default=0),
    ],
    returns="Terrain analysis with elevation, slope, data source, and coverage",
    category="science"
)
async def analyse_terrain(lat: float, lon: float, radius: float = 0) -> ToolResult:
    """Analyze terrain at a location (mock - connects to /api/elevation)."""
    # Mock implementation
    import random

    # Simulate rover-measured elevation if near known waypoints
    rover_waypoints = [
        {"lat": 18.451331, "lon": 77.401257, "elev": -2520.9, "sol": 13},
        {"lat": 18.4589, "lon": 77.4062, "elev": -2518.5, "sol": 2022},
    ]

    nearest = min(rover_waypoints, key=lambda w: (
        (w["lat"] - lat)**2 + (w["lon"] - lon)**2
    )**0.5)

    distance_km = ((nearest["lat"] - lat)**2 + (nearest["lon"] - lon)**2)**0.5 * 111  # rough km

    if distance_km < 0.04:  # Within 40m
        return ToolResult.success_result({
            "elevation_m": nearest["elev"],
            "slope_deg": None,
            "source_dataset": f"Perseverance rover-measured (Sol {nearest['sol']})",
            "data_available": True,
            "data_type": "measured",
            "note": "Rover-measured elevation at waypoint"
        }, metadata={"rover_distance_m": round(distance_km * 1000)})

    # Mock DEM data
    elevation = -2520 + random.uniform(-50, 50)
    slope = random.uniform(0, 20)

    return ToolResult.success_result({
        "elevation_m": round(elevation, 1),
        "slope_deg": round(slope, 1),
        "source_dataset": "HiRISE DTM (mock)",
        "data_available": True,
        "data_type": "dem_derived",
        "note": "Digital Elevation Model derived slope and elevation"
    }, metadata={"note": "Mock data - connect to real backend for actual terrain analysis"})


@tool(
    name="calculate_slope",
    description="Calculate slope between two points given their elevations and distance.",
    parameters=[
        ToolParameter("elevation1", "number", "Elevation of first point in meters", required=True),
        ToolParameter("elevation2", "number", "Elevation of second point in meters", required=True),
        ToolParameter("distance_m", "number", "Horizontal distance between points in meters", required=True),
    ],
    returns="Slope in degrees and percentage",
    category="science"
)
async def calculate_slope(elevation1: float, elevation2: float, distance_m: float) -> ToolResult:
    """Calculate slope between two points."""
    import math

    if distance_m <= 0:
        return ToolResult.error_result("Distance must be positive")

    elevation_diff = elevation2 - elevation1
    slope_ratio = elevation_diff / distance_m
    slope_deg = math.degrees(math.atan(slope_ratio))
    slope_pct = slope_ratio * 100

    return ToolResult.success_result({
        "slope_deg": round(slope_deg, 2),
        "slope_percentage": round(slope_pct, 2),
        "elevation_difference_m": round(elevation_diff, 2),
        "horizontal_distance_m": distance_m,
        "classification": "gentle" if abs(slope_deg) < 5 else "moderate" if abs(slope_deg) < 15 else "steep"
    })


@tool(
    name="analyse_environment",
    description="Analyze environmental conditions for a route or location including radiation, temperature, and weather.",
    parameters=[
        ToolParameter("lat", "number", "Latitude", required=True),
        ToolParameter("lon", "number", "Longitude", required=True),
        ToolParameter("duration_hours", "number", "EVA duration in hours", required=False, default=8),
    ],
    returns="Environmental analysis with radiation dose, temperature range, weather",
    category="science"
)
async def analyse_environment(lat: float, lon: float, duration_hours: float = 8) -> ToolResult:
    """Analyze environmental conditions (mock - connects to /api/eva and /api/meda)."""
    # Mock implementation
    import random

    # Base radiation from Curiosity RAD at Gale crater (0.21 mGy/day)
    # Jezero is at similar latitude, scale by duration
    daily_dose = 0.21  # mGy/day
    mission_dose = daily_dose * (duration_hours / 24)

    return ToolResult.success_result({
        "radiation": {
            "dose_estimate_mgy": round(mission_dose, 4),
            "dose_rate_mgy_per_day": daily_dose,
            "source": "Curiosity RAD measurement at Gale Crater (scaled)",
            "note": "Estimate based on Gale Crater measurements, not Jezero-specific"
        },
        "temperature": {
            "min_c": -90,
            "max_c": -10,
            "avg_c": -60,
            "source": "MEDA typical range (mock)"
        },
        "pressure": {
            "avg_pa": 840,
            "source": "MEDA typical (mock)"
        },
        "wind": {
            "avg_ms": 5,
            "max_ms": 20,
            "source": "MEDA typical (mock)"
        },
        "duration_hours": duration_hours,
        "location": {"lat": lat, "lon": lon}
    }, metadata={"note": "Mock data - connect to real MEDA and radiation backend"})


@tool(
    name="retrieve_mission_information",
    description="Retrieve mission information about Perseverance, Jezero Crater, or Mars exploration.",
    parameters=[
        ToolParameter("topic", "string", "Topic to retrieve: perseverance, jezero, mars_missions, geology, meda, radiation", required=True),
        ToolParameter("query", "string", "Specific question or search query", required=False),
    ],
    returns="Mission information with sources",
    category="knowledge"
)
async def retrieve_mission_information(topic: str, query: str = None) -> ToolResult:
    """Retrieve mission information from knowledge base."""
    # This will use RAG when implemented
    knowledge_base = {
        "perseverance": {
            "name": "Perseverance Rover (Mars 2020)",
            "landing_date": "2021-02-18",
            "landing_site": "Octavia E. Butler Landing, Jezero Crater",
            "mission_duration": "Active (landed Sol 0, currently Sol 2000+)",
            "instruments": ["MEDA", "MOXIE", "PIXL", "SHERLOC", "SuperCam", "Mastcam-Z", "RIMFAX"],
            "total_distance": "30+ km",
            "samples_collected": "20+ rock cores",
            "key_findings": [
                "Confirmed ancient lake delta in Jezero",
                "Organic molecules in sedimentary rocks",
                "Igneous rocks on crater floor",
                "Dust devil and weather observations"
            ]
        },
        "jezero": {
            "name": "Jezero Crater",
            "diameter_km": 45,
            "location": "18.4°N, 77.5°E (Isidis Planitia)",
            "age": "~3.9 billion years",
            "key_features": [
                "Ancient river delta (western crater)",
                "Inlet valleys (NW and NE)",
                "Outlet valley (eastern rim)",
                "Crater floor igneous units",
                "Marginal carbonate deposits"
            ],
            "scientific_importance": "Prime astrobiology target - ancient lake with delta preserves potential biosignatures"
        },
        "mars_missions": {
            "active": ["Perseverance", "Curiosity", "InSight (ended)", "Mars Reconnaissance Orbiter", "MAVEN", "ExoMars TGO"],
            "past": ["Spirit", "Opportunity", "Phoenix", "Pathfinder", "Viking 1&2"],
            "upcoming": ["Mars Sample Return", "ExoMars Rosalind Franklin"]
        },
        "geology": {
            "jezero_units": [
                "Crater floor fractured rough (Igneous)",
                "Seitah (Igneous, olivine-rich)",
                "Maaz (Igneous, upper unit)",
                "Delta front (Sedimentary)",
                "Kodiak butte (Delta remnant)",
                "Margin carbonate (Lacustrine)"
            ],
            "key_minerals": ["Olivine", "Pyroxene", "Carbonate", "Fe/Mg smectite", "Hydrated silica"]
        },
        "meda": {
            "name": "Mars Environmental Dynamics Analyzer",
            "measurements": ["Air temperature", "Ground temperature", "Pressure", "Humidity", "Wind speed/direction", "Radiation", "Dust optical depth"],
            "typical_range": {
                "temp_c": [-90, -10],
                "pressure_pa": [700, 900],
                "wind_ms": [0, 20]
            }
        },
        "radiation": {
            "gcr_dose_rate": "0.21 mGy/day (Curiosity RAD at Gale)",
            "spe_events": "Sporadic, can increase dose significantly",
            "shielding": "Atmosphere provides minimal shielding (~10-20 g/cm2)",
            "eva_limits": "NASA career limits vary by age/sex (~600-1200 mSv)"
        }
    }

    topic_lower = topic.lower()
    if topic_lower not in knowledge_base:
        return ToolResult.error_result(f"Unknown topic: {topic}. Available: {list(knowledge_base.keys())}")

    info = knowledge_base[topic_lower]

    # If query provided, filter/summarize
    if query:
        # Simple keyword matching for now
        query_lower = query.lower()
        filtered = {k: v for k, v in info.items() if query_lower in str(v).lower()}
        if filtered:
            info = filtered

    return ToolResult.success_result({
        "topic": topic,
        "information": info,
        "source": "Marswalk internal knowledge base",
        "query": query
    })


@tool(
    name="search_knowledge_base",
    description="Search the local knowledge base for scientific documents, papers, or mission documentation.",
    parameters=[
        ToolParameter("query", "string", "Search query", required=True),
        ToolParameter("top_k", "number", "Number of results to return", required=False, default=5),
        ToolParameter("filter", "object", "Optional filters: source_type, date_range", required=False),
    ],
    returns="List of relevant documents with snippets and sources",
    category="knowledge"
)
async def search_knowledge_base(query: str, top_k: int = 5, filter: Dict = None) -> ToolResult:
    """Search the knowledge base (will use ChromaDB when implemented)."""
    # Mock implementation - returns placeholder
    return ToolResult.success_result({
        "query": query,
        "results": [
            {
                "title": "Jezero Crater Delta Stratigraphy",
                "source": "NASA Mars 2020 Mission Documentation",
                "snippet": "The Jezero delta shows well-preserved stratigraphic sequences...",
                "relevance": 0.95,
                "type": "mission_doc"
            },
            {
                "title": "Perseverance MEDA Observations",
                "source": "MEDA Team Publications",
                "snippet": "Seasonal pressure variations and dust cycle measurements...",
                "relevance": 0.88,
                "type": "science_paper"
            }
        ][:top_k],
        "note": "Mock results - ChromaDB integration pending"
    })


@tool(
    name="highlight_map_location",
    description="Highlight a location on the Marswalk map with a marker and optional info panel.",
    parameters=[
        ToolParameter("lat", "number", "Latitude in degrees", required=True),
        ToolParameter("lon", "number", "Longitude in degrees", required=True),
        ToolParameter("label", "string", "Label for the marker", required=False),
        ToolParameter("info", "string", "Additional information to display", required=False),
        ToolParameter("marker_type", "string", "Type: 'point', 'stop', 'habitat', 'waypoint'", required=False, default="point"),
    ],
    returns="Confirmation of map action",
    category="map"
)
async def highlight_map_location(
    lat: float,
    lon: float,
    label: str = None,
    info: str = None,
    marker_type: str = "point"
) -> ToolResult:
    """Highlight a location on the map (sends action to frontend)."""
    return ToolResult.success_result({
        "action": "highlight_location",
        "lat": lat,
        "lon": lon,
        "label": label or f"Location ({lat:.4f}, {lon:.4f})",
        "info": info,
        "marker_type": marker_type,
        "frontend_action": True
    }, metadata={"frontend_action": True})


@tool(
    name="zoom_map",
    description="Zoom the map to a specific location or bounds.",
    parameters=[
        ToolParameter("lat", "number", "Center latitude", required=False),
        ToolParameter("lon", "number", "Center longitude", required=False),
        ToolParameter("zoom", "number", "Zoom level (3-19)", required=False, default=15),
        ToolParameter("bounds", "array", "Bounds as [[south, west], [north, east]]", required=False),
    ],
    returns="Confirmation of map action",
    category="map"
)
async def zoom_map(
    lat: float = None,
    lon: float = None,
    zoom: int = 15,
    bounds: List[List[float]] = None
) -> ToolResult:
    """Zoom map to location or bounds (sends action to frontend)."""
    if bounds:
        return ToolResult.success_result({
            "action": "fit_bounds",
            "bounds": bounds,
            "frontend_action": True
        }, metadata={"frontend_action": True})

    if lat is None or lon is None:
        return ToolResult.error_result("Either lat/lon or bounds required")

    return ToolResult.success_result({
        "action": "set_view",
        "lat": lat,
        "lon": lon,
        "zoom": zoom,
        "frontend_action": True
    }, metadata={"frontend_action": True})


@tool(
    name="change_map_layer",
    description="Change the active map layer in the Marswalk application.",
    parameters=[
        ToolParameter("layer_id", "string", "Layer ID: viking, ctx_regional, ctx_trn, hirise, mola, themis, or backend overlay ID", required=True),
        ToolParameter("visible", "boolean", "Whether to show or hide the layer", required=True),
        ToolParameter("opacity", "number", "Layer opacity 0-1", required=False, default=1.0),
    ],
    returns="Confirmation of layer change",
    category="map"
)
async def change_map_layer(layer_id: str, visible: bool, opacity: float = 1.0) -> ToolResult:
    """Change map layer visibility (sends action to frontend)."""
    valid_layers = ["viking", "ctx_regional", "ctx_trn", "hirise", "mola", "themis"]
    if layer_id not in valid_layers and not layer_id.startswith("overlay_"):
        return ToolResult.error_result(f"Unknown layer: {layer_id}. Valid: {valid_layers}")

    return ToolResult.success_result({
        "action": "change_layer",
        "layer_id": layer_id,
        "visible": visible,
        "opacity": opacity,
        "frontend_action": True
    }, metadata={"frontend_action": True})


@tool(
    name="display_route_on_map",
    description="Display a generated route on the Marswalk map with slope coloring.",
    parameters=[
        ToolParameter("route_id", "string", "Route identifier: 'A' or 'B'", required=True),
        ToolParameter("route_data", "object", "Route analysis data from generate_route", required=True),
        ToolParameter("show_profile", "boolean", "Whether to show elevation profile chart", required=False, default=True),
    ],
    returns="Confirmation of route display",
    category="map"
)
async def display_route_on_map(route_id: str, route_data: Dict[str, Any], show_profile: bool = True) -> ToolResult:
    """Display route on map (sends action to frontend)."""
    return ToolResult.success_result({
        "action": "display_route",
        "route_id": route_id,
        "route_data": route_data,
        "show_profile": show_profile,
        "frontend_action": True
    }, metadata={"frontend_action": True})


@tool(
    name="navigate_application",
    description="Navigate to a specific section or tab in the Marswalk application.",
    parameters=[
        ToolParameter("target", "string", "Target: 'layers', 'route', 'environment', 'records', 'chat', '2d', '3d'", required=True),
        ToolParameter("action", "string", "Action: 'open', 'close', 'toggle'", required=False, default="open"),
    ],
    returns="Confirmation of navigation",
    category="map"
)
async def navigate_application(target: str, action: str = "open") -> ToolResult:
    """Navigate application UI (sends action to frontend)."""
    valid_targets = ["layers", "route", "environment", "records", "chat", "2d", "3d"]
    if target not in valid_targets:
        return ToolResult.error_result(f"Unknown target: {target}. Valid: {valid_targets}")

    return ToolResult.success_result({
        "action": f"navigate_{action}",
        "target": target,
        "frontend_action": True
    }, metadata={"frontend_action": True})


# Initialize all tools
def init_tools():
    """Initialize all tools (called at startup)."""
    # Tools are registered via decorators
    logger.info(f"Initialized {len(tool_registry._tools)} tools")
    for name in tool_registry._tools:
        logger.info(f"  - {name} ({tool_registry._tools[name].category})")