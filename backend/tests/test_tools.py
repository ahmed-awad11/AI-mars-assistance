"""
Marswalk AI Assistant - Tool Tests
"""
import pytest
import asyncio
from backend.tools import (
    tool_registry, ToolResult,
    get_current_mission_state, generate_route, compare_routes,
    analyse_terrain, calculate_slope, analyse_environment,
    retrieve_mission_information, search_knowledge_base,
    highlight_map_location, zoom_map, change_map_layer,
    display_route_on_map, navigate_application
)


class TestToolRegistry:
    """Test tool registry functionality."""

    def test_registry_initialized(self):
        """Verify all tools are registered."""
        tools = tool_registry.list_tools()
        assert len(tools) >= 18

        tool_names = {t.name for t in tools}
        expected = {
            'get_current_mission_state', 'generate_route', 'compare_routes',
            'analyse_terrain', 'calculate_slope', 'analyse_environment',
            'retrieve_mission_information', 'search_knowledge_base',
            'highlight_map_location', 'zoom_map', 'change_map_layer',
            'display_route_on_map', 'navigate_application'
        }
        assert expected.issubset(tool_names)

    def test_get_tool_definition(self):
        """Test getting tool definition."""
        tool = tool_registry.get_tool('generate_route')
        assert tool is not None
        assert tool.name == 'generate_route'
        assert 'waypoints' in [p.name for p in tool.parameters]

    def test_tool_json_schema(self):
        """Test JSON schema generation."""
        tool = tool_registry.get_tool('calculate_slope')
        schema = tool.to_json_schema()
        assert schema['name'] == 'calculate_slope'
        assert 'parameters' in schema
        assert 'properties' in schema['parameters']
        assert 'elevation1' in schema['parameters']['properties']


class TestMissionTools:
    """Test mission-related tools."""

    @pytest.mark.asyncio
    async def test_get_current_mission_state(self):
        result = await get_current_mission_state()
        assert result.success
        assert 'mission' in result.data
        assert result.data['mission'] == 'EVA-01 · Jezero delta'

    @pytest.mark.asyncio
    async def test_generate_route_valid(self):
        waypoints = [
            {"lat": 18.45, "lon": 77.40},
            {"lat": 18.46, "lon": 77.41}
        ]
        result = await generate_route(waypoints)
        assert result.success
        assert 'profile' in result.data
        assert 'distance_m' in result.data
        assert len(result.data['profile']) == 1

    @pytest.mark.asyncio
    async def test_generate_route_invalid(self):
        result = await generate_route([{"lat": 18.45, "lon": 77.40}])
        assert not result.success
        assert 'At least 2 waypoints' in result.error

    @pytest.mark.asyncio
    async def test_compare_routes(self):
        routes = [
            {"distance_m": 1000, "max_slope_deg": 10, "elevation_gain_m": 50, "data_coverage_pct": 90},
            {"distance_m": 1200, "max_slope_deg": 8, "elevation_gain_m": 30, "data_coverage_pct": 85}
        ]
        result = await compare_routes(routes)
        assert result.success
        assert 'routes' in result.data
        assert 'best_by_criterion' in result.data
        assert 'summary' in result.data


class TestScienceTools:
    """Test science/analysis tools."""

    @pytest.mark.asyncio
    async def test_analyse_terrain(self):
        result = await analyse_terrain(18.45, 77.40)
        assert result.success
        assert 'elevation_m' in result.data
        assert 'slope_deg' in result.data
        assert 'source_dataset' in result.data

    @pytest.mark.asyncio
    async def test_calculate_slope(self):
        result = await calculate_slope(-2500, -2490, 100)
        assert result.success
        assert 'slope_deg' in result.data
        assert 'slope_percentage' in result.data
        assert result.data['slope_deg'] > 0

    @pytest.mark.asyncio
    async def test_calculate_slope_zero_distance(self):
        result = await calculate_slope(-2500, -2490, 0)
        assert not result.success

    @pytest.mark.asyncio
    async def test_analyse_environment(self):
        result = await analyse_environment(18.45, 77.40, 8)
        assert result.success
        assert 'radiation' in result.data
        assert 'temperature' in result.data
        assert result.data['radiation']['dose_estimate_mgy'] > 0

    @pytest.mark.asyncio
    async def test_retrieve_mission_information(self):
        result = await retrieve_mission_information('perseverance')
        assert result.success
        assert 'information' in result.data
        assert 'landing_date' in result.data['information']

    @pytest.mark.asyncio
    async def test_retrieve_mission_information_unknown(self):
        result = await retrieve_mission_information('unknown_topic')
        assert not result.success

    @pytest.mark.asyncio
    async def test_search_knowledge_base(self):
        result = await search_knowledge_base('Jezero delta', top_k=3)
        assert result.success
        assert 'results' in result.data
        assert len(result.data['results']) > 0


class TestMapTools:
    """Test map interaction tools."""

    @pytest.mark.asyncio
    async def test_highlight_map_location(self):
        result = await highlight_map_location(18.45, 77.40, "Test Site")
        assert result.success
        assert result.data['action'] == 'highlight_location'
        assert result.data['lat'] == 18.45
        assert result.metadata.get('frontend_action') is True

    @pytest.mark.asyncio
    async def test_zoom_map_coords(self):
        result = await zoom_map(lat=18.45, lon=77.40, zoom=16)
        assert result.success
        assert result.data['action'] == 'set_view'
        assert result.data['zoom'] == 16

    @pytest.mark.asyncio
    async def test_zoom_map_bounds(self):
        bounds = [[18.4, 77.3], [18.5, 77.5]]
        result = await zoom_map(bounds=bounds)
        assert result.success
        assert result.data['action'] == 'fit_bounds'

    @pytest.mark.asyncio
    async def test_change_map_layer(self):
        result = await change_map_layer('mola', True, 0.5)
        assert result.success
        assert result.data['action'] == 'change_layer'
        assert result.data['layer_id'] == 'mola'
        assert result.data['opacity'] == 0.5

    @pytest.mark.asyncio
    async def test_change_map_layer_invalid(self):
        result = await change_map_layer('invalid_layer', True)
        assert not result.success

    @pytest.mark.asyncio
    async def test_display_route_on_map(self):
        route_data = {
            "profile": [{"lat": 18.45, "lon": 77.40, "dist_m": 100, "elevation_m": -2500, "slope_deg": 5}],
            "distance_m": 100
        }
        result = await display_route_on_map('A', route_data)
        assert result.success
        assert result.data['action'] == 'display_route'
        assert result.data['route_id'] == 'A'

    @pytest.mark.asyncio
    async def test_navigate_application(self):
        result = await navigate_application('route', 'open')
        assert result.success
        assert result.data['action'] == 'navigate_open'
        assert result.data['target'] == 'route'

    @pytest.mark.asyncio
    async def test_navigate_application_invalid(self):
        result = await navigate_application('invalid', 'open')
        assert not result.success


class TestToolExecution:
    """Test tool execution through registry."""

    @pytest.mark.asyncio
    async def test_execute_valid_tool(self):
        result = await tool_registry.execute('calculate_slope', {
            'elevation1': -2500,
            'elevation2': -2490,
            'distance_m': 100
        })
        assert result.success
        assert 'slope_deg' in result.data

    @pytest.mark.asyncio
    async def test_execute_invalid_tool(self):
        result = await tool_registry.execute('nonexistent_tool', {})
        assert not result.success
        assert 'not found' in result.error

    @pytest.mark.asyncio
    async def test_execute_missing_required_param(self):
        result = await tool_registry.execute('calculate_slope', {
            'elevation1': -2500
            # missing elevation2 and distance_m
        })
        assert not result.success
        assert 'Missing required parameter' in result.error


if __name__ == '__main__':
    pytest.main([__file__, '-v'])