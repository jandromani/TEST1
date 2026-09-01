from assetgraph.adapters import STACItemAdapter


def test_stac_adapter_normalizes_provider_item():
    item = {
        "type": "Feature",
        "stac_version": "1.0.0",
        "id": "ACQ-001",
        "collection": "customer-imagery",
        "geometry": {"type": "Polygon", "coordinates": []},
        "properties": {
            "datetime": "2026-09-01T10:00:00Z",
            "platform": "uav-01",
            "gsd": 0.15,
            "proj:epsg": 32630
        },
        "assets": {
            "visual": {"href": "file:///data/acq001.tif", "type": "image/tiff; application=geotiff"}
        }
    }
    env = list(STACItemAdapter().normalize(item))[0]
    assert env.source_id == "ACQ-001"
    assert env.sensor == "uav-01"
    assert env.collection == "customer-imagery"
    assert env.telemetry["gsd"] == 0.15
    assert env.metadata["asset_key"] == "visual"
