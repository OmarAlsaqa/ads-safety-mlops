import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Ensure src and project root are in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from feast import FeatureStore
    from feature_store.features import (
        ip_entity,
        app_entity,
        device_entity,
        channel_entity,
        ip_features_view,
        app_features_view,
        channel_features_view,
        device_features_view,
    )
except ImportError as e:
    print(f"Notice: Feast import issue: {e}. Run: pip install 'feast[redis,aws]'")
    sys.exit(1)


def materialize():
    repo_path = str(BASE_DIR / "feature_store")
    print(f"🍱 Initializing Feast Feature Store at: {repo_path}")

    store = FeatureStore(repo_path=repo_path)

    # 1. Apply feature definitions to registry
    print("1. Applying Feast feature definitions & updating registry...")
    feast_objects = [
        ip_entity,
        app_entity,
        device_entity,
        channel_entity,
        ip_features_view,
        app_features_view,
        channel_features_view,
        device_features_view,
    ]
    store.apply(feast_objects)

    # 2. Materialize features covering historical dataset to present
    start_date = datetime(2017, 11, 1)
    end_date = datetime.utcnow() + timedelta(days=1)
    print(f"2. Materializing features to Online Store ({start_date.strftime('%Y-%m-%d')} -> {end_date.strftime('%Y-%m-%d')})...")

    try:
        store.materialize(start_date=start_date, end_date=end_date)
        print("✅ Feast Online Store Successfully Materialized!")
    except Exception as e:
        print(f"Notice on materialization: {e}")


if __name__ == "__main__":
    materialize()
