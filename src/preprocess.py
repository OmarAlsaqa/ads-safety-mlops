import os
import sys
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from schema import RawAdClickSchema, ProcessedAdClickSchema


def build_pyg_graph(df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15) -> Data:
    """
    Constructs a PyTorch Geometric Graph Data Object (T-Finance / GraphNC compatible):
    - Nodes: Click events with 15 normalized behavioral, temporal delta, and interaction features
    - Edges: Co-occurrence relations across shared IP and App/Channel entities
    - Targets: Binary fraud conversion labels
    - Masks: Chronological train, validation, and test boolean masks
    """
    feature_cols = [
        "hour", "day", "ip_click_count", "ip_unique_apps", "app_freq", "channel_freq", "device_freq",
        "next_click_delta", "prev_click_delta", "ip_device_os_cumcount", "ip_app_cumcount",
        "ip_hh_app_count", "ip_hh_device_count", "app_channel_count", "ip_unique_channels"
    ]
    X_raw = df[feature_cols].values.astype(np.float32)
    X_norm = (X_raw - X_raw.mean(axis=0)) / (X_raw.std(axis=0) + 1e-6)
    x = torch.tensor(X_norm, dtype=torch.float)

    # Categorical entity matrix for learnable embeddings (app, channel, device, os, hour)
    cat_cols = ["app", "channel", "device", "os", "hour"]
    x_cat = torch.tensor(df[cat_cols].values.astype(np.int64), dtype=torch.long)

    # 1. Build Multi-Relational Co-occurrence Edges
    src_list = []
    dst_list = []

    # Relation 1: Shared IP (connect consecutive clicks per IP)
    ip_groups = df.groupby("ip").indices
    for indices in ip_groups.values():
        if len(indices) > 1:
            idx_arr = indices[:30]
            src_list.extend(idx_arr[:-1])
            dst_list.extend(idx_arr[1:])
            src_list.extend(idx_arr[1:])
            dst_list.extend(idx_arr[:-1])

    # Relation 2: Shared App & Channel (detects campaign target clusters)
    app_chan_groups = df.groupby(["app", "channel"]).indices
    for indices in app_chan_groups.values():
        if 1 < len(indices) <= 20:
            idx_arr = indices[:20]
            src_list.extend(idx_arr[:-1])
            dst_list.extend(idx_arr[1:])
            src_list.extend(idx_arr[1:])
            dst_list.extend(idx_arr[:-1])

    # Relation 3: Shared IP & Channel (detects cross-app spamming on same channel)
    ip_chan_groups = df.groupby(["ip", "channel"]).indices
    for indices in ip_chan_groups.values():
        if 1 < len(indices) <= 20:
            idx_arr = indices[:20]
            src_list.extend(idx_arr[:-1])
            dst_list.extend(idx_arr[1:])
            src_list.extend(idx_arr[1:])
            dst_list.extend(idx_arr[:-1])

    # Relation 4: Shared Device Fingerprint (ip, device, os)
    dev_fp_groups = df.groupby(["ip", "device", "os"]).indices
    for indices in dev_fp_groups.values():
        if 1 < len(indices) <= 20:
            idx_arr = indices[:20]
            src_list.extend(idx_arr[:-1])
            dst_list.extend(idx_arr[1:])
            src_list.extend(idx_arr[1:])
            dst_list.extend(idx_arr[:-1])

    # Self-loops for all nodes
    node_ids = list(range(len(df)))
    src_list.extend(node_ids)
    dst_list.extend(node_ids)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_index = torch.unique(edge_index, dim=1)

    # 2. Labels and Chronological Masks
    y = torch.tensor(df["is_attributed"].values, dtype=torch.long)
    num_nodes = len(df)
    train_cutoff = int(num_nodes * train_ratio)
    val_cutoff = int(num_nodes * (train_ratio + val_ratio))

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[:train_cutoff] = True
    val_mask[train_cutoff:val_cutoff] = True
    test_mask[val_cutoff:] = True

    return Data(
        x=x.contiguous(),
        x_cat=x_cat.contiguous(),
        edge_index=edge_index.contiguous(),
        y=y.contiguous(),
        train_mask=train_mask.contiguous(),
        val_mask=val_mask.contiguous(),
        test_mask=test_mask.contiguous()
    )


def preprocess(input_path: str = None):
    raw_sample_path = "data/raw/train_sample.csv"
    raw_full_path = "data/raw/train.csv"
    output_dir = "data/processed"
    prod_dir = "data/simulated_production"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(prod_dir, exist_ok=True)

    # 1. Resolve raw input file path
    target_path = input_path or (sys.argv[1] if len(sys.argv) > 1 else None)
    dtypes = {
        "ip": "uint32",
        "app": "uint16",
        "device": "uint16",
        "os": "uint16",
        "channel": "uint16",
        "is_attributed": "uint8"
    }

    n_rows_env = os.getenv("N_ROWS", "1000000")
    nrows = None if n_rows_env in ["0", "all", "none", "None"] else int(n_rows_env)

    print("1. Loading raw ad-click dataset...")
    if target_path and os.path.exists(target_path):
        raw_path = target_path
        if "sample" in target_path or not nrows:
            print(f"   -> Loading dataset from target path: {raw_path}")
            df = pd.read_csv(raw_path, dtype=dtypes)
        else:
            print(f"   -> Loading dataset slice ({nrows:,} rows) from target path: {raw_path}")
            df = pd.read_csv(raw_path, nrows=nrows, dtype=dtypes)
    elif os.path.exists(raw_full_path):
        raw_path = raw_full_path
        if nrows:
            print(f"   -> Loading large dataset slice ({nrows:,} rows) from: {raw_path}")
            df = pd.read_csv(raw_path, nrows=nrows, dtype=dtypes)
        else:
            print(f"   -> Loading FULL dataset from: {raw_path}")
            df = pd.read_csv(raw_path, dtype=dtypes)
    elif os.path.exists(raw_sample_path):
        raw_path = raw_sample_path
        print(f"   -> Loading sample dataset from: {raw_path}")
        df = pd.read_csv(raw_path, dtype=dtypes)
    else:
        raise FileNotFoundError("No raw ad click CSV found in data/raw/.")

    print(f"   -> Loaded {len(df):,} records successfully.")

    print("2. Validating raw click events with Pandera (RawAdClickSchema)...")
    RawAdClickSchema.validate(df)

    print("3. Feature engineering: Behavioral entity frequencies & temporal dynamics...")
    df["dt"] = pd.to_datetime(df["click_time"])
    df["event_timestamp"] = df["dt"]
    df["hour"] = df["dt"].dt.hour
    df["day"] = df["dt"].dt.day
    df["date_str"] = df["dt"].dt.strftime("%Y-%m-%d")

    # Chronological sort to eliminate temporal leakage
    df = df.sort_values("dt").reset_index(drop=True)

    # 1. Behavioral frequency aggregations
    df["ip_click_count"] = df.groupby("ip")["app"].transform("count").astype("float32")
    df["ip_unique_apps"] = df.groupby("ip")["app"].transform("nunique").astype("float32")
    df["app_freq"] = df.groupby("app")["ip"].transform("count").astype("float32")
    df["channel_freq"] = df.groupby("channel")["ip"].transform("count").astype("float32")
    df["device_freq"] = df.groupby("device")["ip"].transform("count").astype("float32")

    # 2. Advanced Temporal Deltas & Sequences (Kaggle Insights)
    # Next click interval in seconds for the same (ip, app, device, os)
    next_sec = df.groupby(["ip", "app", "device", "os"])["dt"].shift(-1)
    next_delta = (pd.to_datetime(next_sec) - df["dt"]).dt.total_seconds().fillna(3600.0)
    df["next_click_delta"] = np.log1p(np.clip(next_delta, 0.0, 86400.0)).astype("float32")

    # Previous click interval in seconds for the same (ip, channel)
    prev_sec = df.groupby(["ip", "channel"])["dt"].shift(1)
    prev_delta = (df["dt"] - pd.to_datetime(prev_sec)).dt.total_seconds().fillna(3600.0)
    df["prev_click_delta"] = np.log1p(np.clip(prev_delta, 0.0, 86400.0)).astype("float32")

    # Cumulative sequential counters
    df["ip_device_os_cumcount"] = np.log1p(df.groupby(["ip", "device", "os"]).cumcount()).astype("float32")
    df["ip_app_cumcount"] = np.log1p(df.groupby(["ip", "app"]).cumcount()).astype("float32")

    # 3. High-Order Cross Interactions & Diversity
    df["ip_hh_app_count"] = df.groupby(["ip", "hour", "app"])["channel"].transform("count").astype("float32")
    df["ip_hh_device_count"] = df.groupby(["ip", "hour", "device"])["channel"].transform("count").astype("float32")
    df["app_channel_count"] = df.groupby(["app", "channel"])["ip"].transform("count").astype("float32")
    df["ip_unique_channels"] = df.groupby("ip")["channel"].transform("nunique").astype("float32")

    print("4. Validating processed dataset with Pandera (ProcessedAdClickSchema)...")
    ProcessedAdClickSchema.validate(df)

    print("5. Performing chronological day/hour split...")
    num_records = len(df)
    train_end = int(num_records * 0.70)
    val_end = int(num_records * 0.85)

    train_df = df.iloc[:train_end].copy().drop(columns=["dt", "date_str"])
    val_df = df.iloc[train_end:val_end].copy().drop(columns=["dt", "date_str"])
    test_df = df.iloc[val_end:].copy().drop(columns=["dt", "date_str"])
    drift_df = df.iloc[val_end:].copy() # Production drift stream

    # Export Tabular splits
    train_df.to_parquet(f"{output_dir}/train.parquet", index=False)
    val_df.to_parquet(f"{output_dir}/val.parquet", index=False)
    test_df.to_parquet(f"{output_dir}/test.parquet", index=False)
    drift_df.drop(columns=["dt"]).to_csv(f"{prod_dir}/production_drift_stream.csv", index=False)

    print("6. Constructing and serializing PyTorch Geometric Graph Tensors...")
    graph_data = build_pyg_graph(df, train_ratio=0.70, val_ratio=0.15)

    # Save full graph with masks and split subsets
    torch.save(graph_data, f"{output_dir}/ad_click_graph.pt")
    torch.save(graph_data, f"{output_dir}/train_graph.pt")
    torch.save(graph_data, f"{output_dir}/val_graph.pt")
    torch.save(graph_data, f"{output_dir}/test_graph.pt")

    print(f"\n✅ Preprocessing & Graph Construction Complete!\n"
          f" - Total Nodes:              {graph_data.num_nodes:,}\n"
          f" - Total Graph Edges:        {graph_data.edge_index.shape[1]:,}\n"
          f" - Training Set (70%):       {len(train_df):,} clicks (train_mask: {graph_data.train_mask.sum():,})\n"
          f" - Validation Set (15%):     {len(val_df):,} clicks (val_mask: {graph_data.val_mask.sum():,})\n"
          f" - Test Set (15%):           {len(test_df):,} clicks (test_mask: {graph_data.test_mask.sum():,})\n"
          f" - Production Drift Stream:  {len(drift_df):,} rows\n"
          f" - Graph Tensor Serialized:  {output_dir}/ad_click_graph.pt")


if __name__ == "__main__":
    preprocess()