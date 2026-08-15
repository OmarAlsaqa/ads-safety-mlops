import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from schema import RawAdClickSchema, ProcessedAdClickSchema


def build_pyg_graph(df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15) -> Data:
    """
    Constructs a PyTorch Geometric Graph Data Object (T-Finance / GraphNC compatible):
    - Nodes: Click events with 7 normalized behavioral features
    - Edges: Co-occurrence relations across shared IP and App/Channel entities
    - Targets: Binary fraud conversion labels
    - Masks: Chronological train, validation, and test boolean masks
    """
    feature_cols = ["hour", "day", "ip_click_count", "ip_unique_apps", "app_freq", "channel_freq", "device_freq"]
    X_raw = df[feature_cols].values.astype(np.float32)
    X_norm = (X_raw - X_raw.mean(axis=0)) / (X_raw.std(axis=0) + 1e-6)
    x = torch.tensor(X_norm, dtype=torch.float)

    # 1. Build Co-occurrence Edges
    src_list = []
    dst_list = []

    # Relation 1: Shared IP
    ip_groups = df.groupby("ip").groups
    for _, indices in ip_groups.items():
        idx_arr = list(indices)
        if len(idx_arr) > 1:
            for i in range(len(idx_arr) - 1):
                src_list.extend([idx_arr[i], idx_arr[i + 1]])
                dst_list.extend([idx_arr[i + 1], idx_arr[i]])

    # Relation 2: Shared App & Channel (capped to prevent mega-cliques on top apps)
    app_chan_groups = df.groupby(["app", "channel"]).groups
    for _, indices in app_chan_groups.items():
        idx_arr = list(indices)
        if 1 < len(idx_arr) <= 30:
            for i in range(len(idx_arr) - 1):
                src_list.extend([idx_arr[i], idx_arr[i + 1]])
                dst_list.extend([idx_arr[i + 1], idx_arr[i]])

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

    return Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)


def preprocess():
    raw_sample_path = "data/raw/train_sample.csv"
    raw_full_path = "data/raw/train.csv"
    output_dir = "data/processed"
    prod_dir = "data/simulated_production"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(prod_dir, exist_ok=True)

    print("1. Loading raw ad-click dataset...")
    if os.path.exists(raw_sample_path):
        raw_path = raw_sample_path
        print(f"   -> Loading sample dataset from: {raw_path}")
        df = pd.read_csv(raw_path)
    elif os.path.exists(raw_full_path):
        raw_path = raw_full_path
        print(f"   -> Loading full dataset from: {raw_path}")
        df = pd.read_csv(raw_path, nrows=100000)
    else:
        raise FileNotFoundError("No raw ad click CSV found in data/raw/.")

    print(f"   -> Loaded {len(df):,} records.")

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

    # Behavioral frequency aggregations
    df["ip_click_count"] = df.groupby("ip")["app"].transform("count").astype("float32")
    df["ip_unique_apps"] = df.groupby("ip")["app"].transform("nunique").astype("float32")
    df["app_freq"] = df.groupby("app")["ip"].transform("count").astype("float32")
    df["channel_freq"] = df.groupby("channel")["ip"].transform("count").astype("float32")
    df["device_freq"] = df.groupby("device")["ip"].transform("count").astype("float32")

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