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
    X_raw = df[feature_cols].fillna(0.0).values.astype(np.float32)
    X_raw = np.nan_to_num(X_raw, nan=0.0)
    std_val = X_raw.std(axis=0)
    std_val[std_val < 1e-6] = 1.0
    X_norm = np.nan_to_num((X_raw - X_raw.mean(axis=0)) / std_val, nan=0.0, posinf=0.0, neginf=0.0)
    x = torch.tensor(X_norm, dtype=torch.float)

    # Categorical entity matrix for learnable embeddings (app, channel, device, os, hour)
    cat_cols = ["app", "channel", "device", "os", "hour"]
    x_cat = torch.tensor(np.nan_to_num(df[cat_cols].fillna(0).values, nan=0).astype(np.int64), dtype=torch.long)

    # 1. Build Multi-Relational Co-occurrence Edges (NumPy Vectorized)
    print("   -> Building Multi-Relational Graph Edges (Vectorized)...")
    src_list = []
    dst_list = []

    def build_edges_from_keys(keys: np.ndarray, max_cluster: int = 20):
        order = np.argsort(keys)
        sorted_k = keys[order]
        split_idx = np.where(sorted_k[:-1] != sorted_k[1:])[0] + 1
        groups = np.split(order, split_idx)
        srcs = []
        dsts = []
        for g in groups:
            if len(g) > 1:
                sub = g[:max_cluster]
                srcs.extend(sub[:-1])
                dsts.extend(sub[1:])
                srcs.extend(sub[1:])
                dsts.extend(sub[:-1])
        return srcs, dsts

    # Relation 1: Shared IP (connect consecutive clicks per IP)
    s1, d1 = build_edges_from_keys(df["ip"].values, max_cluster=30)
    src_list.extend(s1)
    dst_list.extend(d1)
    del s1, d1

    # Relation 2: Shared App & Channel (detects campaign target clusters)
    app_chan_arr = (df["app"].values.astype(np.uint32) << 16) | df["channel"].values.astype(np.uint32)
    s2, d2 = build_edges_from_keys(app_chan_arr, max_cluster=20)
    src_list.extend(s2)
    dst_list.extend(d2)
    del app_chan_arr, s2, d2

    # Relation 3: Shared IP & Channel (detects cross-app spamming on same channel)
    ip_chan_arr = (df["ip"].values.astype(np.uint64) << 16) | df["channel"].values.astype(np.uint64)
    s3, d3 = build_edges_from_keys(ip_chan_arr, max_cluster=20)
    src_list.extend(s3)
    dst_list.extend(d3)
    del ip_chan_arr, s3, d3

    # Relation 4: Shared Device Fingerprint (ip, device, os)
    dev_fp_arr = (df["ip"].values.astype(np.uint64) << 32) | (df["device"].values.astype(np.uint64) << 16) | df["os"].values.astype(np.uint64)
    s4, d4 = build_edges_from_keys(dev_fp_arr, max_cluster=20)
    src_list.extend(s4)
    dst_list.extend(d4)
    del dev_fp_arr, s4, d4

    # Self-loops for all nodes
    node_ids = list(range(len(df)))
    src_list.extend(node_ids)
    dst_list.extend(node_ids)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_index = torch.unique(edge_index, dim=1)
    del src_list, dst_list

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

    target_path = input_path or (sys.argv[1] if len(sys.argv) > 1 else None)
    dtypes = {
        "ip": "uint32",
        "app": "uint16",
        "device": "uint16",
        "os": "uint16",
        "channel": "uint16",
        "is_attributed": "uint8"
    }
    usecols = ["ip", "app", "device", "os", "channel", "click_time", "is_attributed"]

    n_rows_env = os.getenv("N_ROWS", "1000000")
    nrows = None if n_rows_env in ["0", "all", "none", "None"] else int(n_rows_env)

    if target_path and os.path.exists(target_path):
        raw_path = target_path
    elif os.path.exists(raw_full_path):
        raw_path = raw_full_path
    elif os.path.exists(raw_sample_path):
        raw_path = raw_sample_path
    else:
        raise FileNotFoundError("No raw ad click CSV found in data/raw/.")

    import gc
    from collections import Counter

    # ==============================================================================
    # MODE 1: Fast Direct Path for Datasets <= 15,000,000 Rows
    # ==============================================================================
    if nrows and nrows <= 15000000:
        print(f"1. Loading dataset ({nrows:,} rows) [Engine: Fast In-Memory SIMD]...")
        df = pd.read_csv(raw_path, nrows=nrows, dtype=dtypes, usecols=usecols)
        print("2. Validating raw click events with Pandera (RawAdClickSchema)...")
        RawAdClickSchema.validate(df)

        print("3. Feature engineering: Entity frequencies & temporal interactions...")
        if not pd.api.types.is_datetime64_any_dtype(df["click_time"]):
            df["dt"] = pd.to_datetime(df["click_time"])
        else:
            df["dt"] = df["click_time"]
        df["event_timestamp"] = df["dt"]
        df["hour"] = df["dt"].dt.hour.astype("uint8")
        df["day"] = df["dt"].dt.day.astype("uint8")
        df["date_str"] = df["dt"].dt.strftime("%Y-%m-%d")

        df = df.sort_values("dt").reset_index(drop=True)
        gc.collect()

        df["ip_click_count"] = df["ip"].map(df["ip"].value_counts()).astype("float32")
        df["app_freq"] = df["app"].map(df["app"].value_counts()).astype("float32")
        df["channel_freq"] = df["channel"].map(df["channel"].value_counts()).astype("float32")
        df["device_freq"] = df["device"].map(df["device"].value_counts()).astype("float32")

        ip_uapps = df.groupby("ip")["app"].nunique().astype("float32")
        df["ip_unique_apps"] = df["ip"].map(ip_uapps).astype("float32")
        del ip_uapps

        ip_uchans = df.groupby("ip")["channel"].nunique().astype("float32")
        df["ip_unique_channels"] = df["ip"].map(ip_uchans).astype("float32")
        del ip_uchans

        app_chan_arr = (df["app"].values.astype(np.uint32) << 16) | df["channel"].values.astype(np.uint32)
        app_chan_key = pd.Series(app_chan_arr)
        df["app_channel_count"] = app_chan_key.map(app_chan_key.value_counts()).astype("float32")
        del app_chan_arr, app_chan_key

        ip_hh_app_arr = (df["ip"].values.astype(np.uint64) << 32) | (df["hour"].values.astype(np.uint64) << 16) | df["app"].values.astype(np.uint64)
        ip_hh_app_key = pd.Series(ip_hh_app_arr)
        df["ip_hh_app_count"] = ip_hh_app_key.map(ip_hh_app_key.value_counts()).astype("float32")
        del ip_hh_app_arr, ip_hh_app_key

        ip_hh_dev_arr = (df["ip"].values.astype(np.uint64) << 32) | (df["hour"].values.astype(np.uint64) << 16) | df["device"].values.astype(np.uint64)
        ip_hh_dev_key = pd.Series(ip_hh_dev_arr)
        df["ip_hh_device_count"] = ip_hh_dev_key.map(ip_hh_dev_key.value_counts()).astype("float32")
        del ip_hh_dev_arr, ip_hh_dev_key

        df["ip_device_os_cumcount"] = np.log1p(df.groupby(["ip", "device", "os"]).cumcount()).astype("float32")
        df["ip_app_cumcount"] = np.log1p(df.groupby(["ip", "app"]).cumcount()).astype("float32")

        next_sec = df.groupby(["ip", "app", "device", "os"])["dt"].shift(-1)
        next_delta = (pd.to_datetime(next_sec) - df["dt"]).dt.total_seconds().fillna(3600.0)
        df["next_click_delta"] = np.log1p(np.clip(next_delta, 0.0, 86400.0)).astype("float32")
        del next_sec, next_delta

        prev_sec = df.groupby(["ip", "channel"])["dt"].shift(1)
        prev_delta = (df["dt"] - pd.to_datetime(prev_sec)).dt.total_seconds().fillna(3600.0)
        df["prev_click_delta"] = np.log1p(np.clip(prev_delta, 0.0, 86400.0)).astype("float32")
        del prev_sec, prev_delta
        gc.collect()

        print("4. Validating processed dataset with Pandera (ProcessedAdClickSchema)...")
        ProcessedAdClickSchema.validate(df)

        print("5. Performing chronological day/hour split...")
        num_records = len(df)
        train_end = int(num_records * 0.70)
        val_end = int(num_records * 0.85)

        train_df = df.iloc[:train_end].copy().drop(columns=["dt", "date_str"])
        val_df = df.iloc[train_end:val_end].copy().drop(columns=["dt", "date_str"])
        test_df = df.iloc[val_end:].copy().drop(columns=["dt", "date_str"])
        drift_df = df.iloc[val_end:].copy()

        train_df.to_parquet(f"{output_dir}/train.parquet", index=False)
        val_df.to_parquet(f"{output_dir}/val.parquet", index=False)
        test_df.to_parquet(f"{output_dir}/test.parquet", index=False)
        drift_df.drop(columns=["dt"]).to_csv(f"{prod_dir}/production_drift_stream.csv", index=False)

        print("6. Constructing and serializing PyTorch Geometric Graph Tensors...")
        graph_data = build_pyg_graph(df, train_ratio=0.70, val_ratio=0.15)
        torch.save(graph_data, f"{output_dir}/ad_click_graph.pt")
        torch.save(graph_data, f"{output_dir}/train_graph.pt")
        torch.save(graph_data, f"{output_dir}/val_graph.pt")
        torch.save(graph_data, f"{output_dir}/test_graph.pt")

        print(f"\n✅ Preprocessing Complete! Total Nodes: {graph_data.num_nodes:,} | Edges: {graph_data.edge_index.shape[1]:,}")
        return

    # ==============================================================================
    # MODE 2: Ultra-Lean Single-Pass Streaming Engine (Zero OOM at 20M - 185M Rows)
    # ==============================================================================
    print(f"1. Starting Ultra-Lean Single-Pass Streaming Engine for: {raw_path}")
    chunk_size = 2500000

    import pyarrow as pa
    import pyarrow.parquet as pq
    from collections import defaultdict

    # Estimate total rows for splitting if nrows given, else count
    total_expected = nrows if nrows else 184903890
    train_end = int(total_expected * 0.70)
    val_end = int(total_expected * 0.85)

    # Parquet Streaming Writers (direct-to-disk, 0 RAM build-up)
    train_writer = None
    val_writer = None
    test_writer = None
    drift_writer = None

    # Persistent Point-in-Time Streaming State Trackers
    ip_counts = {}
    app_counts = {}
    chan_counts = {}
    dev_counts = {}
    app_chan_counts = {}
    ip_hh_app_counts = {}
    ip_hh_dev_counts = {}
    ido_counts = {}
    ia_counts = {}
    ip_app_sets = defaultdict(set)
    ip_chan_sets = defaultdict(set)
    prev_click_ip_chan = {}

    # O(1) Streaming Multi-Relational Edge Trackers
    last_ip = {}
    last_ac = {}
    last_ic = {}
    last_dfp = {}

    # Streamed Node Tensors (lightweight arrays)
    x_chunks = []
    x_cat_chunks = []
    y_chunks = []
    edge_src_chunks = []
    edge_dst_chunks = []

    total_processed = 0
    print("2. Streaming feature extraction, direct-to-disk parquet & O(1) graph edges...")

    feature_cols = [
        "hour", "day", "ip_click_count", "ip_unique_apps", "app_freq", "channel_freq", "device_freq",
        "next_click_delta", "prev_click_delta", "ip_device_os_cumcount", "ip_app_cumcount",
        "ip_hh_app_count", "ip_hh_device_count", "app_channel_count", "ip_unique_channels"
    ]
    cat_cols = ["app", "channel", "device", "os", "hour"]

    for chunk in pd.read_csv(raw_path, chunksize=chunk_size, dtype=dtypes, usecols=usecols):
        N = len(chunk)
        if not pd.api.types.is_datetime64_any_dtype(chunk["click_time"]):
            chunk_dt = pd.to_datetime(chunk["click_time"])
        else:
            chunk_dt = chunk["click_time"]

        chunk["event_timestamp"] = chunk_dt
        hour_arr = chunk_dt.dt.hour.values.astype(np.uint8)
        day_arr = chunk_dt.dt.day.values.astype(np.uint8)
        chunk["hour"] = hour_arr
        chunk["day"] = day_arr

        ip_arr = chunk["ip"].values
        app_arr = chunk["app"].values
        chan_arr = chunk["channel"].values
        dev_arr = chunk["device"].values
        os_arr = chunk["os"].values
        dt_arr = chunk_dt.values

        # Fast bit-shifted composite keys
        app_chan_arr = (app_arr.astype(np.uint32) << 16) | chan_arr.astype(np.uint32)
        ip_hh_app_arr = (ip_arr.astype(np.uint64) << 32) | (hour_arr.astype(np.uint64) << 16) | app_arr.astype(np.uint64)
        ip_hh_dev_arr = (ip_arr.astype(np.uint64) << 32) | (hour_arr.astype(np.uint64) << 16) | dev_arr.astype(np.uint64)
        ido_arr = (ip_arr.astype(np.uint64) << 32) | (dev_arr.astype(np.uint64) << 16) | os_arr.astype(np.uint64)
        ia_arr = (ip_arr.astype(np.uint64) << 16) | app_arr.astype(np.uint64)
        ic_arr = (ip_arr.astype(np.uint64) << 16) | chan_arr.astype(np.uint64)
        ip_app_dev_os_arr = (ip_arr.astype(np.uint64) << 48) | (app_arr.astype(np.uint64) << 32) | (dev_arr.astype(np.uint64) << 16) | os_arr.astype(np.uint64)

        # Pre-allocate output arrays for the chunk
        ip_counts_out = np.empty(N, dtype=np.float32)
        app_freq_out = np.empty(N, dtype=np.float32)
        chan_freq_out = np.empty(N, dtype=np.float32)
        dev_freq_out = np.empty(N, dtype=np.float32)
        app_chan_out = np.empty(N, dtype=np.float32)
        ip_hh_app_out = np.empty(N, dtype=np.float32)
        ip_hh_dev_out = np.empty(N, dtype=np.float32)
        ido_cum_out = np.empty(N, dtype=np.float32)
        ia_cum_out = np.empty(N, dtype=np.float32)
        ip_uapps_out = np.empty(N, dtype=np.float32)
        ip_uchans_out = np.empty(N, dtype=np.float32)
        prev_delta_out = np.full(N, np.log1p(3600.0), dtype=np.float32)
        next_delta_out = np.full(N, np.log1p(3600.0), dtype=np.float32)

        # Single-pass point-in-time stateful extraction + O(1) graph edge generation
        chunk_src_edges = []
        chunk_dst_edges = []
        start_idx = total_processed

        for i in range(N):
            g = start_idx + i
            ip = int(ip_arr[i])
            app = int(app_arr[i])
            chan = int(chan_arr[i])
            dev = int(dev_arr[i])
            ac = int(app_chan_arr[i])
            iha = int(ip_hh_app_arr[i])
            ihd = int(ip_hh_dev_arr[i])
            ido = int(ido_arr[i])
            ia = int(ia_arr[i])
            ic = int(ic_arr[i])
            t = dt_arr[i]

            # 1. Running entity frequencies (point-in-time)
            cnt = ip_counts.get(ip, 0) + 1
            ip_counts[ip] = cnt
            ip_counts_out[i] = cnt

            cnt = app_counts.get(app, 0) + 1
            app_counts[app] = cnt
            app_freq_out[i] = cnt

            cnt = chan_counts.get(chan, 0) + 1
            chan_counts[chan] = cnt
            chan_freq_out[i] = cnt

            cnt = dev_counts.get(dev, 0) + 1
            dev_counts[dev] = cnt
            dev_freq_out[i] = cnt

            # 2. Composite key frequencies
            cnt = app_chan_counts.get(ac, 0) + 1
            app_chan_counts[ac] = cnt
            app_chan_out[i] = cnt

            cnt = ip_hh_app_counts.get(iha, 0) + 1
            ip_hh_app_counts[iha] = cnt
            ip_hh_app_out[i] = cnt

            cnt = ip_hh_dev_counts.get(ihd, 0) + 1
            ip_hh_dev_counts[ihd] = cnt
            ip_hh_dev_out[i] = cnt

            # 3. Cumulative counts
            cum = ido_counts.get(ido, 0)
            ido_counts[ido] = cum + 1
            ido_cum_out[i] = np.log1p(cum)

            cum = ia_counts.get(ia, 0)
            ia_counts[ia] = cum + 1
            ia_cum_out[i] = np.log1p(cum)

            # 4. Running unique sets
            ip_app_sets[ip].add(app)
            ip_uapps_out[i] = len(ip_app_sets[ip])

            ip_chan_sets[ip].add(chan)
            ip_uchans_out[i] = len(ip_chan_sets[ip])

            # 5. prev_click_delta
            if ic in prev_click_ip_chan:
                d = (t - prev_click_ip_chan[ic]) / np.timedelta64(1, 's')
                prev_delta_out[i] = np.log1p(max(0.0, min(float(d), 86400.0)))
            prev_click_ip_chan[ic] = t

            # 6. O(1) Multi-Relational Graph Edge Wiring (connects chronological predecessors)
            if ip in last_ip:
                p = last_ip[ip]
                chunk_src_edges.extend([p, g])
                chunk_dst_edges.extend([g, p])
            last_ip[ip] = g

            if ac in last_ac:
                p = last_ac[ac]
                chunk_src_edges.extend([p, g])
                chunk_dst_edges.extend([g, p])
            last_ac[ac] = g

            if ic in last_ic:
                p = last_ic[ic]
                chunk_src_edges.extend([p, g])
                chunk_dst_edges.extend([g, p])
            last_ic[ic] = g

            dfp = int(ip_app_dev_os_arr[i])
            if dfp in last_dfp:
                p = last_dfp[dfp]
                chunk_src_edges.extend([p, g])
                chunk_dst_edges.extend([g, p])
            last_dfp[dfp] = g

            # Self-loop
            chunk_src_edges.append(g)
            chunk_dst_edges.append(g)

        # Reverse pass for next_click_delta within chunk
        temp_next = {}
        for i in range(N - 1, -1, -1):
            key = int(ip_app_dev_os_arr[i])
            t = dt_arr[i]
            if key in temp_next:
                d = (temp_next[key] - t) / np.timedelta64(1, 's')
                next_delta_out[i] = np.log1p(max(0.0, min(float(d), 86400.0)))
            temp_next[key] = t

        # Assign computed features to chunk
        chunk["ip_click_count"] = ip_counts_out
        chunk["app_freq"] = app_freq_out
        chunk["channel_freq"] = chan_freq_out
        chunk["device_freq"] = dev_freq_out
        chunk["app_channel_count"] = app_chan_out
        chunk["ip_hh_app_count"] = ip_hh_app_out
        chunk["ip_hh_device_count"] = ip_hh_dev_out
        chunk["ip_device_os_cumcount"] = ido_cum_out
        chunk["ip_app_cumcount"] = ia_cum_out
        chunk["ip_unique_apps"] = ip_uapps_out
        chunk["ip_unique_channels"] = ip_uchans_out
        chunk["prev_click_delta"] = prev_delta_out
        chunk["next_click_delta"] = next_delta_out

        end_idx = start_idx + N
        total_processed = end_idx

        # Direct-to-Tensor serialization (continuous features & categorical IDs)
        x_chunk = torch.from_numpy(chunk[feature_cols].values.astype(np.float32))
        x_cat_chunk = torch.from_numpy(chunk[cat_cols].values.astype(np.int64))
        y_chunk = torch.from_numpy(chunk["is_attributed"].values.astype(np.int64))

        x_chunks.append(x_chunk)
        x_cat_chunks.append(x_cat_chunk)
        y_chunks.append(y_chunk)
        edge_src_chunks.append(np.array(chunk_src_edges, dtype=np.int64))
        edge_dst_chunks.append(np.array(chunk_dst_edges, dtype=np.int64))

        # Direct-to-Disk Parquet Stream (Streaming PyArrow Table Writer)
        chunk_clean = chunk.drop(columns=["dt"]) if "dt" in chunk.columns else chunk
        table = pa.Table.from_pandas(chunk_clean)

        if end_idx <= train_end:
            if train_writer is None:
                train_writer = pq.ParquetWriter(f"{output_dir}/train.parquet", table.schema)
            train_writer.write_table(table)
        elif start_idx >= val_end:
            if test_writer is None:
                test_writer = pq.ParquetWriter(f"{output_dir}/test.parquet", table.schema)
            test_writer.write_table(table)
        else:
            if val_writer is None:
                val_writer = pq.ParquetWriter(f"{output_dir}/val.parquet", table.schema)
            val_writer.write_table(table)

        # Free DataFrame chunk from RAM immediately
        del chunk, chunk_clean, table, chunk_src_edges, chunk_dst_edges
        gc.collect()

        print(f"      Streamed {total_processed:,} / {total_expected:,} rows directly to disk & graph tensors...")
        if nrows and total_processed >= nrows:
            break

    # Close all parquet writers
    if train_writer:
        train_writer.close()
    if val_writer:
        val_writer.close()
    if test_writer:
        test_writer.close()

    # Create production drift CSV sample from test set
    print("3. Generating production drift sample...")
    if os.path.exists(f"{output_dir}/test.parquet"):
        test_table = pq.read_table(f"{output_dir}/test.parquet")
        test_sample_df = test_table.slice(0, min(100000, test_table.num_rows)).to_pandas()
        test_sample_df.to_csv(f"{prod_dir}/production_drift_stream.csv", index=False)
        del test_table, test_sample_df

    # Free streaming state dictionaries
    del ip_counts, app_counts, chan_counts, dev_counts, app_chan_counts
    del ip_hh_app_counts, ip_hh_dev_counts, ido_counts, ia_counts
    del ip_app_sets, ip_chan_sets, prev_click_ip_chan
    del last_ip, last_ac, last_ic, last_dfp
    gc.collect()

    print(f"4. Assembling and normalizing complete {total_processed:,}-node GraphNC object...")
    x = torch.cat(x_chunks, dim=0)
    x_cat = torch.cat(x_cat_chunks, dim=0)
    y = torch.cat(y_chunks, dim=0)
    del x_chunks, x_cat_chunks, y_chunks
    gc.collect()

    # Z-score normalize features in-place
    mean_val = x.mean(dim=0, keepdim=True)
    std_val = x.std(dim=0, keepdim=True)
    std_val[std_val < 1e-6] = 1.0
    x = torch.nan_to_num((x - mean_val) / std_val, nan=0.0, posinf=0.0, neginf=0.0)

    # Assemble edge index
    edge_src = np.concatenate(edge_src_chunks)
    edge_dst = np.concatenate(edge_dst_chunks)
    del edge_src_chunks, edge_dst_chunks
    gc.collect()

    edge_index = torch.from_numpy(np.vstack([edge_src, edge_dst])).to(torch.long)
    del edge_src, edge_dst
    gc.collect()

    # Chronological Masks
    num_nodes = total_processed
    actual_train_cutoff = int(num_nodes * 0.70)
    actual_val_cutoff = int(num_nodes * 0.85)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[:actual_train_cutoff] = True
    val_mask[actual_train_cutoff:actual_val_cutoff] = True
    test_mask[actual_val_cutoff:] = True

    graph_data = Data(
        x=x.contiguous(),
        x_cat=x_cat.contiguous(),
        edge_index=edge_index.contiguous(),
        y=y.contiguous(),
        train_mask=train_mask.contiguous(),
        val_mask=val_mask.contiguous(),
        test_mask=test_mask.contiguous()
    )

    print("5. Serializing full graph tensors to disk...")
    torch.save(graph_data, f"{output_dir}/ad_click_graph.pt")
    torch.save(graph_data, f"{output_dir}/train_graph.pt")
    torch.save(graph_data, f"{output_dir}/val_graph.pt")
    torch.save(graph_data, f"{output_dir}/test_graph.pt")

    print(f"\n✅ Streaming Preprocessing Complete (100% Data in Graph)!\n"
          f" - Total Nodes:       {graph_data.num_nodes:,}\n"
          f" - Graph Edges:       {graph_data.edge_index.shape[1]:,}\n"
          f" - Processed Splits:  train.parquet, val.parquet, test.parquet")


if __name__ == "__main__":
    preprocess()