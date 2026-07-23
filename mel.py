"""
墨尔本充电站负荷分析入口。

该脚本直接读取 c:\\dachuang 下已有的 Melbourne 数据文件，完成：
- 站点负荷概览
- 异常站点筛选
- 单站点特征拼接
- 相关性分析

使用方式:
  python mel.py
  python mel.py --data_dir c:\\dachuang --top_k 20
  python mel.py --station 7
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config


DEFAULT_CONFIG = Config()
DEFAULT_DATA_DIR = Path(DEFAULT_CONFIG.data_dir)
OUTPUT_DIR = Path(DEFAULT_CONFIG.output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Melbourne EV charging load analysis")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_CONFIG.data_dir,
                        help="Directory containing volume.csv, weather.csv, e_price.csv, s_price.csv, sites.csv")
    parser.add_argument("--station", type=str, default=None,
                        help="Analyze one station id from volume.csv / weather.csv / price files, e.g. 7")
    parser.add_argument("--top_k", type=int, default=DEFAULT_CONFIG.data.top_k_stations,
                        help="Number of top stations to report")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.seed)
    return parser.parse_args()


def _read_csv(path: Path, index_col=0):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path, index_col=index_col)


def _to_datetime_index(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index()
    return frame


def _normalize_station_id(value) -> str:
    text = str(value)
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _build_time_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    hour = index.hour.to_numpy(dtype=float)
    dayofweek = index.dayofweek.to_numpy(dtype=float)
    month = index.month.to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hour / 24.0),
            "hour_cos": np.cos(2 * np.pi * hour / 24.0),
            "dow_sin": np.sin(2 * np.pi * dayofweek / 7.0),
            "dow_cos": np.cos(2 * np.pi * dayofweek / 7.0),
            "month_sin": np.sin(2 * np.pi * month / 12.0),
            "month_cos": np.cos(2 * np.pi * month / 12.0),
            "is_weekend": (dayofweek >= 5).astype(int),
        },
        index=index,
    )


def load_melbourne_data(data_dir: Path):
    volume = _to_datetime_index(_read_csv(data_dir / "volume.csv"))
    weather = _to_datetime_index(_read_csv(data_dir / "weather.csv"))
    e_price = _to_datetime_index(_read_csv(data_dir / "e_price.csv"))
    s_price = _to_datetime_index(_read_csv(data_dir / "s_price.csv"))
    sites = _read_csv(data_dir / "sites.csv", index_col=None)
    info = _read_csv(data_dir / "info.csv", index_col=None)
    chargers = _read_csv(data_dir / "chargers.csv", index_col=None)

    for frame in (volume, e_price, s_price):
        if frame.columns[0] == "Unnamed: 0":
            frame.drop(columns=[frame.columns[0]], inplace=True)

    return {
        "volume": volume,
        "weather": weather,
        "e_price": e_price,
        "s_price": s_price,
        "sites": sites,
        "info": info,
        "chargers": chargers,
    }


def rank_stations(volume: pd.DataFrame) -> pd.DataFrame:
    stats = pd.DataFrame(
        {
            "mean": volume.mean(axis=0),
            "std": volume.std(axis=0),
            "min": volume.min(axis=0),
            "max": volume.max(axis=0),
            "zero_ratio": volume.eq(0).sum(axis=0) / len(volume),
            "missing_ratio": volume.isna().sum(axis=0) / len(volume),
        }
    )
    stats.index = stats.index.map(_normalize_station_id)
    return stats.sort_values(["mean", "std"], ascending=[False, False])


def build_station_frame(station_id: str, data: dict) -> pd.DataFrame:
    volume = data["volume"]
    weather = data["weather"].copy()
    e_price = data["e_price"].copy()
    s_price = data["s_price"].copy()

    station_key = str(station_id)
    if station_key not in volume.columns:
        raise KeyError(f"Station {station_key} not found in volume.csv")
    if station_key not in e_price.columns:
        raise KeyError(f"Station {station_key} not found in e_price.csv")
    if station_key not in s_price.columns:
        raise KeyError(f"Station {station_key} not found in s_price.csv")

    weather = weather.apply(pd.to_numeric, errors="coerce")
    weather = weather.reindex(volume.index).interpolate(limit_direction="both")
    weather = weather.ffill().bfill().dropna(axis=1, how="all")
    panel = pd.DataFrame(index=volume.index)
    panel["target_volume"] = volume[station_key]
    panel["e_price"] = e_price[station_key]
    panel["s_price"] = s_price[station_key]
    panel = panel.join(weather)
    panel = panel.join(_build_time_features(panel.index))
    return panel


def describe_city(data: dict, data_dir: Path, top_k: int):
    volume = data["volume"]
    ranking = rank_stations(volume)

    print("=" * 72)
    print("Melbourne EV Charging Load Analysis")
    print("=" * 72)
    print(f"Data directory: {data_dir}")
    print(f"Time span: {volume.index.min()} -> {volume.index.max()}")
    print(f"Time steps: {len(volume)}")
    print(f"Stations: {volume.shape[1]}")
    print("")
    print("Top stations by mean load:")
    print(ranking.head(top_k).round(3).to_string())

    low_quality = ranking[(ranking["std"] < 0.01) | (ranking["zero_ratio"] > 0.3)]
    print("")
    print(f"Low-quality stations: {len(low_quality)}")
    if len(low_quality) > 0:
        print(low_quality.head(top_k).round(3).to_string())

    return ranking


def analyze_station(panel: pd.DataFrame, station_id: str):
    print("")
    print(f"Station {station_id} feature snapshot:")
    print(panel.head(3).round(3).to_string())
    print("")
    print("Correlation with target_volume:")
    corr = panel.corr(numeric_only=True)["target_volume"].drop("target_volume").sort_values(key=lambda s: s.abs(), ascending=False)
    print(corr.round(3).head(15).to_string())


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    data = load_melbourne_data(data_dir)
    OUTPUT_DIR.mkdir(exist_ok=True)

    ranking = describe_city(data, data_dir, args.top_k)
    ranking.to_csv(OUTPUT_DIR / "melbourne_station_ranking.csv", encoding="utf-8-sig")

    if args.station is not None:
        station_id = _normalize_station_id(args.station)
    else:
        station_id = ranking.index[0]

    station_frame = build_station_frame(station_id, data)
    station_frame.to_csv(OUTPUT_DIR / f"melbourne_station_{station_id}_panel.csv", encoding="utf-8-sig")
    analyze_station(station_frame, station_id)

    summary_path = OUTPUT_DIR / "melbourne_summary.txt"
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write(f"data_dir={data_dir}\n")
        handle.write(f"time_span={data['volume'].index.min()} -> {data['volume'].index.max()}\n")
        handle.write(f"stations={data['volume'].shape[1]}\n")
        handle.write(f"top_station={station_id}\n")
        handle.write(f"panel_shape={station_frame.shape}\n")

    print("")
    print(f"Saved ranking to: {OUTPUT_DIR / 'melbourne_station_ranking.csv'}")
    print(f"Saved station panel to: {OUTPUT_DIR / f'melbourne_station_{station_id}_panel.csv'}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
