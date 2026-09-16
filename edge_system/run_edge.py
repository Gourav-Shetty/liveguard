import argparse
import sys
import time

from edge_system import config
from edge_system.drivers.serial_driver import ArduinoSerialDriver
from edge_system.drivers.mock_driver import MockSensorDriver
from edge_system.signal_processing import RealTimeFilter, PanTompkinsQRSDetector, BeatSegmenter
from edge_system.edge_infer import EdgeInferenceEngine
from edge_system.telemetry_server import TelemetryServer

try:
    from edge_system.drivers.rpi_mcp3008_driver import RPiMCP3008Driver
except Exception:
    RPiMCP3008Driver = None

try:
    from edge_system.drivers.ads1115_driver import ADS1115Driver
except Exception:
    ADS1115Driver = None


def parse_args():
    parser = argparse.ArgumentParser(description="LiveGuard Edge Pipeline")
    parser.add_argument(
        "--source",
        choices=["ARDUINO", "MOCK", "RPI_SPI", "ADS1115"],
        default=config.DATA_SOURCE,
        help="Input data stream source (default: %(default)s)"
    )
    parser.add_argument(
        "--port",
        type=str,
        default=None,
        help="Serial COM port for Arduino (e.g., COM3, /dev/ttyACM0)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=config.DEFAULT_ANOMALY_THRESHOLD,
        help="Anomaly classification threshold (default: %(default)s)"
    )
    parser.add_argument(
        "--no-ws",
        action="store_true",
        help="Disable WebSocket telemetry streaming"
    )
    return parser.parse_args()


def get_driver(source: str, port: str = None):
    if source == "ARDUINO":
        driver = ArduinoSerialDriver(port=port)
        driver.connect()
        return driver
    elif source == "MOCK":
        return MockSensorDriver()
    elif source == "RPI_SPI":
        if RPiMCP3008Driver is None:
            print("[ERROR] RPiMCP3008Driver is only available on Raspberry Pi.")
            sys.exit(1)
        return RPiMCP3008Driver()
    elif source == "ADS1115":
        if ADS1115Driver is None:
            print("[ERROR] ADS1115Driver is only available on Raspberry Pi.")
            sys.exit(1)
        return ADS1115Driver()
    else:
        raise ValueError(f"Unknown data source: {source}")


def main():
    args = parse_args()

    print("=" * 65)
    print(" LiveGuard-EHMS Edge Intelligence Pipeline")
    print(f" Source: {args.source} | Rate: {config.SAMPLING_RATE_ECG} Hz | Window: {config.BEAT_WINDOW_SIZE}")
    print("=" * 65)

    driver = get_driver(args.source, args.port)
    filter_engine = RealTimeFilter(fs=config.SAMPLING_RATE_ECG)
    qrs_detector = PanTompkinsQRSDetector(fs=config.SAMPLING_RATE_ECG)
    segmenter = BeatSegmenter(
        pre_r=config.PRE_R_SAMPLES,
        post_r=config.POST_R_SAMPLES,
        window_size=config.BEAT_WINDOW_SIZE
    )
    infer_engine = EdgeInferenceEngine(threshold=args.threshold)

    ws_server = None
    if not args.no_ws:
        ws_server = TelemetryServer()
        ws_server.start()

    sample_count = 0
    total_beats = 0
    anomalies_detected = 0
    start_time = time.time()

    try:
        while True:
            sample_data = driver.read_sample()
            if sample_data is None:
                continue

            sample_count += 1
            raw_ecg = sample_data["ecg_raw"]
            leads_off = sample_data["leads_off"]

            if leads_off:
                print("\r[WARNING] Leads-off detected!", end="", flush=True)
                continue

            filtered_ecg = filter_engine.process_sample(raw_ecg)
            is_r_peak = qrs_detector.process_sample(filtered_ecg)
            current_hr = qrs_detector.get_heart_rate()

            beat_tensor = segmenter.add_sample(filtered_ecg, is_r_peak)
            prediction_info = None

            if beat_tensor is not None:
                total_beats += 1
                prediction_info = infer_engine.predict_beat(beat_tensor)

                if prediction_info["is_anomaly"]:
                    anomalies_detected += 1
                    status_str = f"[ABNORMAL/ARRHYTHMIA] (Prob: {prediction_info['abnormal_prob']:.2f})"
                else:
                    status_str = f"[NORMAL BEAT] (Conf: {prediction_info['confidence']}%)"

                print(
                    f"Beat #{total_beats:04d} | HR: {current_hr:.1f} BPM | {status_str} "
                    f"| Total Alerts: {anomalies_detected}"
                )

            if ws_server and sample_count % config.STREAM_BATCH_SIZE == 0:
                payload = {
                    "type": "telemetry",
                    "raw_ecg": raw_ecg,
                    "filtered_ecg": round(filtered_ecg, 2),
                    "is_r_peak": is_r_peak,
                    "heart_rate": round(current_hr, 1),
                    "prediction": prediction_info["prediction"] if prediction_info else None,
                    "is_anomaly": prediction_info["is_anomaly"] if prediction_info else False,
                    "timestamp": time.time()
                }
                ws_server.broadcast(payload)

    except KeyboardInterrupt:
        print("\nPipeline stopped.")
    finally:
        driver.close()
        elapsed = time.time() - start_time
        print(f"Elapsed: {elapsed:.1f}s | Beats: {total_beats} | Alerts: {anomalies_detected}")


if __name__ == "__main__":
    main()
